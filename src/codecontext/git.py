import contextlib
import difflib
import hashlib
import json
import os
import re
import select
import signal
import sqlite3
import subprocess
import tempfile
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codecontext.deadline import SyncDeadlineExceeded, check_deadline, remaining_timeout
from codecontext.errors import CodeContextError
from codecontext.identity import HEX_OID_RE
from codecontext.store import Store, inspect_local_git_repository, now_us

if TYPE_CHECKING:
    from codecontext.workspace import RepoDirFds

PYTHON_DIFF_MAX_LINES = 2000
HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
# Each cached revision holds a full path->blob listing, so switching between a handful of
# branches stays fast without letting the table grow with every revision ever prepared.
MAX_CACHED_REVISION_SNAPSHOTS = 8
# A killed fetch keeps none of its objects. An initial import that outlives one sync budget
# resumes as fetches of this many commits each, so every completed step persists.
FETCH_DEPTH_STEP = 1000
FETCH_RESUME_MARKER = "codecontext-fetch-resume"

STATUS_TO_CHANGE_KIND: dict[str, str] = {
    "A": "added",
    "M": "modified",
    "D": "deleted",
    "R": "renamed",
    "C": "copied",
    "T": "type_changed",
    "U": "unmerged",
}


def validate_revision_arg(revision: str) -> str:
    if not revision or len(revision) > 1024:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Revision must be 1..1024 characters.",
            retryable=False,
        )
    if "\x00" in revision or "\n" in revision or "\r" in revision or revision.startswith("-"):
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Revision must not contain NUL, newline, or a leading hyphen.",
            retryable=False,
        )
    return revision


def git_blob_oid(content: bytes, *, oid_len: int = 40) -> str:
    algorithm = "sha1" if oid_len == 40 else "sha256"
    header = f"blob {len(content)}\0".encode("ascii")
    h = hashlib.new(algorithm)
    h.update(header)
    h.update(content)
    return h.hexdigest()


def build_safe_git_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_ATTR_NOSYSTEM": "1",
        "LC_ALL": "C",
    }
    if extra:
        env.update(extra)
    return env


def run_safe_git(
    args: list[str],
    *,
    cwd: Path | None = None,
    git_dir: Path | None = None,
    empty_hooks_dir: Path | None = None,
    stdin_bytes: bytes | None = None,
    timeout: float = 30.0,
    allow_file_protocol: bool = False,
    allow_https_protocol: bool = False,
    extra_env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    cmd = [
        "git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "gc.auto=0",
        "-c",
        "maintenance.auto=false",
        "-c",
        "diff.external=",
        "-c",
        "filter.lfs.smudge=",
        "-c",
        "filter.lfs.clean=",
        "-c",
        "filter.lfs.process=",
        "-c",
        "filter.lfs.required=false",
    ]
    if empty_hooks_dir is not None:
        cmd.extend(["-c", f"core.hooksPath={empty_hooks_dir}"])
    if allow_file_protocol and allow_https_protocol:
        cmd.extend(["-c", "protocol.allow=never", "-c", "protocol.file.allow=always", "-c", "protocol.https.allow=always"])
    elif allow_file_protocol:
        cmd.extend(["-c", "protocol.allow=never", "-c", "protocol.file.allow=always"])
    elif allow_https_protocol:
        cmd.extend(["-c", "protocol.allow=never", "-c", "protocol.https.allow=always"])
    else:
        cmd.extend(["-c", "protocol.allow=never"])

    if git_dir is not None:
        cmd.extend(["--git-dir", str(git_dir)])
    cmd.extend(args)

    env = build_safe_git_env(extra_env)
    with subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    ) as proc:
        try:
            stdout, stderr = proc.communicate(input=stdin_bytes, timeout=remaining_timeout(timeout))
        except subprocess.TimeoutExpired as exc:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            check_deadline()
            raise CodeContextError(
                code="IO_ERROR",
                message="Git command timed out.",
                retryable=True,
                details={"command": args[0] if args else "git"},
            ) from exc
        except BaseException:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise

        ret = proc.returncode
        if check and ret != 0:
            err_tail = stderr.decode("utf-8", errors="replace").strip()[-400:]
            raise CodeContextError(
                code="IO_ERROR",
                message=f"Git command failed (exit {ret}).",
                retryable=False,
                details={"command": args[0] if args else "git", "exit_code": ret, "stderr_tail": err_tail},
            )
        return subprocess.CompletedProcess(cmd, ret, stdout, stderr)


@dataclass(frozen=True)
class TreeEntry:
    mode: int
    obj_type: str
    oid: str
    path: bytes


@dataclass(frozen=True)
class ParsedCommit:
    oid: str
    tree_oid: str
    parent_oids: list[str]
    author_name: str | None
    author_email: str | None
    authored_at_us: int | None
    committer_name: str | None
    committer_email: str | None
    committed_at_us: int | None
    encoding: str | None
    message_bytes: bytes
    message_text: str
    decode_ok: bool
    header_byte_len: int


@dataclass(frozen=True)
class RawChange:
    parent_index: int
    parent_oid: str | None
    old_mode: int | None
    new_mode: int | None
    old_blob_oid: str | None
    new_blob_oid: str | None
    change_kind: str
    rename_score: int | None
    old_path: bytes | None
    new_path: bytes | None


@dataclass(frozen=True)
class ParsedHunk:
    ordinal: int
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    patch_text: str
    patch_bytes: bytes


def parse_ls_tree_z(raw_stdout: bytes) -> list[TreeEntry]:
    """Parse `git ls-tree -r -z --full-tree <commit>` output."""
    if not raw_stdout:
        return []
    entries: list[TreeEntry] = []
    for record in raw_stdout.split(b"\x00"):
        if not record:
            continue
        tab_idx = record.find(b"\t")
        if tab_idx == -1:
            raise CodeContextError(
                code="IO_ERROR",
                message="Malformed ls-tree record without tab separator.",
                retryable=False,
            )
        meta = record[:tab_idx].decode("ascii", errors="strict")
        path_bytes = record[tab_idx + 1 :]
        parts = meta.split(" ")
        if len(parts) != 3:
            raise CodeContextError(
                code="IO_ERROR",
                message="Malformed ls-tree metadata fields.",
                retryable=False,
            )
        mode_str, obj_type, oid = parts
        mode = int(mode_str, 8)
        entries.append(TreeEntry(mode=mode, obj_type=obj_type, oid=oid, path=path_bytes))
    return entries


def _parse_git_ident_line(line: bytes) -> tuple[str | None, str | None, int | None]:
    """Parse `<name> <<email>> <timestamp> <tz>` into (name, email, timestamp_us)."""
    text = line.decode("utf-8", errors="replace").strip()
    gt = text.rfind(">")
    if gt == -1:
        return (text or None, None, None)
    lt = text.rfind("<", 0, gt)
    if lt == -1 or gt < lt:
        return (text or None, None, None)
    name = text[:lt].strip() or None
    email = text[lt + 1 : gt].strip() or None
    rest = text[gt + 1 :].strip().split()
    ts_us: int | None = None
    if rest:
        try:
            ts_sec = int(rest[0])
            ts_us = ts_sec * 1_000_000
        except ValueError:
            ts_us = None
    return (name, email, ts_us)


def parse_raw_commit_object(oid: str, raw_bytes: bytes) -> ParsedCommit:
    """Parse raw commit object bytes, handling folded headers (e.g. gpgsig) until first empty line."""
    sep_idx = raw_bytes.find(b"\n\n")
    if sep_idx == -1:
        header_part = raw_bytes
        message_bytes = b""
        header_byte_len = len(raw_bytes)
    else:
        header_part = raw_bytes[:sep_idx]
        message_bytes = raw_bytes[sep_idx + 2 :]
        header_byte_len = sep_idx + 2

    tree_oid = ""
    parent_oids: list[str] = []
    author_name: str | None = None
    author_email: str | None = None
    authored_at_us: int | None = None
    committer_name: str | None = None
    committer_email: str | None = None
    committed_at_us: int | None = None
    encoding: str | None = None

    raw_lines = header_part.split(b"\n")
    unfolded: list[bytes] = []
    for line in raw_lines:
        if line.startswith(b" ") and unfolded:
            unfolded[-1] = unfolded[-1] + b"\n" + line[1:]
        else:
            unfolded.append(line)

    for hline in unfolded:
        if hline.startswith(b"tree "):
            tree_oid = hline[5:].decode("ascii", errors="strict").strip()
        elif hline.startswith(b"parent "):
            parent_oids.append(hline[7:].decode("ascii", errors="strict").strip())
        elif hline.startswith(b"author "):
            author_name, author_email, authored_at_us = _parse_git_ident_line(hline[7:])
        elif hline.startswith(b"committer "):
            committer_name, committer_email, committed_at_us = _parse_git_ident_line(hline[10:])
        elif hline.startswith(b"encoding "):
            encoding = hline[9:].decode("ascii", errors="replace").strip()

    if not tree_oid:
        raise CodeContextError(
            code="IO_ERROR",
            message=f"Commit {oid} missing tree header.",
            retryable=False,
        )

    decode_ok = True
    try:
        if encoding and encoding.lower() not in ("utf-8", "utf8"):
            message_text = message_bytes.decode(encoding, errors="strict")
        else:
            message_text = message_bytes.decode("utf-8", errors="strict")
    except (LookupError, UnicodeDecodeError):
        decode_ok = False
        message_text = message_bytes.decode("utf-8", errors="replace")

    return ParsedCommit(
        oid=oid,
        tree_oid=tree_oid,
        parent_oids=parent_oids,
        author_name=author_name,
        author_email=author_email,
        authored_at_us=authored_at_us,
        committer_name=committer_name,
        committer_email=committer_email,
        committed_at_us=committed_at_us,
        encoding=encoding,
        message_bytes=message_bytes,
        message_text=message_text,
        decode_ok=decode_ok,
        header_byte_len=header_byte_len,
    )


def _normalize_oid_or_null(oid: str) -> str | None:
    if not oid or all(c == "0" for c in oid):
        return None
    return oid


def _normalize_mode_or_null(mode_str: str) -> int | None:
    val = int(mode_str, 8)
    return None if val == 0 else val


def parse_diff_tree_raw_z(
    raw_stdout: bytes,
    *,
    parent_index: int,
    parent_oid: str | None,
) -> list[RawChange]:
    """Parse `git diff-tree --no-commit-id --raw -r -z --no-abbrev ...` output."""
    tokens = _split_z_tokens(raw_stdout)
    changes, end = _parse_raw_z_records(tokens, 0, parent_index=parent_index, parent_oid=parent_oid)
    if end != len(tokens):
        raise CodeContextError(
            code="IO_ERROR",
            message="Malformed diff-tree metadata token.",
            retryable=False,
        )
    return changes


def _split_z_tokens(raw_stdout: bytes) -> list[bytes]:
    tokens = raw_stdout.split(b"\x00") if raw_stdout else []
    if tokens and tokens[-1] == b"":
        tokens.pop()
    return tokens


def _parse_raw_z_records(
    tokens: list[bytes],
    start: int,
    *,
    parent_index: int,
    parent_oid: str | None,
) -> tuple[list[RawChange], int]:
    """Parse raw records from `tokens[start:]` up to the first token that is not a `:` record.
    Returns the changes and the index of that token. Path tokens are consumed by position, so a
    path that looks like an OID is never mistaken for a `--stdin` commit header.
    """
    changes: list[RawChange] = []
    i = start
    n = len(tokens)
    while i < n:
        meta_tok = tokens[i]
        if not meta_tok.startswith(b":"):
            break
        i += 1
        meta_str = meta_tok[1:].decode("ascii", errors="strict")
        parts = meta_str.split(" ")
        if len(parts) < 5:
            raise CodeContextError(
                code="IO_ERROR",
                message="Malformed diff-tree metadata fields.",
                retryable=False,
            )
        old_mode_s, new_mode_s, old_oid_s, new_oid_s, status_tok = parts[:5]
        status_letter = status_tok[0]
        score_str = status_tok[1:] if len(status_tok) > 1 else ""
        rename_score = int(score_str) if score_str.isdigit() else None
        change_kind = STATUS_TO_CHANGE_KIND.get(status_letter, "unknown")

        old_mode = _normalize_mode_or_null(old_mode_s)
        new_mode = _normalize_mode_or_null(new_mode_s)
        old_blob = _normalize_oid_or_null(old_oid_s)
        new_blob = _normalize_oid_or_null(new_oid_s)

        if status_letter in ("R", "C"):
            if i + 1 >= n:
                raise CodeContextError(
                    code="IO_ERROR",
                    message="Truncated diff-tree rename/copy paths.",
                    retryable=False,
                )
            old_path: bytes | None = tokens[i]
            new_path: bytes | None = tokens[i + 1]
            i += 2
        else:
            if i >= n:
                raise CodeContextError(
                    code="IO_ERROR",
                    message="Truncated diff-tree path token.",
                    retryable=False,
                )
            p = tokens[i]
            i += 1
            if change_kind == "added":
                old_path = None
                new_path = p
            elif change_kind == "deleted":
                old_path = p
                new_path = None
            else:
                old_path = p
                new_path = p

        changes.append(
            RawChange(
                parent_index=parent_index,
                parent_oid=parent_oid,
                old_mode=old_mode,
                new_mode=new_mode,
                old_blob_oid=old_blob,
                new_blob_oid=new_blob,
                change_kind=change_kind,
                rename_score=rename_score,
                old_path=old_path,
                new_path=new_path,
            )
        )
    return changes, i


def _split_git_lines(raw_bytes: bytes | None) -> list[bytes]:
    """Split bytes on LF (`\\n`) matching Git's line definition, preserving trailing `\\n`."""
    if not raw_bytes:
        return []
    parts = raw_bytes.split(b"\n")
    if parts[-1] == b"":
        return [p + b"\n" for p in parts[:-1]]
    return [*(p + b"\n" for p in parts[:-1]), parts[-1]]


def compute_unified_zero_hunks(
    old_bytes: bytes | None,
    new_bytes: bytes | None,
    *,
    mirror: "GitMirror | None" = None,
    old_oid: str | None = None,
    new_oid: str | None = None,
) -> list[ParsedHunk] | None:
    """Compute unified=0 hunks between two blobs.

    Small blobs are diffed in-process. Above `PYTHON_DIFF_MAX_LINES` lines SequenceMatcher can go
    cubic, so the diff goes to git's xdiff when both sides are in the mirror's object store (a side
    with no OID must be empty), and returns None (hunks unavailable) when they are not.
    """
    check_deadline()
    if (old_bytes or b"") == (new_bytes or b""):
        return []
    old_lines = _split_git_lines(old_bytes)
    new_lines = _split_git_lines(new_bytes)
    if max(len(old_lines), len(new_lines)) > PYTHON_DIFF_MAX_LINES:
        if mirror is None or (old_oid is None and old_lines) or (new_oid is None and new_lines):
            return None
        return mirror.diff_blobs_unified_zero(old_oid, new_oid)
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    hunks: list[ParsedHunk] = []
    ordinal = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        check_deadline()
        old_count = i2 - i1
        new_count = j2 - j1
        old_start = i1 if old_count == 0 else i1 + 1
        new_start = j1 if new_count == 0 else j1 + 1
        old_range = f"{old_start}" if old_count == 1 else f"{old_start},{old_count}"
        new_range = f"{new_start}" if new_count == 1 else f"{new_start},{new_count}"
        hunk_lines: list[bytes] = [f"@@ -{old_range} +{new_range} @@\n".encode("ascii")]
        for line in old_lines[i1:i2]:
            if line.endswith(b"\n"):
                hunk_lines.append(b"-" + line)
            else:
                hunk_lines.append(b"-" + line + b"\n\\ No newline at end of file\n")
        for line in new_lines[j1:j2]:
            if line.endswith(b"\n"):
                hunk_lines.append(b"+" + line)
            else:
                hunk_lines.append(b"+" + line + b"\n\\ No newline at end of file\n")
        patch_bytes = b"".join(hunk_lines)
        hunks.append(
            ParsedHunk(
                ordinal=ordinal,
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                patch_text=patch_bytes.decode("utf-8", errors="replace"),
                patch_bytes=patch_bytes,
            )
        )
        ordinal += 1
    return hunks


def parse_unified_zero_diff(raw_diff: bytes) -> list[ParsedHunk]:
    """Parse `git diff ... --unified=0 <old_blob> <new_blob> --` output into hunks.
    Validates counted body lines and ignores `\\ No newline at end of file` markers.
    """
    if not raw_diff:
        return []
    lines = _split_git_lines(raw_diff)
    hunks: list[ParsedHunk] = []
    i = 0
    n = len(lines)

    while i < n and not lines[i].startswith(b"@@ "):
        i += 1

    ordinal = 0
    while i < n:
        header_line = lines[i]
        header_str = header_line.decode("utf-8", errors="replace")
        m = HUNK_HEADER_RE.match(header_str)
        if not m:
            raise CodeContextError(
                code="IO_ERROR",
                message=f"Malformed unified=0 hunk header: {header_str[:80]!r}",
                retryable=False,
            )
        old_start = int(m.group(1))
        old_count = int(m.group(2)) if m.group(2) is not None else 1
        new_start = int(m.group(3))
        new_count = int(m.group(4)) if m.group(4) is not None else 1

        # Drop git's function-context suffix so headers match the in-process diff.
        hunk_lines = [m.group(0).encode("ascii") + b"\n"]
        i += 1
        seen_old = 0
        seen_new = 0
        while i < n and not lines[i].startswith(b"@@ "):
            bline = lines[i]
            if bline.startswith(b"\\"):
                hunk_lines.append(bline)
                i += 1
                continue
            if bline.startswith(b"-"):
                seen_old += 1
                hunk_lines.append(bline)
            elif bline.startswith(b"+"):
                seen_new += 1
                hunk_lines.append(bline)
            elif bline.startswith(b" "):
                seen_old += 1
                seen_new += 1
                hunk_lines.append(bline)
            else:
                raise CodeContextError(
                    code="IO_ERROR",
                    message="Unexpected line prefix in unified=0 diff.",
                    retryable=False,
                )
            i += 1

        if seen_old != old_count or seen_new != new_count:
            raise CodeContextError(
                code="IO_ERROR",
                message=f"Inconsistent hunk line counts: expected -{old_count}/+{new_count}, got -{seen_old}/+{seen_new}.",
                retryable=False,
            )

        patch_bytes = b"".join(hunk_lines)
        patch_text = patch_bytes.decode("utf-8", errors="replace")
        hunks.append(
            ParsedHunk(
                ordinal=ordinal,
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                patch_text=patch_text,
                patch_bytes=patch_bytes,
            )
        )
        ordinal += 1

    return hunks


class CatFileBatchSession:
    """Long-lived buffered `git cat-file --batch` session reading length-framed objects."""

    def __init__(self, git_dir: Path, empty_hooks_dir: Path) -> None:
        self.git_dir = git_dir
        self.empty_hooks_dir = empty_hooks_dir
        self._proc: subprocess.Popen[bytes] | None = None
        self._buf = bytearray()
        self._buf_pos = 0

    def _reset_buffer(self) -> None:
        self._buf.clear()
        self._buf_pos = 0

    def _compact_buffer_if_needed(self) -> None:
        if self._buf_pos > 0 and (self._buf_pos >= 32768 or self._buf_pos == len(self._buf)):
            del self._buf[: self._buf_pos]
            self._buf_pos = 0

    def _ensure_proc(self) -> subprocess.Popen[bytes]:
        if self._proc is None or self._proc.poll() is not None:
            self._reset_buffer()
            cmd = [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "gc.auto=0",
                "-c",
                "maintenance.auto=false",
                "-c",
                f"core.hooksPath={self.empty_hooks_dir}",
                "-c",
                "protocol.allow=never",
                "--git-dir",
                str(self.git_dir),
                "cat-file",
                "--batch",
            ]
            self._proc = subprocess.Popen(
                cmd,
                env=build_safe_git_env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                start_new_session=True,
            )
        return self._proc

    def _fill_chunk(self) -> bool:
        assert self._proc is not None and self._proc.stdout is not None
        ready, _, _ = select.select([self._proc.stdout], [], [], remaining_timeout(30.0))
        if not ready:
            self.close()
            check_deadline()
            raise CodeContextError(code="IO_ERROR", message="Git object read timed out.", retryable=True)
        chunk = os.read(self._proc.stdout.fileno(), 65536)
        if not chunk:
            return False
        self._compact_buffer_if_needed()
        self._buf.extend(chunk)
        return True

    def _readline(self) -> bytes:
        while True:
            nl = self._buf.find(b"\n", self._buf_pos)
            if nl != -1:
                end = nl + 1
                line = bytes(self._buf[self._buf_pos : end])
                self._buf_pos = end
                self._compact_buffer_if_needed()
                return line
            if not self._fill_chunk():
                line = bytes(self._buf[self._buf_pos :])
                self._reset_buffer()
                return line

    def _read(self, size: int) -> bytes:
        if size <= 0:
            return b""
        avail = len(self._buf) - self._buf_pos
        if avail >= size:
            end = self._buf_pos + size
            out = bytes(self._buf[self._buf_pos : end])
            self._buf_pos = end
            self._compact_buffer_if_needed()
            return out

        data = bytearray()
        if avail > 0:
            data.extend(self._buf[self._buf_pos :])
            self._reset_buffer()

        assert self._proc is not None and self._proc.stdout is not None
        while len(data) < size:
            ready, _, _ = select.select([self._proc.stdout], [], [], remaining_timeout(30.0))
            if not ready:
                self.close()
                check_deadline()
                raise CodeContextError(code="IO_ERROR", message="Git object read timed out.", retryable=True)
            chunk = os.read(self._proc.stdout.fileno(), min(size - len(data), 65536))
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data)

    def _discard(self, size: int) -> int:
        if size <= 0:
            return 0
        avail = len(self._buf) - self._buf_pos
        if avail >= size:
            self._buf_pos += size
            self._compact_buffer_if_needed()
            return size

        discarded = 0
        if avail > 0:
            discarded += avail
            self._reset_buffer()

        assert self._proc is not None and self._proc.stdout is not None
        while discarded < size:
            ready, _, _ = select.select([self._proc.stdout], [], [], remaining_timeout(30.0))
            if not ready:
                self.close()
                check_deadline()
                raise CodeContextError(code="IO_ERROR", message="Git object read timed out.", retryable=True)
            chunk = os.read(self._proc.stdout.fileno(), min(size - discarded, 65536))
            if not chunk:
                break
            discarded += len(chunk)
        return discarded

    def _write_oid_request(self, oid: str) -> None:
        payload = oid.encode("ascii") + b"\n"
        for attempt in range(2):
            proc = self._ensure_proc()
            assert proc.stdin is not None
            try:
                proc.stdin.write(payload)
                proc.stdin.flush()
                return
            except (BrokenPipeError, OSError) as exc:
                self.close()
                if attempt == 1:
                    raise CodeContextError(
                        code="IO_ERROR",
                        message="Failed writing request to git cat-file --batch.",
                        retryable=True,
                    ) from exc

    def _read_object_impl(self, oid: str, *, max_bytes: int | None = None) -> tuple[str, int, bytes | None] | None:
        if not HEX_OID_RE.match(oid):
            raise CodeContextError(
                code="INVALID_REQUEST",
                message=f"Invalid Git object ID for cat-file: {oid!r}",
                retryable=False,
            )
        check_deadline()
        self._write_oid_request(oid)

        header_line = self._readline()
        if not header_line:
            self.close()
            raise CodeContextError(
                code="IO_ERROR",
                message="EOF reading from git cat-file --batch.",
                retryable=False,
            )
        header_str = header_line.decode("ascii", errors="replace").strip()
        if header_str.endswith(" missing"):
            return None
        parts = header_str.split(" ")
        if len(parts) < 3:
            self.close()
            raise CodeContextError(
                code="IO_ERROR",
                message=f"Unexpected cat-file --batch header: {header_str!r}",
                retryable=False,
            )
        _, obj_type, size_str = parts[:3]
        size = int(size_str)

        if max_bytes is not None and size > max_bytes:
            discarded = self._discard(size)
            if discarded != size:
                self.close()
                raise CodeContextError(
                    code="IO_ERROR",
                    message=f"Truncated cat-file --batch object payload for {oid}",
                    retryable=False,
                )
            framing = self._read(1)
            if framing != b"\n":
                self.close()
                raise CodeContextError(
                    code="IO_ERROR",
                    message="Missing framing newline in cat-file --batch.",
                    retryable=False,
                )
            return obj_type, size, None

        data = self._read(size)
        if len(data) != size:
            self.close()
            raise CodeContextError(
                code="IO_ERROR",
                message=f"Truncated cat-file --batch object payload for {oid}",
                retryable=False,
            )
        framing = self._read(1)
        if framing != b"\n":
            self.close()
            raise CodeContextError(
                code="IO_ERROR",
                message="Missing framing newline in cat-file --batch.",
                retryable=False,
            )
        return obj_type, size, data

    def read_object(self, oid: str) -> tuple[str, int, bytes] | None:
        """Return (obj_type, size, content_bytes) or None if missing."""
        res = self._read_object_impl(oid, max_bytes=None)
        if res is None:
            return None
        obj_type, size, data = res
        assert data is not None
        return obj_type, size, data

    def read_object_bounded(self, oid: str, *, max_bytes: int | None = None) -> tuple[str, int, bytes | None] | None:
        """Return (obj_type, size, content_bytes_or_None_if_oversized) or None if missing."""
        if max_bytes is None:
            return self.read_object(oid)
        return self._read_object_impl(oid, max_bytes=max_bytes)

    def close(self) -> None:
        self._reset_buffer()
        if self._proc is not None:
            proc = self._proc
            self._proc = None
            with contextlib.suppress(OSError):
                if proc.stdin:
                    proc.stdin.close()
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                # The child leads its own session, so this also reaps anything it forked.
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(proc.pid, signal.SIGKILL)
                with contextlib.suppress(Exception):
                    proc.kill()
                    proc.wait(timeout=1.0)

    def __enter__(self) -> "CatFileBatchSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class _PooledCatFileBatchSession(CatFileBatchSession):
    """Thread-locked reusable `CatFileBatchSession` kept alive across leases until `GitMirror.close()`."""

    def __init__(self, git_dir: Path, empty_hooks_dir: Path) -> None:
        super().__init__(git_dir, empty_hooks_dir)
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        return self._lock.acquire(blocking=False)

    def release_lease(self, *, close_proc: bool = False) -> None:
        try:
            if close_proc:
                self.close()
        finally:
            with contextlib.suppress(RuntimeError):
                self._lock.release()

    def __enter__(self) -> "CatFileBatchSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release_lease(close_proc=exc_type is not None)


class GitMirror:
    """Manages owned bare `repository.git/` and safe operations against the source repository."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.git_dir = store.git_dir
        self.empty_hooks_dir = Path(os.devnull)
        self._empty_tree_oid: str | None = None
        self._empty_blob_oid: str | None = None
        self._pooled_batch = _PooledCatFileBatchSession(self.git_dir, self.empty_hooks_dir)
        if hasattr(store, "register_close_callback"):
            store.register_close_callback(self.close)

    def close(self) -> None:
        self._pooled_batch.close()

    def verify_source_binding(self, workspace_override: Path | None = None) -> Path:
        """Verify the local source repository still matches store_meta's common_dir binding."""
        target_repo = workspace_override or (Path(self.store.config.repo_path) if self.store.config.repo_path else None)
        if target_repo is None:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Store has no bound local source repository.",
                retryable=False,
            )
        resolved_repo, common_real, st_dev, st_ino, obj_fmt = inspect_local_git_repository(target_repo)
        remount_dev_changed = False
        with self.store.connect_read() as conn:
            meta = conn.execute(
                "SELECT common_dir_realpath, common_dir_device, common_dir_inode, object_format "
                "FROM store_meta WHERE singleton = 1"
            ).fetchone()
            if not meta:
                raise CodeContextError(
                    code="STORE_NOT_INITIALIZED",
                    message="Missing store_meta row.",
                    retryable=False,
                )
            if (
                meta["common_dir_realpath"] != str(common_real)
                or int(meta["common_dir_inode"]) != st_ino
                or meta["object_format"] != obj_fmt
            ):
                raise CodeContextError(
                    code="REPOSITORY_MISMATCH",
                    message="Source repository common-dir identity changed or does not match store binding.",
                    retryable=False,
                )
            if meta["common_dir_device"] is not None and int(meta["common_dir_device"]) != st_dev:
                remount_dev_changed = True
        if remount_dev_changed and not self.store.read_only:
            with contextlib.suppress(Exception), self.store.write_transaction(bump_generation=False) as conn:
                conn.execute("UPDATE store_meta SET common_dir_device = ? WHERE singleton = 1", (st_dev,))
        return resolved_repo

    def ensure_empty_objects(self) -> tuple[str, str]:
        """Create and retain empty-tree and empty-blob objects using the repository's actual hash format."""
        if self._empty_tree_oid and self._empty_blob_oid:
            return self._empty_tree_oid, self._empty_blob_oid

        blob_proc = run_safe_git(
            ["hash-object", "-w", "-t", "blob", "--stdin"],
            git_dir=self.git_dir,
            empty_hooks_dir=self.empty_hooks_dir,
            stdin_bytes=b"",
        )
        empty_blob = blob_proc.stdout.decode("ascii").strip()

        tree_proc = run_safe_git(
            ["hash-object", "-w", "-t", "tree", "--stdin"],
            git_dir=self.git_dir,
            empty_hooks_dir=self.empty_hooks_dir,
            stdin_bytes=b"",
        )
        empty_tree = tree_proc.stdout.decode("ascii").strip()
        self._empty_tree_oid = empty_tree
        self._empty_blob_oid = empty_blob
        return empty_tree, empty_blob

    def is_source_shallow(self, source_repo: Path) -> bool:
        proc = run_safe_git(
            ["rev-parse", "--is-shallow-repository"],
            cwd=source_repo,
            empty_hooks_dir=self.empty_hooks_dir,
        )
        return proc.stdout.decode("ascii", errors="replace").strip().lower() == "true"

    def resolve_source_revision(self, source_repo: Path, revision: str) -> str:
        validate_revision_arg(revision)
        try:
            proc = run_safe_git(
                ["rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
                cwd=source_repo,
                empty_hooks_dir=self.empty_hooks_dir,
            )
        except CodeContextError as exc:
            raise CodeContextError(
                code="RECORD_NOT_FOUND",
                message=f"Could not resolve revision {revision!r} in source repository.",
                retryable=False,
                details={"revision": revision[:200]},
            ) from exc
        oid = proc.stdout.decode("ascii").strip()
        if not HEX_OID_RE.match(oid):
            raise CodeContextError(
                code="IO_ERROR",
                message=f"Unexpected resolved commit OID: {oid!r}",
                retryable=False,
            )
        return oid

    def enumerate_source_refs(self, source_repo: Path) -> dict[str, str]:
        """Return declared commit-ish source refs: refs/heads/*, refs/remotes/*, refs/tags/*, and HEAD.
        Non-commit tags (pointing to trees or blobs) are filtered out so `rev-list` never fails.
        """
        proc = run_safe_git(
            [
                "for-each-ref",
                "--format=%(refname)%00%(objectname)%00%(objecttype)%00%(*objectname)%00%(*objecttype)",
                "refs/heads/",
                "refs/remotes/",
                "refs/tags/",
            ],
            cwd=source_repo,
            empty_hooks_dir=self.empty_hooks_dir,
        )
        refs: dict[str, str] = {}
        for line in proc.stdout.splitlines():
            if not line:
                continue
            parts = line.split(b"\x00")
            if len(parts) < 3:
                continue
            refname = parts[0].decode("utf-8", errors="replace")
            oid = parts[1].decode("ascii", errors="replace").strip()
            obj_type = parts[2].decode("ascii", errors="replace").strip()
            peeled = parts[3].decode("ascii", errors="replace").strip() if len(parts) > 3 else ""
            peeled_type = parts[4].decode("ascii", errors="replace").strip() if len(parts) > 4 else ""

            if peeled and HEX_OID_RE.match(peeled):
                if peeled_type == "commit":
                    refs[refname] = peeled
            elif obj_type == "commit" and HEX_OID_RE.match(oid):
                refs[refname] = oid

        try:
            head_oid = self.resolve_source_revision(source_repo, "HEAD")
            refs["HEAD"] = head_oid
        except CodeContextError:
            pass
        return refs

    def pin_retained_oid(self, oid: str) -> None:
        self.pin_retained_oids({oid})

    def pin_retained_oids(self, oids: set[str]) -> None:
        oids = {oid for oid in oids if HEX_OID_RE.match(oid)}
        if not oids:
            return
        retained = run_safe_git(
            ["for-each-ref", "--format=%(refname)%00%(objectname)", "refs/codecontext/retained/"],
            git_dir=self.git_dir,
            empty_hooks_dir=self.empty_hooks_dir,
        )
        for line in retained.stdout.splitlines():
            refname, _, oid = line.decode("ascii", errors="replace").partition("\x00")
            if refname == f"refs/codecontext/retained/{oid}":
                oids.discard(oid)
        missing = sorted(oids)
        for start in range(0, len(missing), 128):
            check_deadline()
            updates = "".join(
                f"update refs/codecontext/retained/{oid} {oid}\n"
                for oid in missing[start : start + 128]
            )
            run_safe_git(
                ["update-ref", "--stdin"],
                stdin_bytes=updates.encode("ascii"),
                git_dir=self.git_dir,
                empty_hooks_dir=self.empty_hooks_dir,
            )

    def unpin_retained_oid(self, oid: str) -> None:
        if not HEX_OID_RE.match(oid):
            return
        run_safe_git(
            ["update-ref", "-d", f"refs/codecontext/retained/{oid}"],
            git_dir=self.git_dir,
            empty_hooks_dir=self.empty_hooks_dir,
            check=False,
        )

    def import_local_refs_and_revision(
        self,
        source_repo: Path,
        requested_revision: str,
        *,
        # The sync deadline is the real bound; this only caps a fetch made outside any sync.
        timeout: float = 86_400.0,
    ) -> tuple[str, dict[str, str], bool]:
        """Import declared local refs and the requested revision into owned `refs/codecontext/local/*` namespaces.
        Returns (resolved_commit_oid, ref_map, is_shallow).
        """
        is_shallow = self.is_source_shallow(source_repo)
        resolved_oid = self.resolve_source_revision(source_repo, requested_revision)
        source_refs = self.enumerate_source_refs(source_repo)

        # Pin current tips first: once their refs are pruned, gc could drop commits we still index.
        existing_proc = run_safe_git(
            ["for-each-ref", "--format=%(refname)%00%(objectname)%00%(objecttype)", "refs/codecontext/local/"],
            git_dir=self.git_dir,
            empty_hooks_dir=self.empty_hooks_dir,
        )
        old_oids: set[str] = set()
        expected_mirror_refs: set[str] = set()
        for src_ref in source_refs:
            if src_ref == "HEAD":
                expected_mirror_refs.add("refs/codecontext/local/HEAD")
            elif src_ref.startswith("refs/"):
                expected_mirror_refs.add("refs/codecontext/local/" + src_ref[5:])

        stale_mirror_refs: list[bytes] = []
        for line in existing_proc.stdout.splitlines():
            parts = line.split(b"\x00")
            if len(parts) >= 2:
                m_ref_b = parts[0]
                m_ref = m_ref_b.decode("utf-8", errors="replace")
                old_oid = parts[1].decode("ascii", errors="replace").strip()
                obj_type = parts[2].decode("ascii", errors="replace").strip() if len(parts) > 2 else "commit"
                if obj_type == "commit":
                    old_oids.add(old_oid)
                if m_ref not in expected_mirror_refs:
                    stale_mirror_refs.append(m_ref_b)

        self.pin_retained_oids(old_oids)

        # A stale `refs/codecontext/local/a` blocks fetching a new `a/b` ref (directory/file conflict).
        for start in range(0, len(stale_mirror_refs), 128):
            check_deadline()
            del_payload = b"".join(
                b"delete " + ref_b + b"\x00\x00"
                for ref_b in stale_mirror_refs[start : start + 128]
            )
            run_safe_git(
                ["update-ref", "--stdin", "-z"],
                stdin_bytes=del_payload,
                git_dir=self.git_dir,
                empty_hooks_dir=self.empty_hooks_dir,
                check=False,
            )

        refspecs = [
            "+refs/heads/*:refs/codecontext/local/heads/*",
            "+refs/remotes/*:refs/codecontext/local/remotes/*",
            "+refs/tags/*:refs/codecontext/local/tags/*",
            f"+{resolved_oid}:refs/codecontext/retained/{resolved_oid}",
        ]
        if "HEAD" in source_refs:
            head_oid = source_refs["HEAD"]
            refspecs.append(f"+{head_oid}:refs/codecontext/local/HEAD")

        resume_marker = self.git_dir / FETCH_RESUME_MARKER
        mirror_shallow = not is_shallow and self.is_source_shallow(self.git_dir)
        stepping = resume_marker.exists() and not is_shallow and (mirror_shallow or not old_oids)
        if stepping:
            depth_args = [f"--deepen={FETCH_DEPTH_STEP}" if mirror_shallow else f"--depth={FETCH_DEPTH_STEP}"]
        else:
            depth_args = ["--unshallow"] if mirror_shallow else []
        while True:
            try:
                run_safe_git(
                    [
                        "fetch",
                        "--no-tags",
                        "--no-write-fetch-head",
                        "--no-auto-maintenance",
                        "--update-shallow",
                        *depth_args,
                        "--end-of-options",
                        str(source_repo),
                        *refspecs,
                    ],
                    git_dir=self.git_dir,
                    empty_hooks_dir=self.empty_hooks_dir,
                    allow_file_protocol=True,
                    timeout=timeout,
                )
            except SyncDeadlineExceeded:
                # Only an empty mirror may go shallow: a depth fetch into existing history would
                # hide the commits already imported.
                if not old_oids and not is_shallow:
                    resume_marker.touch()
                raise
            if not stepping or not self.is_source_shallow(self.git_dir):
                break
            depth_args = [f"--deepen={FETCH_DEPTH_STEP}"]
        resume_marker.unlink(missing_ok=True)
        self.ensure_empty_objects()
        return resolved_oid, source_refs, is_shallow

    def list_reachable_commits_with_parents(self, tip_oids: list[str]) -> list[tuple[str, list[str]]]:
        """Run `git rev-list --topo-order --parents --stdin` for validated tip OIDs."""
        valid_tips = [oid for oid in dict.fromkeys(tip_oids) if HEX_OID_RE.match(oid)]
        if not valid_tips:
            return []
        stdin_data = ("\n".join(valid_tips) + "\n").encode("ascii")
        proc = run_safe_git(
            ["rev-list", "--topo-order", "--parents", "--stdin"],
            git_dir=self.git_dir,
            empty_hooks_dir=self.empty_hooks_dir,
            stdin_bytes=stdin_data,
        )
        result: list[tuple[str, list[str]]] = []
        for line in proc.stdout.splitlines():
            if not line:
                continue
            parts = line.decode("ascii", errors="strict").strip().split(" ")
            commit_oid = parts[0]
            parents = parts[1:]
            result.append((commit_oid, parents))
        return result

    def list_tree_entries(self, commit_oid: str) -> list[TreeEntry]:
        if not HEX_OID_RE.match(commit_oid):
            raise CodeContextError(
                code="INVALID_REQUEST",
                message=f"Invalid commit OID: {commit_oid!r}",
                retryable=False,
            )
        proc = run_safe_git(
            ["ls-tree", "-r", "-z", "--full-tree", commit_oid],
            git_dir=self.git_dir,
            empty_hooks_dir=self.empty_hooks_dir,
        )
        return parse_ls_tree_z(proc.stdout)

    def diff_tree_parent(self, parent_oid: str | None, commit_oid: str, *, parent_index: int) -> list[RawChange]:
        empty_tree, _ = self.ensure_empty_objects()
        lhs = parent_oid if parent_oid is not None else empty_tree
        proc = run_safe_git(
            [
                "diff-tree",
                "--no-commit-id",
                "--raw",
                "-r",
                "-z",
                "--no-abbrev",
                "--no-renames",
                lhs,
                commit_oid,
            ],
            git_dir=self.git_dir,
            empty_hooks_dir=self.empty_hooks_dir,
        )
        return parse_diff_tree_raw_z(proc.stdout, parent_index=parent_index, parent_oid=parent_oid)

    def diff_tree_batch(self, commits: list[tuple[str, str | None]]) -> dict[str, list[RawChange]]:
        """Diff each `(commit, first_parent)` pair (`None` parent: against the empty tree) with one
        `git diff-tree --stdin` process. Renames use the same `-M50% -l200` detection as the
        per-commit similarity pass, so callers get raw changes and rename candidates together.
        """
        if not commits:
            return {}
        parents: dict[str, str | None] = {}
        lines: list[bytes] = []
        for commit_oid, parent_oid in commits:
            for oid in (commit_oid, parent_oid):
                if oid is not None and not HEX_OID_RE.fullmatch(oid):
                    raise CodeContextError(
                        code="INVALID_REQUEST",
                        message=f"Invalid Git object ID for diff-tree: {oid!r}",
                        retryable=False,
                    )
            parents[commit_oid] = parent_oid
            line = commit_oid if parent_oid is None else f"{commit_oid} {parent_oid}"
            lines.append(line.encode("ascii") + b"\n")
        proc = run_safe_git(
            [
                "diff-tree",
                "--stdin",
                "--root",
                "--always",
                "--raw",
                "-r",
                "-z",
                "--no-abbrev",
                "-M50%",
                "-l200",
            ],
            git_dir=self.git_dir,
            empty_hooks_dir=self.empty_hooks_dir,
            stdin_bytes=b"".join(lines),
            timeout=30.0 * len(lines),
        )
        tokens = _split_z_tokens(proc.stdout)
        result: dict[str, list[RawChange]] = {}
        i = 0
        while i < len(tokens):
            commit_oid = tokens[i].decode("ascii", errors="replace")
            if commit_oid not in parents:
                raise CodeContextError(
                    code="IO_ERROR",
                    message="Unexpected commit header in diff-tree --stdin output.",
                    retryable=False,
                )
            parent_oid = parents[commit_oid]
            result[commit_oid], i = _parse_raw_z_records(
                tokens,
                i + 1,
                parent_index=-1 if parent_oid is None else 0,
                parent_oid=parent_oid,
            )
        return result

    def diff_blobs_unified_zero(self, old_blob_oid: str | None, new_blob_oid: str | None) -> list[ParsedHunk]:
        for oid in (old_blob_oid, new_blob_oid):
            if oid is not None and not HEX_OID_RE.match(oid):
                raise CodeContextError(
                    code="INVALID_REQUEST",
                    message=f"Invalid blob OID: {oid!r}",
                    retryable=False,
                )
        _, empty_blob = self.ensure_empty_objects()
        lhs = old_blob_oid if old_blob_oid is not None else empty_blob
        rhs = new_blob_oid if new_blob_oid is not None else empty_blob
        if lhs == rhs:
            return []
        proc = run_safe_git(
            [
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--no-color",
                "--no-renames",
                "--text",
                "--diff-algorithm=myers",
                "--unified=0",
                lhs,
                rhs,
                "--",
            ],
            git_dir=self.git_dir,
            empty_hooks_dir=self.empty_hooks_dir,
            check=False,
        )
        if proc.returncode not in (0, 1):
            raise CodeContextError(
                code="IO_ERROR",
                message=f"git diff between blobs failed with code {proc.returncode}",
                retryable=False,
            )
        return parse_unified_zero_diff(proc.stdout)

    def open_batch_session(self) -> CatFileBatchSession:
        if self._pooled_batch.try_acquire():
            return self._pooled_batch
        return CatFileBatchSession(self.git_dir, self.empty_hooks_dir)

    def _prepare_workspace_revision(
        self,
        source_repo: Path,
        head_oid: str,
        *,
        include_untracked: bool = False,
    ) -> tuple[str, list[bytes]]:
        """Capture tracked (and optionally untracked) working files in the owned mirror without writing the source index."""
        from codecontext.workspace import RepoDirFds

        with RepoDirFds(source_repo) as dir_fds:
            return self._capture_workspace_revision(
                source_repo, head_oid, include_untracked=include_untracked, dir_fds=dir_fds
            )

    def _capture_workspace_revision(
        self,
        source_repo: Path,
        head_oid: str,
        *,
        include_untracked: bool,
        dir_fds: "RepoDirFds",
    ) -> tuple[str, list[bytes]]:
        from codecontext.syntax import MAX_PARSE_BYTES
        from codecontext.workspace import (
            _lstat_root_confined,
            _parse_ls_files_stage_z,
            _read_root_confined_regular_file,
            _read_root_confined_symlink_target,
            index_entry_matches_stat,
            read_git_index_stat_cache,
        )

        staged = _parse_ls_files_stage_z(
            run_safe_git(
                ["ls-files", "--stage", "-z"],
                cwd=source_repo,
                empty_hooks_dir=self.empty_hooks_dir,
            ).stdout
        )
        if any(stage != 0 for _, _, stage, _ in staged):
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Resolve unmerged files before syncing the workspace.",
                retryable=False,
            )
        original = {entry.path: entry for entry in self.list_tree_entries(head_oid)}
        tracked = {path for _, _, _, path in staged}
        updates: list[bytes] = []
        oversized: list[bytes] = []
        oid_len = len(head_oid)
        zero_oid = "0" * oid_len
        dir_cache: dict[Path, Path | None] = {}
        max_dirty_mtime_sec = 0
        index_path_out = run_safe_git(
            ["rev-parse", "--git-path", "index"],
            cwd=source_repo,
            empty_hooks_dir=self.empty_hooks_dir,
            check=False,
        )
        stat_cache = (
            read_git_index_stat_cache(
                source_repo / os.fsdecode(index_path_out.stdout.rstrip(b"\n")), oid_bytes=oid_len // 2
            )
            if index_path_out.returncode == 0 and index_path_out.stdout.strip()
            else None
        )

        for path in original.keys() - tracked:
            updates.append(b"0 " + zero_oid.encode("ascii") + b"\t" + path + b"\0")

        for mode, staged_oid, _, path in staged:
            check_deadline()
            previous = original.get(path)
            if mode == 0o160000:
                if previous is None or previous.oid != staged_oid or previous.mode != mode:
                    updates.append(f"{mode:o} {staged_oid}\t".encode("ascii") + path + b"\0")
                continue
            if mode == 0o120000:
                sym_target = _read_root_confined_symlink_target(source_repo, path, dir_cache=dir_cache, dir_fds=dir_fds)
                if sym_target is None:
                    if path in original:
                        updates.append(b"0 " + zero_oid.encode("ascii") + b"\t" + path + b"\0")
                    continue
                sym_oid = git_blob_oid(sym_target, oid_len=oid_len)
                if previous is None or previous.oid != sym_oid or previous.mode != 0o120000:
                    self._write_loose_blob(sym_oid, sym_target)
                    updates.append(f"{0o120000:o} {sym_oid}\t".encode("ascii") + path + b"\0")
                continue

            if stat_cache is not None and previous is not None and previous.oid == staged_oid and previous.mode == mode:
                cached = stat_cache[0].get(path)
                if cached is not None and cached.oid == staged_oid:
                    st = _lstat_root_confined(source_repo, path, dir_cache=dir_cache, dir_fds=dir_fds)
                    if st is not None and st.st_size <= MAX_PARSE_BYTES and index_entry_matches_stat(cached, st, stat_cache[1]):
                        continue

            state, content, working_mode = _read_root_confined_regular_file(
                source_repo, path, dir_cache=dir_cache, dir_fds=dir_fds
            )
            if state == "deleted":
                if path in original:
                    updates.append(b"0 " + zero_oid.encode("ascii") + b"\t" + path + b"\0")
                continue
            if state == "oversized":
                oversized.append(path)
                if path in original:
                    updates.append(b"0 " + zero_oid.encode("ascii") + b"\t" + path + b"\0")
                continue
            if state == "symlink":
                sym_target = _read_root_confined_symlink_target(source_repo, path, dir_cache=dir_cache, dir_fds=dir_fds)
                if sym_target is not None:
                    sym_oid = git_blob_oid(sym_target, oid_len=oid_len)
                    if previous is None or previous.oid != sym_oid or previous.mode != 0o120000:
                        self._write_loose_blob(sym_oid, sym_target)
                        updates.append(f"{0o120000:o} {sym_oid}\t".encode("ascii") + path + b"\0")
                else:
                    if path in original:
                        updates.append(b"0 " + zero_oid.encode("ascii") + b"\t" + path + b"\0")
                continue
            if state == "unavailable":
                if path in original:
                    updates.append(b"0 " + zero_oid.encode("ascii") + b"\t" + path + b"\0")
                continue
            if state != "ok" or content is None or working_mode is None:
                raise CodeContextError(
                    code="IO_ERROR",
                    message="Could not safely read a tracked workspace file.",
                    retryable=True,
                )
            oid_str = git_blob_oid(content, oid_len=oid_len)
            if previous is None or previous.oid != oid_str or previous.mode != working_mode:
                self._write_loose_blob(oid_str, content)
                updates.append(f"{working_mode:o} {oid_str}\t".encode("ascii") + path + b"\0")
                st = _lstat_root_confined(source_repo, path, dir_cache=dir_cache, dir_fds=dir_fds)
                if st is not None:
                    max_dirty_mtime_sec = max(max_dirty_mtime_sec, int(st.st_mtime))

        if include_untracked:
            untracked_raw = run_safe_git(
                ["ls-files", "--others", "--exclude-standard", "-z"],
                cwd=source_repo,
                empty_hooks_dir=self.empty_hooks_dir,
            ).stdout
            for path in untracked_raw.split(b"\x00"):
                if not path or path in tracked:
                    continue
                check_deadline()
                state, content, working_mode = _read_root_confined_regular_file(
                    source_repo, path, dir_cache=dir_cache, dir_fds=dir_fds
                )
                if state == "oversized":
                    oversized.append(path)
                    continue
                if state == "ok" and content is not None and working_mode is not None:
                    oid_str = git_blob_oid(content, oid_len=oid_len)
                    self._write_loose_blob(oid_str, content)
                    updates.append(f"{working_mode:o} {oid_str}\t".encode("ascii") + path + b"\0")
                    st = _lstat_root_confined(
                        source_repo, path, dir_cache=dir_cache, dir_fds=dir_fds
                    )
                    if st is not None:
                        max_dirty_mtime_sec = max(max_dirty_mtime_sec, int(st.st_mtime))

        if not updates:
            return head_oid, oversized

        with tempfile.TemporaryDirectory(prefix="codecontext-index-") as temp_dir:
            env = {"GIT_INDEX_FILE": str(Path(temp_dir) / "index")}
            for args, payload in [
                (["read-tree", head_oid], None),
                (["update-index", "-z", "--index-info"], b"".join(updates)),
            ]:
                run_safe_git(
                    args,
                    git_dir=self.git_dir,
                    empty_hooks_dir=self.empty_hooks_dir,
                    extra_env=env,
                    stdin_bytes=payload,
                )
            tree_oid = (
                run_safe_git(
                    ["write-tree"],
                    git_dir=self.git_dir,
                    empty_hooks_dir=self.empty_hooks_dir,
                    extra_env=env,
                )
                .stdout.decode("ascii")
                .strip()
            )

        commit_ts_sec = max_dirty_mtime_sec
        with self.store.connect_read() as r_conn:
            parent_row = r_conn.execute(
                "SELECT committed_at_us FROM commits WHERE oid = ?",
                (head_oid,),
            ).fetchone()
            if parent_row and parent_row["committed_at_us"]:
                commit_ts_sec = max(commit_ts_sec, int(parent_row["committed_at_us"]) // 1_000_000)
            prev_meta = r_conn.execute("SELECT active_revision FROM store_meta WHERE singleton = 1").fetchone()
            prev_active = str(prev_meta["active_revision"]) if prev_meta and prev_meta["active_revision"] else None
            if prev_active and prev_active != head_oid:
                prev_c = r_conn.execute(
                    "SELECT c.tree_oid, cp.parent_oid FROM commits c "
                    "LEFT JOIN commit_parents cp ON cp.commit_oid = c.oid AND cp.parent_index = 0 "
                    "WHERE c.oid = ?",
                    (prev_active,),
                ).fetchone()
                if prev_c and str(prev_c["tree_oid"]) == tree_oid and str(prev_c["parent_oid"]) == head_oid:
                    self.pin_retained_oid(prev_active)
                    return prev_active, oversized

        if commit_ts_sec <= 0:
            commit_ts_sec = now_us() // 1_000_000

        raw_commit = (
            f"tree {tree_oid}\nparent {head_oid}\n"
            f"author CodeContext Workspace <workspace@codecontext.invalid> {commit_ts_sec} +0000\n"
            f"committer CodeContext Workspace <workspace@codecontext.invalid> {commit_ts_sec} +0000\n\n"
            "Uncommitted workspace snapshot (created in the CodeContext store only).\n"
        ).encode()
        snapshot_oid = (
            run_safe_git(
                ["hash-object", "-w", "-t", "commit", "--stdin"],
                git_dir=self.git_dir,
                empty_hooks_dir=self.empty_hooks_dir,
                stdin_bytes=raw_commit,
            )
            .stdout.decode("ascii")
            .strip()
        )
        self.pin_retained_oid(snapshot_oid)
        return snapshot_oid, oversized

    def _write_loose_blob(self, oid: str, content: bytes) -> None:
        """Store `content` as a loose blob under `oid` straight from the bytes already read, so a
        file that changes after it was read cannot desync the snapshot and no git process runs."""
        obj_dir = self.git_dir / "objects" / oid[:2]
        target = obj_dir / oid[2:]
        if target.exists():
            return
        obj_dir.mkdir(exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=obj_dir, prefix="tmp_obj_")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(zlib.compress(f"blob {len(content)}\0".encode("ascii") + content))
            os.chmod(tmp_name, 0o444)
            os.replace(tmp_name, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise

    def prepare_snapshot(
        self,
        revision: str = "HEAD",
        *,
        include_workspace: bool = False,
        include_untracked: bool = False,
        workspace_override: Path | None = None,
    ) -> str:
        """Import objects for `revision`, parse commit graph and file map, and atomically
        publish `snapshot_files`, `snapshot_commits`, `refs`, and `active_revision`.
        """
        source_repo = self.verify_source_binding(workspace_override=workspace_override)
        resolved_oid, source_refs, is_shallow = self.import_local_refs_and_revision(
            source_repo, revision
        )
        source_oid = resolved_oid
        oversized: list[bytes] = []
        if include_workspace:
            resolved_oid, oversized = self._prepare_workspace_revision(
                source_repo,
                resolved_oid,
                include_untracked=include_untracked,
            )
        is_workspace_snapshot = resolved_oid != source_oid

        cached_tree_entries: list[TreeEntry] | None = None
        cached_snapshot_commits: set[str] | None = None
        has_v2 = getattr(self.store, "schema_version", 1) >= 2
        if has_v2 and not is_workspace_snapshot:
            with self.store.connect_read() as r_conn:
                cached_rows = r_conn.execute(
                    "SELECT path, blob_oid, mode FROM revision_snapshots WHERE revision = ?",
                    (resolved_oid,),
                ).fetchall()
                anc_rows = r_conn.execute(
                    "SELECT commit_oid FROM revision_ancestors WHERE revision = ?",
                    (resolved_oid,),
                ).fetchall()
                if cached_rows and anc_rows:
                    cached_tree_entries = [
                        TreeEntry(
                            mode=int(r["mode"]),
                            obj_type="commit" if int(r["mode"]) == 0o160000 else "blob",
                            oid=str(r["blob_oid"]),
                            path=bytes(r["path"]),
                        )
                        for r in cached_rows
                    ]
                    cached_snapshot_commits = {str(r["commit_oid"]) for r in anc_rows}

        tree_entries = (
            cached_tree_entries
            if cached_tree_entries is not None
            else self.list_tree_entries(resolved_oid)
        )
        if cached_snapshot_commits is not None:
            snapshot_commit_oids = cached_snapshot_commits
        else:
            snapshot_reachable = self.list_reachable_commits_with_parents([resolved_oid])
            snapshot_commit_oids = {oid for oid, _ in snapshot_reachable}

        all_tips = list(dict.fromkeys([resolved_oid, *source_refs.values()]))
        all_reachable = self.list_reachable_commits_with_parents(all_tips)

        with self.store.connect_read() as conn:
            known_commits = {r[0] for r in conn.execute("SELECT oid FROM commits")}
        missing_commits = [oid for oid, _ in all_reachable if oid not in known_commits]
        ts = now_us()
        with self.open_batch_session() as batch:
            for offset in range(0, len(missing_commits), 64):
                parsed_commits = []
                for c_oid in missing_commits[offset : offset + 64]:
                    check_deadline()
                    obj = batch.read_object(c_oid)
                    if obj is not None and obj[0] == "commit":
                        parsed_commits.append(parse_raw_commit_object(c_oid, obj[2]))
                with self.store.write_transaction(bump_generation=True) as conn:
                    for pc in parsed_commits:
                        check_deadline()
                        initial_diff_state = (
                            "complete"
                            if (is_workspace_snapshot and pc.oid == resolved_oid)
                            else "pending"
                        )
                        conn.execute(
                            """
                            INSERT INTO commits (
                                oid, tree_oid, authored_at_us, committed_at_us,
                                author_name, author_email, message, first_seen_at_us, diff_state
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(oid) DO NOTHING
                            """,
                            (
                                pc.oid,
                                pc.tree_oid,
                                pc.authored_at_us,
                                pc.committed_at_us,
                                pc.author_name,
                                pc.author_email,
                                pc.message_text,
                                ts,
                                initial_diff_state,
                            ),
                        )
                        for p_idx, p_oid in enumerate(pc.parent_oids):
                            conn.execute(
                                """
                                INSERT INTO commit_parents (commit_oid, parent_index, parent_oid)
                                VALUES (?, ?, ?)
                                ON CONFLICT(commit_oid, parent_index) DO NOTHING
                                """,
                                (pc.oid, p_idx, p_oid),
                            )

        ts = now_us()
        old_synthetic_to_unpin: str | None = None
        with self.store.write_transaction(bump_generation=True) as conn:
            prev_rev_row = conn.execute(
                "SELECT active_revision FROM store_meta WHERE singleton = 1"
            ).fetchone()
            prev_rev = str(prev_rev_row["active_revision"]) if prev_rev_row and prev_rev_row["active_revision"] else None
            prev_ws_base: str | None = None
            prev_ckpt = conn.execute(
                "SELECT cursor_json FROM checkpoints WHERE stream_key = 'git:snapshot'"
            ).fetchone()
            if prev_ckpt and prev_ckpt[0]:
                with contextlib.suppress(Exception):
                    prev_ws_base = json.loads(str(prev_ckpt[0])).get("workspace_base_revision")

            Store.resolve_gaps_in_scope(conn, scope="workspace:snapshot")
            if oversized:
                Store.upsert_gap(
                    conn,
                    repository_id=self.store.config.repository_id,
                    code="BLOB_TOO_LARGE",
                    scope="workspace:snapshot",
                    message=f"{len(oversized)} workspace file(s) exceed the bounded read limit and use committed metadata only.",
                    retryable=False,
                    observed_us=ts,
                )
            if is_workspace_snapshot:
                Store.upsert_gap(
                    conn,
                    repository_id=self.store.config.repository_id,
                    code="WORKSPACE_SNAPSHOT",
                    scope="workspace:snapshot",
                    message=f"Snapshot includes uncommitted tracked files based on {source_oid}; the synthetic revision exists only in this store.",
                    retryable=False,
                    observed_us=ts,
                )

            conn.execute("UPDATE refs SET active = 0 WHERE source = 'local'")
            for ref_name, ref_oid in source_refs.items():
                conn.execute(
                    """
                    INSERT INTO refs (name, commit_oid, source, last_seen_at_us, active)
                    VALUES (?, ?, 'local', ?, 1)
                    ON CONFLICT(name) DO UPDATE SET
                        commit_oid = excluded.commit_oid,
                        source = excluded.source,
                        last_seen_at_us = excluded.last_seen_at_us,
                        active = 1
                    """,
                    (ref_name, ref_oid, ts),
                )

            if is_shallow:
                Store.upsert_gap(
                    conn,
                    repository_id=self.store.config.repository_id,
                    code="SHALLOW_HISTORY",
                    scope="git:refs",
                    message="Source repository is shallow; history before the shallow boundary is unavailable.",
                    retryable=True,
                    observed_us=ts,
                )
            else:
                Store.resolve_gaps_in_scope(conn, scope="git:refs", code="SHALLOW_HISTORY")
                conn.execute(
                    "UPDATE commits SET diff_state = 'partial' WHERE oid IN ("
                    "SELECT SUBSTR(scope, 12) FROM coverage_gaps "
                    "WHERE code = 'MISSING_PARENT' AND active = 1 AND scope LIKE 'git:commit:%'"
                    ")"
                )

            from codecontext.syntax import EXTRACTOR_FINGERPRINTS, detect_language_for_path

            distinct_blobs = list({entry.oid for entry in tree_entries if entry.mode not in (0o160000, 0o120000)})
            blob_to_ext: dict[tuple[str, str], tuple[str, str]] = {}
            for start in range(0, len(distinct_blobs), 500):
                chunk_oids = distinct_blobs[start : start + 500]
                placeholders = ",".join("?" for _ in chunk_oids)
                existing_ext_rows = conn.execute(
                    f"SELECT blob_oid, id, state, language, extractor_fingerprint FROM extractions WHERE blob_oid IN ({placeholders})",
                    chunk_oids,
                ).fetchall()
                for r in existing_ext_rows:
                    if r["extractor_fingerprint"] == EXTRACTOR_FINGERPRINTS.get(r["language"]):
                        blob_to_ext[(str(r["blob_oid"]), str(r["language"]))] = (str(r["id"]), str(r["state"]))

            old_commits = {r[0] for r in conn.execute("SELECT commit_oid FROM snapshot_commits")}
            new_commits = snapshot_commit_oids
            conn.executemany(
                "DELETE FROM snapshot_commits WHERE commit_oid = ?",
                [(oid,) for oid in old_commits - new_commits],
            )
            conn.executemany(
                "INSERT INTO snapshot_commits (commit_oid) VALUES (?)",
                [(oid,) for oid in new_commits - old_commits],
            )

            old_files = {bytes(r["path"]): r for r in conn.execute("SELECT * FROM snapshot_files")}
            new_paths = {entry.path for entry in tree_entries}
            conn.executemany(
                "DELETE FROM snapshot_files WHERE path = ?",
                [(p,) for p in old_files.keys() - new_paths],
            )
            snapshot_changed = old_files.keys() != new_paths
            pending_structure_count = 0
            resolved_snapshot_rows: list[tuple[str, bytes, str, int, str | None, str, int]] = []

            for entry in tree_entries:
                check_deadline()
                lang = detect_language_for_path(entry.path)
                ext_key = (entry.oid, lang)
                previous = old_files.get(entry.path)
                if (
                    previous is not None
                    and previous["blob_oid"] == entry.oid
                    and previous["mode"] == entry.mode
                    and (
                        previous["extraction_id"] is None
                        or previous["extraction_id"] == blob_to_ext.get(ext_key, (None, None))[0]
                    )
                ):
                    pending_structure_count += previous["content_state"] == "pending"
                    resolved_snapshot_rows.append(
                        (
                            resolved_oid,
                            entry.path,
                            entry.oid,
                            entry.mode,
                            previous["extraction_id"],
                            str(previous["content_state"]),
                            ts,
                        )
                    )
                    continue

                snapshot_changed = True
                conn.execute("DELETE FROM snapshot_files WHERE path = ?", (entry.path,))
                if entry.mode == 0o160000:
                    conn.execute(
                        "INSERT INTO snapshot_files (path, blob_oid, mode, extraction_id, content_state) VALUES (?, ?, ?, NULL, 'metadata_only')",
                        (entry.path, entry.oid, entry.mode),
                    )
                    Store.upsert_gap(
                        conn,
                        repository_id=self.store.config.repository_id,
                        code="SUBMODULE_NOT_FETCHED",
                        scope=f"blob:{entry.oid}",
                        message="Submodule commit reference retained without fetching external repository contents.",
                        retryable=False,
                        observed_us=ts,
                    )
                    resolved_snapshot_rows.append((resolved_oid, entry.path, entry.oid, entry.mode, None, "metadata_only", ts))
                elif entry.mode == 0o120000:
                    conn.execute(
                        "INSERT INTO snapshot_files (path, blob_oid, mode, extraction_id, content_state) VALUES (?, ?, ?, NULL, 'metadata_only')",
                        (entry.path, entry.oid, entry.mode),
                    )
                    Store.upsert_gap(
                        conn,
                        repository_id=self.store.config.repository_id,
                        code="SYMLINK_NOT_FOLLOWED",
                        scope=f"blob:{entry.oid}",
                        message="Symlink target text retained as evidence without filesystem traversal.",
                        retryable=False,
                        observed_us=ts,
                    )
                    resolved_snapshot_rows.append((resolved_oid, entry.path, entry.oid, entry.mode, None, "metadata_only", ts))
                elif lang is not None and (entry.oid, lang) in blob_to_ext:
                    ext_id, ext_state = blob_to_ext[(entry.oid, lang)]
                    c_state = (
                        "indexed"
                        if ext_state in ("complete", "partial")
                        else ("text_only" if ext_state == "unsupported" else "metadata_only")
                    )
                    conn.execute(
                        "INSERT INTO snapshot_files (path, blob_oid, mode, extraction_id, content_state) VALUES (?, ?, ?, ?, ?)",
                        (entry.path, entry.oid, entry.mode, ext_id, c_state),
                    )
                    resolved_snapshot_rows.append((resolved_oid, entry.path, entry.oid, entry.mode, ext_id, c_state, ts))
                else:
                    pending_structure_count += 1
                    conn.execute(
                        "INSERT INTO snapshot_files (path, blob_oid, mode, extraction_id, content_state) VALUES (?, ?, ?, NULL, 'pending')",
                        (entry.path, entry.oid, entry.mode),
                    )
                    resolved_snapshot_rows.append((resolved_oid, entry.path, entry.oid, entry.mode, None, "pending", ts))

            if has_v2 and not is_workspace_snapshot:
                conn.execute("DELETE FROM revision_snapshots WHERE revision = ?", (resolved_oid,))
                conn.executemany(
                    """
                    INSERT INTO revision_snapshots (
                        revision, path, blob_oid, mode, extraction_id, content_state, prepared_at_us
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    resolved_snapshot_rows,
                )
                conn.execute("DELETE FROM revision_ancestors WHERE revision = ?", (resolved_oid,))
                conn.executemany(
                    "INSERT INTO revision_ancestors (revision, commit_oid) VALUES (?, ?)",
                    [(resolved_oid, c_oid) for c_oid in new_commits],
                )
                stale_cached = conn.execute(
                    """
                    SELECT revision FROM (
                        SELECT revision, MAX(prepared_at_us) AS last_prep
                        FROM revision_snapshots
                        GROUP BY revision
                        ORDER BY last_prep DESC
                        LIMIT -1 OFFSET ?
                    )
                    """,
                    (MAX_CACHED_REVISION_SNAPSHOTS,),
                ).fetchall()
                for sc in stale_cached:
                    sc_rev = str(sc["revision"])
                    if sc_rev != resolved_oid:
                        conn.execute("DELETE FROM revision_snapshots WHERE revision = ?", (sc_rev,))
                        conn.execute("DELETE FROM revision_ancestors WHERE revision = ?", (sc_rev,))

            if snapshot_changed:
                Store.upsert_gap(
                    conn,
                    repository_id=self.store.config.repository_id,
                    code="STRUCTURE_PENDING",
                    scope="syntax:documents",
                    message="Code search and call relationships are awaiting refresh.",
                    retryable=True,
                )
            if pending_structure_count > 0:
                Store.upsert_gap(
                    conn,
                    repository_id=self.store.config.repository_id,
                    code="STRUCTURE_PENDING",
                    scope="git:snapshot",
                    message=f"{pending_structure_count} snapshot file(s) awaiting syntax/text extraction.",
                    retryable=True,
                    observed_us=ts,
                )
            else:
                Store.resolve_gaps_in_scope(conn, scope="git:snapshot", code="STRUCTURE_PENDING")

            if prev_rev is not None and prev_rev != resolved_oid:
                conn.execute(
                    "UPDATE summaries SET state = 'stale' WHERE revision != ? AND state = 'ready'",
                    (resolved_oid,),
                )
                conn.execute(
                    "DELETE FROM search_documents WHERE record_kind = 'summary' AND record_id IN ("
                    "SELECT id FROM summaries WHERE state != 'ready'"
                    ")"
                )
                # Clean up superseded ephemeral workspace commit so synthetic snapshots never pollute history
                if prev_ws_base is not None and prev_rev != prev_ws_base and prev_rev not in source_refs.values():
                    conn.execute("DELETE FROM jobs WHERE kind = 'git_commit_diff' AND owner_id = ?", (prev_rev,))
                    conn.execute("DELETE FROM episodes WHERE kind = 'commit' AND commit_oid = ?", (prev_rev,))
                    conn.execute("DELETE FROM evidence WHERE git_oid = ?", (prev_rev,))
                    conn.execute("DELETE FROM commits WHERE oid = ?", (prev_rev,))
                    old_synthetic_to_unpin = prev_rev

            ws_base_val = source_oid if is_workspace_snapshot else None
            if has_v2:
                conn.execute(
                    "UPDATE store_meta SET active_revision = ?, workspace_base_revision = ? WHERE singleton = 1",
                    (resolved_oid, ws_base_val),
                )
            else:
                conn.execute(
                    "UPDATE store_meta SET active_revision = ? WHERE singleton = 1",
                    (resolved_oid,),
                )
            conn.execute(
                """
                INSERT INTO checkpoints (stream_key, cursor_json, last_success_at_us, last_reconciled_at_us)
                VALUES ('git:snapshot', ?, ?, ?)
                ON CONFLICT(stream_key) DO UPDATE SET
                    cursor_json = excluded.cursor_json,
                    last_success_at_us = excluded.last_success_at_us,
                    last_reconciled_at_us = excluded.last_reconciled_at_us
                """,
                (json.dumps({"workspace_base_revision": ws_base_val}), ts, ts),
            )

        if old_synthetic_to_unpin:
            self.unpin_retained_oid(old_synthetic_to_unpin)

        return resolved_oid

    def resolve_prepared_revision(self, conn: sqlite3.Connection, requested_revision: str | None) -> str:
        """Resolve `revision` for search/context against the active prepared snapshot."""
        meta = conn.execute("SELECT * FROM store_meta WHERE singleton = 1").fetchone()
        active_rev = meta["active_revision"] if meta else None
        if not active_rev:
            raise CodeContextError(
                code="REVISION_NOT_PREPARED",
                message="No revision snapshot has been prepared yet. Run sync first.",
                retryable=False,
                details={"active_revision": None},
            )
        if requested_revision is None:
            return str(active_rev)

        validate_revision_arg(requested_revision)
        if requested_revision == active_rev:
            return str(active_rev)

        ws_base: str | None = (
            str(meta["workspace_base_revision"])
            if "workspace_base_revision" in meta.keys() and meta["workspace_base_revision"]
            else None
        )
        if ws_base is None:
            checkpoint = conn.execute(
                "SELECT cursor_json FROM checkpoints WHERE stream_key = 'git:snapshot'"
            ).fetchone()
            if checkpoint and checkpoint[0]:
                with contextlib.suppress(Exception):
                    ws_base = json.loads(str(checkpoint[0])).get("workspace_base_revision")

        if requested_revision == "HEAD" and ws_base:
            return str(active_rev)

        candidates = [
            requested_revision,
            f"refs/heads/{requested_revision}",
            f"refs/tags/{requested_revision}",
            f"refs/remotes/{requested_revision}",
        ]
        resolved_ref_oid: str | None = None
        for cand in candidates:
            row = conn.execute(
                "SELECT commit_oid FROM refs WHERE name = ? AND active = 1",
                (cand,),
            ).fetchone()
            if row:
                resolved_ref_oid = str(row["commit_oid"])
                break

        if resolved_ref_oid == active_rev:
            return str(active_rev)

        is_cached = False
        if resolved_ref_oid or HEX_OID_RE.match(requested_revision):
            target_check = resolved_ref_oid or requested_revision
            with contextlib.suppress(sqlite3.OperationalError):
                c_row = conn.execute(
                    "SELECT 1 FROM revision_snapshots WHERE revision = ? LIMIT 1",
                    (target_check,),
                ).fetchone()
                is_cached = c_row is not None

        raise CodeContextError(
            code="REVISION_NOT_PREPARED",
            message=f"Requested revision {requested_revision!r} is not the active prepared snapshot ({active_rev}). Run sync --revision {requested_revision} first.",
            retryable=False,
            details={
                "active_revision": str(active_rev),
                "requested_revision": requested_revision[:200],
                "resolved_revision": resolved_ref_oid,
                "cached_snapshot_available": is_cached,
            },
        )
