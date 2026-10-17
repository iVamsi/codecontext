import os
import re
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codecontext.errors import CodeContextError
from codecontext.identity import HEX_OID_RE
from codecontext.store import Store, inspect_local_git_repository, now_us

HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

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
    ) as proc:
        try:
            stdout, stderr = proc.communicate(input=stdin_bytes, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.wait()
            raise CodeContextError(
                code="IO_ERROR",
                message="Git command timed out.",
                retryable=True,
                details={"command": args[0] if args else "git"},
            ) from exc
        except BaseException:
            proc.kill()
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
    lt = text.find("<")
    gt = text.rfind(">")
    if lt == -1 or gt == -1 or gt < lt:
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
    if not raw_stdout:
        return []
    tokens = raw_stdout.split(b"\x00")
    # Last token after trailing NUL is empty
    if tokens and tokens[-1] == b"":
        tokens.pop()

    changes: list[RawChange] = []
    i = 0
    n = len(tokens)
    while i < n:
        meta_tok = tokens[i]
        i += 1
        if not meta_tok.startswith(b":"):
            raise CodeContextError(
                code="IO_ERROR",
                message="Malformed diff-tree metadata token.",
                retryable=False,
            )
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
    return changes


def parse_unified_zero_diff(raw_diff: bytes) -> list[ParsedHunk]:
    """Parse `git diff ... --unified=0 <old_blob> <new_blob> --` output into hunks.
    Validates counted body lines and ignores `\\ No newline at end of file` markers.
    """
    if not raw_diff:
        return []
    lines = raw_diff.splitlines(keepends=True)
    hunks: list[ParsedHunk] = []
    i = 0
    n = len(lines)

    # Skip file headers until first @@
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

        hunk_lines = [header_line]
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
    """Long-lived `git cat-file --batch` session reading length-framed objects."""

    def __init__(self, git_dir: Path, empty_hooks_dir: Path) -> None:
        self.git_dir = git_dir
        self.empty_hooks_dir = empty_hooks_dir
        self._proc: subprocess.Popen[bytes] | None = None

    def _ensure_proc(self) -> subprocess.Popen[bytes]:
        if self._proc is None or self._proc.poll() is not None:
            cmd = [
                "git",
                "-c",
                "core.fsmonitor=false",
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
                stderr=subprocess.PIPE,
            )
        return self._proc

    def read_object(self, oid: str) -> tuple[str, int, bytes] | None:
        """Return (obj_type, size, content_bytes) or None if missing."""
        if not HEX_OID_RE.match(oid):
            raise CodeContextError(
                code="INVALID_REQUEST",
                message=f"Invalid Git object ID for cat-file: {oid!r}",
                retryable=False,
            )
        proc = self._ensure_proc()
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(oid.encode("ascii") + b"\n")
        proc.stdin.flush()

        header_line = proc.stdout.readline()
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
        data = proc.stdout.read(size)
        if len(data) != size:
            self.close()
            raise CodeContextError(
                code="IO_ERROR",
                message=f"Truncated cat-file --batch object payload for {oid}",
                retryable=False,
            )
        framing = proc.stdout.read(1)
        if framing != b"\n":
            self.close()
            raise CodeContextError(
                code="IO_ERROR",
                message="Missing framing newline in cat-file --batch.",
                retryable=False,
            )
        return obj_type, size, data

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=1.0)
                except Exception:
                    pass
            self._proc = None

    def __enter__(self) -> "CatFileBatchSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class GitMirror:
    """Manages owned bare `repository.git/` and safe operations against the source repository."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.git_dir = store.git_dir
        self.empty_hooks_dir = store.tmp_dir / "_empty_hooks"
        self.empty_hooks_dir.mkdir(parents=True, exist_ok=True)
        self._empty_tree_oid: str | None = None
        self._empty_blob_oid: str | None = None

    def verify_source_binding(self) -> Path:
        """Verify the local source repository still matches store_meta's common_dir binding."""
        if not self.store.config.repo_path:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Store has no bound local source repository.",
                retryable=False,
            )
        src_path = Path(self.store.config.repo_path)
        resolved_repo, common_real, st_dev, st_ino, obj_fmt = inspect_local_git_repository(src_path)
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
                or int(meta["common_dir_device"]) != st_dev
                or int(meta["common_dir_inode"]) != st_ino
                or meta["object_format"] != obj_fmt
            ):
                raise CodeContextError(
                    code="REPOSITORY_MISMATCH",
                    message="Source repository common-dir identity changed or does not match store binding.",
                    retryable=False,
                )
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
        """Return declared source refs: refs/heads/*, refs/remotes/*, refs/tags/*, and HEAD."""
        proc = run_safe_git(
            [
                "for-each-ref",
                "--format=%(refname)%00%(objectname)%00%(*objectname)",
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
            if len(parts) < 2:
                continue
            refname = parts[0].decode("utf-8", errors="replace")
            oid = parts[1].decode("ascii", errors="replace").strip()
            peeled = parts[2].decode("ascii", errors="replace").strip() if len(parts) > 2 else ""
            # Prefer peeled commit OID for annotated tags when present
            target_oid = peeled if peeled and HEX_OID_RE.match(peeled) else oid
            if HEX_OID_RE.match(target_oid):
                refs[refname] = target_oid

        # Also resolve HEAD
        try:
            head_oid = self.resolve_source_revision(source_repo, "HEAD")
            refs["HEAD"] = head_oid
        except CodeContextError:
            pass
        return refs

    def pin_retained_oid(self, oid: str) -> None:
        if not HEX_OID_RE.match(oid):
            return
        run_safe_git(
            ["update-ref", f"refs/codecontext/retained/{oid}", oid],
            git_dir=self.git_dir,
            empty_hooks_dir=self.empty_hooks_dir,
            check=False,
        )

    def import_local_refs_and_revision(
        self,
        source_repo: Path,
        requested_revision: str,
        *,
        timeout: float = 60.0,
    ) -> tuple[str, dict[str, str], bool]:
        """Import declared local refs and the requested revision into owned `refs/codecontext/local/*` namespaces.
        Returns (resolved_commit_oid, ref_map, is_shallow).
        """
        is_shallow = self.is_source_shallow(source_repo)
        resolved_oid = self.resolve_source_revision(source_repo, requested_revision)
        source_refs = self.enumerate_source_refs(source_repo)

        # Pin any existing owned local tips before replacing them
        existing_proc = run_safe_git(
            ["for-each-ref", "--format=%(refname)%00%(objectname)", "refs/codecontext/local/"],
            git_dir=self.git_dir,
            empty_hooks_dir=self.empty_hooks_dir,
        )
        for line in existing_proc.stdout.splitlines():
            parts = line.split(b"\x00")
            if len(parts) == 2:
                old_oid = parts[1].decode("ascii", errors="replace").strip()
                self.pin_retained_oid(old_oid)

        refspecs = [
            "+refs/heads/*:refs/codecontext/local/heads/*",
            "+refs/remotes/*:refs/codecontext/local/remotes/*",
            "+refs/tags/*:refs/codecontext/local/tags/*",
            f"+{resolved_oid}:refs/codecontext/retained/{resolved_oid}",
        ]
        if "HEAD" in source_refs:
            head_oid = source_refs["HEAD"]
            refspecs.append(f"+{head_oid}:refs/codecontext/local/HEAD")

        run_safe_git(
            [
                "fetch",
                "--no-tags",
                "--no-write-fetch-head",
                "--no-auto-maintenance",
                "--end-of-options",
                str(source_repo),
                *refspecs,
            ],
            git_dir=self.git_dir,
            empty_hooks_dir=self.empty_hooks_dir,
            allow_file_protocol=True,
            timeout=timeout,
        )
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

    def diff_blobs_unified_zero(self, old_blob_oid: str | None, new_blob_oid: str | None) -> list[ParsedHunk]:
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
        return CatFileBatchSession(self.git_dir, self.empty_hooks_dir)

    def prepare_snapshot(self, revision: str = "HEAD") -> str:
        """Import objects for `revision`, parse commit graph and file map, and atomically
        publish `snapshot_files`, `snapshot_commits`, `refs`, and `active_revision`.
        """
        source_repo = self.verify_source_binding()
        resolved_oid, source_refs, is_shallow = self.import_local_refs_and_revision(
            source_repo, revision
        )

        # Read tree entries for active revision and reachable commits across all declared refs outside write transaction
        tree_entries = self.list_tree_entries(resolved_oid)
        snapshot_reachable = self.list_reachable_commits_with_parents([resolved_oid])
        all_tips = list(dict.fromkeys([resolved_oid, *source_refs.values()]))
        all_reachable = self.list_reachable_commits_with_parents(all_tips)

        parsed_commits: list[ParsedCommit] = []
        with self.open_batch_session() as batch:
            for c_oid, _ in all_reachable:
                obj = batch.read_object(c_oid)
                if obj is not None and obj[0] == "commit":
                    parsed_commits.append(parse_raw_commit_object(c_oid, obj[2]))

        ts = now_us()
        with self.store.write_transaction(bump_generation=True) as conn:
            # Record observed refs
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

            # Upsert reachable commits and parents
            for pc in parsed_commits:
                conn.execute(
                    """
                    INSERT INTO commits (
                        oid, tree_oid, authored_at_us, committed_at_us,
                        author_name, author_email, message, first_seen_at_us, diff_state
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
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
                    # Check if parent object exists in mirror; if not, record MISSING_PARENT gap
                    # (Checked when shallow or missing)

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

            # Find existing extractions for blobs in the new snapshot so unchanged blobs reuse extraction_id
            existing_ext_rows = conn.execute(
                "SELECT blob_oid, id, state FROM extractions"
            ).fetchall()
            blob_to_ext: dict[str, tuple[str, str]] = {
                str(r["blob_oid"]): (str(r["id"]), str(r["state"])) for r in existing_ext_rows
            }

            # Atomically replace snapshot_commits and snapshot_files
            conn.execute("DELETE FROM snapshot_commits")
            for c_oid, _ in snapshot_reachable:
                conn.execute("INSERT INTO snapshot_commits (commit_oid) VALUES (?)", (c_oid,))

            conn.execute("DELETE FROM snapshot_files")
            pending_structure_count = 0
            for entry in tree_entries:
                # Gitlink (submodule mode 0o160000 = 57344) or symlink (0o120000 = 40960)
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
                elif entry.oid in blob_to_ext:
                    ext_id, ext_state = blob_to_ext[entry.oid]
                    c_state = (
                        "indexed"
                        if ext_state in ("complete", "partial")
                        else ("text_only" if ext_state == "unsupported" else "metadata_only")
                    )
                    conn.execute(
                        "INSERT INTO snapshot_files (path, blob_oid, mode, extraction_id, content_state) VALUES (?, ?, ?, ?, ?)",
                        (entry.path, entry.oid, entry.mode, ext_id, c_state),
                    )
                else:
                    pending_structure_count += 1
                    conn.execute(
                        "INSERT INTO snapshot_files (path, blob_oid, mode, extraction_id, content_state) VALUES (?, ?, ?, NULL, 'pending')",
                        (entry.path, entry.oid, entry.mode),
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

            # Mark old revision-specific summaries stale if active_revision changed
            prev_rev_row = conn.execute(
                "SELECT active_revision FROM store_meta WHERE singleton = 1"
            ).fetchone()
            prev_rev = prev_rev_row["active_revision"] if prev_rev_row else None
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

            conn.execute(
                "UPDATE store_meta SET active_revision = ? WHERE singleton = 1",
                (resolved_oid,),
            )
            conn.execute(
                """
                INSERT INTO checkpoints (stream_key, cursor_json, last_success_at_us, last_reconciled_at_us)
                VALUES ('git:snapshot', '{}', ?, ?)
                ON CONFLICT(stream_key) DO UPDATE SET
                    last_success_at_us = excluded.last_success_at_us,
                    last_reconciled_at_us = excluded.last_reconciled_at_us
                """,
                (ts, ts),
            )

        return resolved_oid

    def resolve_prepared_revision(self, conn: sqlite3.Connection, requested_revision: str | None) -> str:
        """Resolve `revision` for search/context against the active prepared snapshot."""
        meta = conn.execute("SELECT active_revision FROM store_meta WHERE singleton = 1").fetchone()
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

        # Check recorded refs in SQLite
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

        raise CodeContextError(
            code="REVISION_NOT_PREPARED",
            message=f"Requested revision {requested_revision!r} is not the active prepared snapshot ({active_rev}). Run sync --revision {requested_revision} first.",
            retryable=False,
            details={
                "active_revision": str(active_rev),
                "requested_revision": requested_revision[:200],
                "resolved_revision": resolved_ref_oid,
            },
        )
