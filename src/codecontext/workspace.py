import contextlib
import errno
import json
import os
import stat
import struct
from dataclasses import dataclass
from pathlib import Path

from codecontext.errors import CodeContextError
from codecontext.git import git_blob_oid, parse_ls_tree_z, run_safe_git
from codecontext.identity import HEX_OID_RE, path_bytes_to_display, sha256_bytes, us_to_iso_z
from codecontext.models import Gap, WorkspaceState
from codecontext.store import Store, inspect_local_git_repository, now_us
from codecontext.syntax import MAX_PARSE_BYTES


def validate_workspace_binding(store: Store, workspace_path: Path) -> Path:
    """Validate that `workspace_path` belongs to `store` by comparing its `git-common-dir`
    realpath and inode (tolerating remount `st_dev` shifts) against `store_meta`.
    Linked worktrees sharing the original repository's common-dir pass; unrelated clones fail.
    Exports (`read_only_archive=True` or null binding) fail with `REPOSITORY_MISMATCH`.
    """
    if store.config.read_only_archive:
        raise CodeContextError(
            code="REPOSITORY_MISMATCH",
            message="Read-only archive stores have no machine workspace binding.",
            retryable=False,
        )

    resolved_ws, common_real, st_dev, st_ino, obj_fmt = inspect_local_git_repository(
        workspace_path
    )
    remount_dev_changed = False
    with store.connect_read() as conn:
        meta = conn.execute(
            "SELECT common_dir_realpath, common_dir_device, common_dir_inode, object_format "
            "FROM store_meta WHERE singleton = 1"
        ).fetchone()
        if not meta or not meta["common_dir_realpath"]:
            raise CodeContextError(
                code="REPOSITORY_MISMATCH",
                message="Store has no local common-directory binding for workspace inspection.",
                retryable=False,
            )
        if (
            meta["common_dir_realpath"] != str(common_real)
            or int(meta["common_dir_inode"]) != st_ino
            or meta["object_format"] != obj_fmt
        ):
            raise CodeContextError(
                code="REPOSITORY_MISMATCH",
                message="Workspace does not belong to the repository bound to this store.",
                retryable=False,
            )
        if meta["common_dir_device"] is not None and int(meta["common_dir_device"]) != st_dev:
            remount_dev_changed = True

    if remount_dev_changed and not store.read_only:
        with contextlib.suppress(Exception), store.write_transaction(bump_generation=False) as conn:
            conn.execute("UPDATE store_meta SET common_dir_device = ? WHERE singleton = 1", (st_dev,))
    return resolved_ws


def _parse_ls_files_stage_z(raw_stdout: bytes) -> list[tuple[int, str, int, bytes]]:
    """Parse `git ls-files --stage -z` records into `(mode, oid, stage, path_bytes)`."""
    if not raw_stdout:
        return []
    entries: list[tuple[int, str, int, bytes]] = []
    for rec in raw_stdout.split(b"\x00"):
        if not rec:
            continue
        tab_idx = rec.find(b"\t")
        if tab_idx == -1:
            continue
        meta = rec[:tab_idx].decode("ascii", errors="replace")
        path_b = rec[tab_idx + 1 :]
        parts = meta.split()
        if len(parts) == 3:
            mode = int(parts[0], 8)
            oid = parts[1]
            stage = int(parts[2])
            entries.append((mode, oid, stage, path_b))
    return entries


def _normalize_git_file_mode(st_mode: int) -> int:
    if stat.S_ISLNK(st_mode):
        return 0o120000
    if st_mode & 0o111:
        return 0o100755
    return 0o100644


def _resolve_root_confined_parent_dir(
    root_real: Path,
    dir_parts: list[bytes],
    *,
    dir_cache: dict[Path, Path | None] | None = None,
) -> tuple[str, Path | None]:
    cur = root_real
    for part in dir_parts:
        comp = os.fsdecode(part)
        next_cur = cur / comp
        if dir_cache is not None and next_cur in dir_cache:
            cached = dir_cache[next_cur]
            if cached is None:
                return ("deleted", None)
            cur = cached
            continue

        try:
            st = os.lstat(next_cur)
        except FileNotFoundError:
            if dir_cache is not None:
                dir_cache[next_cur] = None
            return ("deleted", None)
        except OSError:
            return ("unavailable", None)

        if stat.S_ISLNK(st.st_mode):
            try:
                resolved_dir = next_cur.resolve()
            except OSError:
                return ("unavailable", None)
            if resolved_dir != root_real and root_real not in resolved_dir.parents:
                return ("symlink", None)
            if dir_cache is not None:
                dir_cache[next_cur] = resolved_dir
            cur = resolved_dir
        elif not stat.S_ISDIR(st.st_mode):
            return ("unavailable", None)
        else:
            if dir_cache is not None:
                dir_cache[next_cur] = next_cur
            cur = next_cur
    return ("ok", cur)


def _read_root_confined_symlink_target(
    workspace_root: Path,
    rel_path_bytes: bytes,
    *,
    dir_cache: dict[Path, Path | None] | None = None,
) -> bytes | None:
    """Read symlink target bytes via `os.readlink` without following the symlink, provided all
    parent directories stay confined inside `workspace_root`.
    """
    parts = [p for p in rel_path_bytes.split(b"/") if p]
    if not parts:
        return None
    root_real = workspace_root.resolve()
    p_status, parent_dir = _resolve_root_confined_parent_dir(
        root_real, parts[:-1], dir_cache=dir_cache
    )
    if p_status != "ok" or parent_dir is None:
        return None
    leaf_path = parent_dir / os.fsdecode(parts[-1])
    try:
        st = os.lstat(leaf_path)
        if not stat.S_ISLNK(st.st_mode):
            return None
        target_str = os.readlink(leaf_path)
        return os.fsencode(target_str)
    except OSError:
        return None


# Large monorepo indexes reach tens of MiB; anything past this is read the slow way.
MAX_GIT_INDEX_BYTES = 256 * 1024 * 1024
_INDEX_STAT_FIELDS = struct.Struct(">10I")
_INDEX_FLAG_EXTENDED = 0x4000
_INDEX_EXT_SKIP_WORKTREE = 0x4000
_INDEX_EXT_INTENT_TO_ADD = 0x2000
# Real paths stay far below this; without a cap, v4 prefix sharing lets each entry grow the path by one byte.
MAX_GIT_INDEX_PATH_BYTES = 4096


@dataclass(frozen=True, slots=True)
class IndexStatEntry:
    ctime_ns: int
    mtime_ns: int
    size: int
    mode: int
    oid: str


def _read_index_offset_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Decode git's offset varint (index v4 path-prefix strip length)."""
    c = data[pos]
    pos += 1
    val = c & 0x7F
    while c & 0x80:
        if val > 1 << 32:
            raise ValueError("index varint overflow")
        c = data[pos]
        pos += 1
        val = ((val + 1) << 7) | (c & 0x7F)
    return val, pos


def parse_git_index(data: bytes, *, oid_bytes: int) -> dict[bytes, IndexStatEntry]:
    """Parse stage-0 entries of a DIRC index (versions 2-4). Raises ValueError on malformed input.

    Intent-to-add and skip-worktree entries are left out: their stat data says nothing about
    the working-tree bytes.
    """
    if len(data) < 12 + oid_bytes or data[:4] != b"DIRC":
        raise ValueError("not a git index")
    version, count = struct.unpack_from(">II", data, 4)
    if version not in (2, 3, 4):
        raise ValueError("unsupported index version")
    end = len(data) - oid_bytes
    fixed = _INDEX_STAT_FIELDS.size + oid_bytes + 2
    if count > (end - 12) // fixed:
        raise ValueError("index entry count exceeds file size")

    out: dict[bytes, IndexStatEntry] = {}
    pos = 12
    prev_path = b""
    for _ in range(count):
        start = pos
        if pos + fixed > end:
            raise ValueError("truncated index entry")
        ctime_s, ctime_n, mtime_s, mtime_n, _dev, _ino, mode, _uid, _gid, size = (
            _INDEX_STAT_FIELDS.unpack_from(data, pos)
        )
        pos += _INDEX_STAT_FIELDS.size
        oid = data[pos : pos + oid_bytes].hex()
        pos += oid_bytes
        (flags,) = struct.unpack_from(">H", data, pos)
        pos += 2
        ext_flags = 0
        if flags & _INDEX_FLAG_EXTENDED:
            if version < 3:
                raise ValueError("extended flags in index v2")
            (ext_flags,) = struct.unpack_from(">H", data, pos)
            pos += 2
        if version == 4:
            strip, pos = _read_index_offset_varint(data, pos)
            if strip > len(prev_path):
                raise ValueError("index v4 prefix strip exceeds previous path")
            nul = data.index(b"\0", pos, end)
            if len(prev_path) - strip + nul - pos > MAX_GIT_INDEX_PATH_BYTES:
                raise ValueError("index entry path too long")
            path = prev_path[: len(prev_path) - strip] + data[pos:nul]
            pos = nul + 1
        else:
            nul = data.index(b"\0", pos, end)
            if nul - pos > MAX_GIT_INDEX_PATH_BYTES:
                raise ValueError("index entry path too long")
            path = data[pos:nul]
            pos = start + ((nul - start + 8) & ~7)
        if pos > end or not path:
            raise ValueError("index entry overruns file")
        prev_path = path
        stage = (flags >> 12) & 0x3
        if stage == 0 and not ext_flags & (_INDEX_EXT_SKIP_WORKTREE | _INDEX_EXT_INTENT_TO_ADD):
            out[path] = IndexStatEntry(
                ctime_ns=ctime_s * 1_000_000_000 + ctime_n,
                mtime_ns=mtime_s * 1_000_000_000 + mtime_n,
                size=size,
                mode=mode,
                oid=oid,
            )
    return out


def read_git_index_stat_cache(
    index_path: Path, *, oid_bytes: int
) -> tuple[dict[bytes, IndexStatEntry], int] | None:
    """Return `(entries, index_mtime_ns)`, or None when the index is missing, too big, or malformed."""
    try:
        with open(index_path, "rb") as fh:
            index_mtime_ns = os.fstat(fh.fileno()).st_mtime_ns
            data = fh.read(MAX_GIT_INDEX_BYTES + 1)
    except OSError:
        return None
    if len(data) > MAX_GIT_INDEX_BYTES:
        return None
    try:
        return parse_git_index(data, oid_bytes=oid_bytes), index_mtime_ns
    except (ValueError, struct.error, IndexError):
        return None


def index_entry_matches_stat(entry: IndexStatEntry, st: os.stat_result, index_mtime_ns: int) -> bool:
    """True when the working file can be trusted to still hold the indexed blob.

    A file whose mtime is not older than the index itself is racily clean: it may have been
    rewritten within the timestamp granularity after git recorded its stat data, so it is re-read.
    """
    return (
        stat.S_ISREG(st.st_mode)
        and st.st_mtime_ns == entry.mtime_ns
        and (st.st_size & 0xFFFFFFFF) == entry.size
        and _normalize_git_file_mode(st.st_mode) == entry.mode
        and entry.mtime_ns < index_mtime_ns
    )


def _lstat_root_confined(
    workspace_root: Path,
    rel_path_bytes: bytes,
    *,
    dir_cache: dict[Path, Path | None] | None = None,
) -> os.stat_result | None:
    parts = [p for p in rel_path_bytes.split(b"/") if p]
    if not parts:
        return None
    p_status, parent_dir = _resolve_root_confined_parent_dir(
        workspace_root.resolve(), parts[:-1], dir_cache=dir_cache
    )
    if p_status != "ok" or parent_dir is None:
        return None
    try:
        return os.lstat(parent_dir / os.fsdecode(parts[-1]))
    except OSError:
        return None


_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
# O_NONBLOCK keeps a FIFO planted at a tracked path from hanging the open.
_FILE_OPEN_FLAGS = os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC | getattr(os, "O_NONBLOCK", 0)
_DIR_OPEN_FLAGS = os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC | _O_DIRECTORY


def open_repo_file_no_follow(repo_fd: int, rel_path: bytes) -> int:
    """Open `rel_path` under the directory `repo_fd` for reading without following any symlink.

    Each component is opened relative to the previous one, so a parent swapped for a symlink
    after a path check cannot redirect the read. A symlink anywhere on the path raises
    `OSError(ELOOP)`; the caller owns the returned fd.
    """
    parts = [p for p in rel_path.split(b"/") if p]
    if not parts:
        raise OSError(errno.ENOENT, "empty repository path")
    fd = os.dup(repo_fd)
    try:
        for part in parts[:-1]:
            try:
                next_fd = os.open(part, _DIR_OPEN_FLAGS, dir_fd=fd)
            except OSError as exc:
                # macOS reports ENOTDIR, not ELOOP, for O_DIRECTORY | O_NOFOLLOW on a symlink.
                if exc.errno == errno.ENOTDIR and stat.S_ISLNK(
                    os.stat(part, dir_fd=fd, follow_symlinks=False).st_mode
                ):
                    raise OSError(errno.ELOOP, "symlink in repository path") from exc
                raise
            os.close(fd)
            fd = next_fd
        return os.open(parts[-1], _FILE_OPEN_FLAGS, dir_fd=fd)
    finally:
        os.close(fd)


def _read_root_confined_regular_file(
    workspace_root: Path,
    rel_path_bytes: bytes,
    *,
    dir_cache: dict[Path, Path | None] | None = None,
) -> tuple[str, bytes | None, int | None]:
    """Safely read a regular file inside `workspace_root` without following symlinks outside the root.
    Retries once if file changes while being read.
    Returns `(status, raw_bytes, normalized_mode)` where status is:
    - `'ok'`
    - `'deleted'`
    - `'symlink'`
    - `'oversized'`
    - `'unavailable'`
    """
    parts = [p for p in rel_path_bytes.split(b"/") if p]
    if not parts:
        return ("unavailable", None, None)

    root_real = workspace_root.resolve()
    p_status, cur = _resolve_root_confined_parent_dir(root_real, parts[:-1], dir_cache=dir_cache)
    if p_status != "ok" or cur is None:
        return (p_status, None, 0o120000 if p_status == "symlink" else None)

    # The lexical check resolved in-root parent symlinks; the fd walk re-checks that real path
    # with no-follow opens, so anything swapped in since is refused rather than followed.
    real_rel = b"/".join([*(os.fsencode(c) for c in cur.relative_to(root_real).parts), parts[-1]])
    try:
        root_fd = os.open(root_real, _DIR_OPEN_FLAGS & ~_O_NOFOLLOW)
    except OSError:
        return ("unavailable", None, None)
    try:
        for _attempt in range(2):
            try:
                fd = open_repo_file_no_follow(root_fd, real_rel)
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.EMLINK):
                    return ("symlink", None, 0o120000)
                if exc.errno == errno.ENOENT:
                    return ("deleted", None, None)
                return ("unavailable", None, None)

            try:
                f_st1 = os.fstat(fd)
                if not stat.S_ISREG(f_st1.st_mode):
                    return ("unavailable", None, None)
                if f_st1.st_size > MAX_PARSE_BYTES:
                    return ("oversized", None, None)
                chunks: list[bytes] = []
                total = 0
                while True:
                    buf = os.read(fd, 65536)
                    if not buf:
                        break
                    total += len(buf)
                    if total > MAX_PARSE_BYTES:
                        return ("oversized", None, None)
                    chunks.append(buf)
                f_st2 = os.fstat(fd)
            except OSError:
                return ("unavailable", None, None)
            finally:
                os.close(fd)

            if f_st1.st_mtime_ns == f_st2.st_mtime_ns and f_st1.st_size == f_st2.st_size == total:
                return ("ok", b"".join(chunks), _normalize_git_file_mode(f_st2.st_mode))
    finally:
        os.close(root_fd)

    return ("unavailable", None, None)


def _lookup_workspace_base_revision(store: Store, active_revision: str) -> str | None:
    with store.connect_read() as conn:
        meta = conn.execute("SELECT * FROM store_meta WHERE singleton = 1").fetchone()
        if meta and str(meta["active_revision"] or "") == active_revision:
            if "workspace_base_revision" in meta.keys() and meta["workspace_base_revision"]:
                return str(meta["workspace_base_revision"])
            ckpt = conn.execute(
                "SELECT cursor_json FROM checkpoints WHERE stream_key = 'git:snapshot'"
            ).fetchone()
            if ckpt and ckpt[0]:
                with contextlib.suppress(Exception):
                    base_rev = json.loads(str(ckpt[0])).get("workspace_base_revision")
                    if isinstance(base_rev, str) and base_rev:
                        return base_rev
    return None


def _is_ignored(ws_root: Path, rel_path: bytes, empty_hooks: Path) -> bool:
    """True when git ignores the untracked `rel_path`; a failed check counts as ignored."""
    proc = run_safe_git(
        ["check-ignore", "--stdin", "-z"],
        cwd=ws_root,
        empty_hooks_dir=empty_hooks,
        stdin_bytes=rel_path + b"\0",
        check=False,
    )
    # Exit 1 means "not ignored"; 0 is ignored and anything else is an error.
    return proc.returncode != 1


def inspect_workspace(
    store: Store,
    workspace_path: Path | None,
    *,
    active_revision: str,
    target_path_bytes: bytes | None,
    index_blob_oid: str | None,
    index_blob_bytes: bytes | None,
) -> tuple[WorkspaceState, list[Gap]]:
    """Inspect workspace state relative to workspace HEAD and prepared memory snapshot without
    invoking `git status`, `diff-files`, index refresh, hooks, or clean/smudge filters.
    Returns `(WorkspaceState, transient_gaps)`.
    """
    # Compared case-insensitively because case-insensitive filesystems open `.GIT/config` as
    # `.git/config`; any component counts so submodule `.git` dirs are covered too.
    if target_path_bytes is not None and any(
        part.lower() == b".git" for part in target_path_bytes.split(b"/")
    ):
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Target path must not point inside a .git directory.",
            retryable=False,
        )
    if workspace_path is None:
        return (
            WorkspaceState(
                checked=False,
                observed_at=None,
                head_revision=None,
                head_matches_index=None,
                repository_diverged=None,
                target_state="not_checked",
                index_blob_oid=index_blob_oid,
                staged_blob_oid=None,
                working_content_hash=None,
            ),
            [],
        )

    observed_iso = us_to_iso_z(now_us())
    empty_hooks = Path(os.devnull)
    gaps: list[Gap] = []

    try:
        ws_root = validate_workspace_binding(store, workspace_path)
    except CodeContextError as exc:
        if exc.code == "REPOSITORY_MISMATCH":
            raise
        gaps.append(
            Gap(
                id=None,
                code="WORKSPACE_UNAVAILABLE",
                scope="workspace:root",
                message=exc.message,
                retryable=False,
            )
        )
        return (
            WorkspaceState(
                checked=True,
                observed_at=observed_iso,
                head_revision=None,
                head_matches_index=None,
                repository_diverged=None,
                target_state="unavailable",
                index_blob_oid=index_blob_oid,
                staged_blob_oid=None,
                working_content_hash=None,
            ),
            gaps,
        )

    ws_base_revision = _lookup_workspace_base_revision(store, active_revision)
    expected_head = ws_base_revision or active_revision

    # Resolve workspace HEAD
    head_proc = run_safe_git(
        ["rev-parse", "--verify", "--end-of-options", "HEAD^{commit}"],
        cwd=ws_root,
        empty_hooks_dir=empty_hooks,
        check=False,
    )
    head_rev: str | None = None
    if head_proc.returncode == 0:
        cand = head_proc.stdout.decode("ascii", errors="replace").strip()
        if HEX_OID_RE.match(cand):
            head_rev = cand

    head_matches_index = (head_rev == expected_head) if head_rev is not None else False
    if head_rev is not None and not head_matches_index:
        gaps.append(
            Gap(
                id=None,
                code="WORKSPACE_HEAD_DIFFERS",
                scope="workspace:head",
                message=f"Workspace HEAD ({head_rev[:12]}) differs from prepared memory revision ({expected_head[:12]}).",
                retryable=False,
            )
        )

    if target_path_bytes is None:
        return (
            WorkspaceState(
                checked=True,
                observed_at=observed_iso,
                head_revision=head_rev,
                head_matches_index=head_matches_index,
                repository_diverged=(not head_matches_index) if not head_matches_index else None,
                target_state="not_checked",
                index_blob_oid=index_blob_oid,
                staged_blob_oid=None,
                working_content_hash=None,
            ),
            gaps,
        )

    disp_target, _ = path_bytes_to_display(target_path_bytes)
    pathspec_arg = f":(literal){os.fsdecode(target_path_bytes)}"

    # 1. Inspect workspace HEAD entry via `ls-tree -z HEAD -- :(literal)<path>`
    head_entry_oid: str | None = None
    head_entry_mode: int | None = None
    if head_rev is not None:
        lst_proc = run_safe_git(
            ["ls-tree", "-z", "--full-tree", head_rev, "--", pathspec_arg],
            cwd=ws_root,
            empty_hooks_dir=empty_hooks,
            check=False,
        )
        if lst_proc.returncode == 0:
            entries = parse_ls_tree_z(lst_proc.stdout)
            for e in entries:
                if e.path == target_path_bytes:
                    head_entry_oid = e.oid
                    head_entry_mode = e.mode
                    break

    # 2. Inspect workspace Git index via `ls-files --stage -z -- :(literal)<path>`
    lsf_proc = run_safe_git(
        ["ls-files", "--stage", "-z", "--", pathspec_arg],
        cwd=ws_root,
        empty_hooks_dir=empty_hooks,
        check=False,
    )
    stage_entries = [
        item
        for item in _parse_ls_files_stage_z(lsf_proc.stdout)
        if item[3] == target_path_bytes
    ]

    if any(item[2] != 0 for item in stage_entries):
        gaps.append(
            Gap(
                id=None,
                code="WORKSPACE_UNMERGED",
                scope=f"workspace:target:{disp_target}",
                message=f"Target {disp_target} has unmerged index stages in workspace.",
                retryable=False,
            )
        )
        return (
            WorkspaceState(
                checked=True,
                observed_at=observed_iso,
                head_revision=head_rev,
                head_matches_index=head_matches_index,
                repository_diverged=True,
                target_state="unavailable",
                index_blob_oid=index_blob_oid,
                staged_blob_oid=None,
                working_content_hash=None,
            ),
            gaps,
        )

    staged_mode: int | None = None
    staged_blob_oid: str | None = None
    if stage_entries:
        staged_mode, staged_blob_oid, _, _ = stage_entries[0]

    if staged_mode == 0o160000:
        gaps.append(
            Gap(
                id=None,
                code="SUBMODULE_NOT_FETCHED",
                scope=f"workspace:target:{disp_target}",
                message=f"Workspace target {disp_target} is a submodule gitlink.",
                retryable=False,
            )
        )
        return (
            WorkspaceState(
                checked=True,
                observed_at=observed_iso,
                head_revision=head_rev,
                head_matches_index=head_matches_index,
                repository_diverged=True if not head_matches_index else None,
                target_state="unavailable",
                index_blob_oid=index_blob_oid,
                staged_blob_oid=staged_blob_oid,
                working_content_hash=None,
            ),
            gaps,
        )

    staged_bytes: bytes | None = None
    if staged_blob_oid is not None:
        size_proc = run_safe_git(
            ["cat-file", "-s", staged_blob_oid],
            cwd=ws_root,
            empty_hooks_dir=empty_hooks,
            check=False,
        )
        staged_size: int | None = None
        if size_proc.returncode == 0:
            with contextlib.suppress(ValueError):
                staged_size = int(size_proc.stdout.decode("ascii", errors="replace").strip())
        if staged_size is not None and staged_size <= MAX_PARSE_BYTES:
            cat_proc = run_safe_git(
                ["cat-file", "blob", staged_blob_oid],
                cwd=ws_root,
                empty_hooks_dir=empty_hooks,
                check=False,
            )
            if cat_proc.returncode == 0:
                staged_bytes = cat_proc.stdout

    # 3. Read working tree file bytes safely
    wt_status, wt_bytes, wt_mode = _read_root_confined_regular_file(ws_root, target_path_bytes)
    working_hash = sha256_bytes(wt_bytes) if wt_bytes is not None else None

    if wt_status == "symlink":
        gaps.append(
            Gap(
                id=None,
                code="SYMLINK_NOT_FOLLOWED",
                scope=f"workspace:target:{disp_target}",
                message=f"Workspace target {disp_target} is a symlink or traverses an external symlink.",
                retryable=False,
            )
        )
        return (
            WorkspaceState(
                checked=True,
                observed_at=observed_iso,
                head_revision=head_rev,
                head_matches_index=head_matches_index,
                repository_diverged=True if not head_matches_index else None,
                target_state="unavailable",
                index_blob_oid=index_blob_oid,
                staged_blob_oid=staged_blob_oid,
                working_content_hash=None,
            ),
            gaps,
        )

    if wt_status == "oversized":
        gaps.append(
            Gap(
                id=None,
                code="BLOB_TOO_LARGE",
                scope=f"workspace:target:{disp_target}",
                message=f"Workspace target {disp_target} exceeds 2 MiB read limit.",
                retryable=False,
            )
        )
        return (
            WorkspaceState(
                checked=True,
                observed_at=observed_iso,
                head_revision=head_rev,
                head_matches_index=head_matches_index,
                repository_diverged=True if not head_matches_index else None,
                target_state="unavailable",
                index_blob_oid=index_blob_oid,
                staged_blob_oid=staged_blob_oid,
                working_content_hash=None,
            ),
            gaps,
        )

    if wt_status == "unavailable":
        gaps.append(
            Gap(
                id=None,
                code="WORKSPACE_UNAVAILABLE",
                scope=f"workspace:target:{disp_target}",
                message=f"Workspace target {disp_target} could not be read reliably.",
                retryable=True,
            )
        )
        return (
            WorkspaceState(
                checked=True,
                observed_at=observed_iso,
                head_revision=head_rev,
                head_matches_index=head_matches_index,
                repository_diverged=True if not head_matches_index else None,
                target_state="unavailable",
                index_blob_oid=index_blob_oid,
                staged_blob_oid=staged_blob_oid,
                working_content_hash=None,
            ),
            gaps,
        )

    # 4. Determine `target_state` relative to workspace HEAD
    if head_entry_oid is None and staged_blob_oid is None:
        if wt_status == "ok":
            target_state = "untracked"
            if _is_ignored(ws_root, target_path_bytes, empty_hooks):
                working_hash = None
            gaps.append(
                Gap(
                    id=None,
                    code="UNTRACKED_TARGET",
                    scope=f"workspace:target:{disp_target}",
                    message=f"Target {disp_target} is untracked in the workspace and has no indexed historical lineage.",
                    retryable=False,
                )
            )
        else:
            target_state = "deleted"
    elif wt_status == "deleted":
        target_state = "deleted"
    else:
        has_staged_change = (staged_blob_oid != head_entry_oid) or (staged_mode != head_entry_mode)
        has_unstaged_change = (
            staged_bytes is None
            or wt_bytes != staged_bytes
            or (wt_mode is not None and staged_mode in (0o100644, 0o100755) and wt_mode != staged_mode)
        )
        if has_staged_change and has_unstaged_change:
            target_state = "staged_and_unstaged"
        elif has_staged_change:
            target_state = "staged_change"
        elif has_unstaged_change:
            target_state = "unstaged_change"
        else:
            target_state = "unchanged"

    # 5. Compare target bytes and staged state against prepared memory snapshot (`index_blob_oid` / `index_blob_bytes`)
    oid_len = len(active_revision) if len(active_revision) in (40, 64) else 40
    wt_blob_oid = git_blob_oid(wt_bytes, oid_len=oid_len) if wt_bytes is not None else None
    differs_from_memory = False
    if target_state == "deleted":
        differs_from_memory = index_blob_oid is not None
    elif target_state == "untracked":
        # If untracked file was included in workspace snapshot via include_untracked=True, compare against index_blob_oid
        if index_blob_oid is not None and wt_blob_oid != index_blob_oid:
            differs_from_memory = True
    elif ws_base_revision is not None:
        # Memory snapshot is a workspace snapshot: working-tree bytes are authoritative
        if index_blob_oid is None or wt_blob_oid != index_blob_oid:
            differs_from_memory = True
        elif index_blob_bytes is not None and wt_bytes != index_blob_bytes:
            differs_from_memory = True
    else:
        # Memory snapshot is a commit snapshot: any working-tree or staged divergence from index_blob_oid differs
        if index_blob_oid is None:
            differs_from_memory = True
        elif staged_blob_oid != index_blob_oid or wt_blob_oid != index_blob_oid:
            differs_from_memory = True
        elif index_blob_bytes is not None and wt_bytes != index_blob_bytes:
            differs_from_memory = True

    if differs_from_memory and (target_state != "untracked" or index_blob_oid is not None):
        gaps.append(
            Gap(
                id=None,
                code="WORKSPACE_TARGET_DIFFERS",
                scope=f"workspace:target:{disp_target}",
                message=f"Workspace target {disp_target} ({target_state}) differs from the prepared memory snapshot; returned code anchors remain pinned to {active_revision[:12]}.",
                retryable=False,
            )
        )

    repo_diverged: bool | None = (
        True
        if (
            not head_matches_index
            or differs_from_memory
            or (ws_base_revision is None and target_state != "unchanged")
        )
        else None
    )

    return (
        WorkspaceState(
            checked=True,
            observed_at=observed_iso,
            head_revision=head_rev,
            head_matches_index=head_matches_index,
            repository_diverged=repo_diverged,
            target_state=target_state,  # type: ignore[arg-type]
            index_blob_oid=index_blob_oid,
            staged_blob_oid=staged_blob_oid,
            working_content_hash=working_hash,
        ),
        gaps,
    )
