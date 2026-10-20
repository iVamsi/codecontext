import errno
import os
import stat
from pathlib import Path

from codecontext.errors import CodeContextError
from codecontext.git import parse_ls_tree_z, run_safe_git
from codecontext.identity import HEX_OID_RE, path_bytes_to_display, sha256_bytes, us_to_iso_z
from codecontext.models import Gap, WorkspaceState
from codecontext.store import Store, inspect_local_git_repository, now_us
from codecontext.syntax import MAX_PARSE_BYTES


def validate_workspace_binding(store: Store, workspace_path: Path) -> Path:
    """Validate that `workspace_path` belongs to `store` by comparing its `git-common-dir`
    realpath, filesystem device, and inode against `store_meta`.
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
            or int(meta["common_dir_device"]) != st_dev
            or int(meta["common_dir_inode"]) != st_ino
            or meta["object_format"] != obj_fmt
        ):
            raise CodeContextError(
                code="REPOSITORY_MISMATCH",
                message="Workspace does not belong to the repository bound to this store.",
                retryable=False,
            )
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


def _read_root_confined_regular_file(
    workspace_root: Path,
    rel_path_bytes: bytes,
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
    cur = root_real

    for part in parts[:-1]:
        comp = os.fsdecode(part)
        cur = cur / comp
        try:
            st = os.lstat(cur)
        except FileNotFoundError:
            return ("deleted", None, None)
        except OSError:
            return ("unavailable", None, None)

        if stat.S_ISLNK(st.st_mode):
            try:
                resolved_dir = cur.resolve()
            except OSError:
                return ("unavailable", None, None)
            if resolved_dir != root_real and root_real not in resolved_dir.parents:
                return ("symlink", None, None)
            cur = resolved_dir
        elif not stat.S_ISDIR(st.st_mode):
            return ("unavailable", None, None)

    leaf_path = cur / os.fsdecode(parts[-1])
    for _attempt in range(2):
        try:
            st1 = os.lstat(leaf_path)
        except FileNotFoundError:
            return ("deleted", None, None)
        except OSError:
            return ("unavailable", None, None)

        if stat.S_ISLNK(st1.st_mode):
            return ("symlink", None, 0o120000)
        if not stat.S_ISREG(st1.st_mode):
            return ("unavailable", None, None)
        if st1.st_size > MAX_PARSE_BYTES:
            return ("oversized", None, None)

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(leaf_path, flags)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.EMLINK):
                return ("symlink", None, 0o120000)
            if exc.errno == errno.ENOENT:
                return ("deleted", None, None)
            return ("unavailable", None, None)

        try:
            f_st1 = os.fstat(fd)
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

        if (
            f_st1.st_ino == st1.st_ino
            and f_st1.st_mtime_ns == f_st2.st_mtime_ns
            and f_st1.st_size == f_st2.st_size == total
        ):
            return ("ok", b"".join(chunks), _normalize_git_file_mode(f_st2.st_mode))

    return ("unavailable", None, None)


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

    head_matches_index = (head_rev == active_revision) if head_rev is not None else False
    if head_rev is not None and not head_matches_index:
        gaps.append(
            Gap(
                id=None,
                code="WORKSPACE_HEAD_DIFFERS",
                scope="workspace:head",
                message=f"Workspace HEAD ({head_rev[:12]}) differs from prepared memory revision ({active_revision[:12]}).",
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

    # Check unmerged stages (stage != 0)
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

    # Read staged blob bytes from workspace Git object store if needed
    staged_bytes: bytes | None = None
    if staged_blob_oid is not None:
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

    # 5. Independently compare target bytes and staged state against prepared memory snapshot (`index_blob_bytes` / `index_blob_oid`)
    differs_from_memory = False
    if target_state in ("staged_change", "unstaged_change", "staged_and_unstaged", "deleted"):
        differs_from_memory = True
    elif index_blob_oid is not None:
        if staged_blob_oid != index_blob_oid:
            differs_from_memory = True
        elif index_blob_bytes is not None and wt_bytes != index_blob_bytes:
            differs_from_memory = True

    if differs_from_memory and target_state != "untracked":
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
        True if (not head_matches_index or differs_from_memory or target_state != "unchanged") else None
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
