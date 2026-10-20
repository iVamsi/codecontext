"""Team snapshot publish and join.

The shared directory is a scrubbed snapshot. Each developer joins it into
their own store and rebinds that store to their checkout. A later publish
replaces the snapshot only after this store has joined the latest publish.
"""

import json
import os
import shutil
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

from codecontext.archive import execute_export
from codecontext.errors import CodeContextError
from codecontext.git import run_safe_git
from codecontext.identity import sha256_bytes
from codecontext.models import ExportReport, ExportRequest
from codecontext.store import Store, inspect_local_git_repository

TEAM_FILE = "team.json"


def project_remote_key(url: str) -> str:
    """Identity of a git remote: host, optional port, and path. Credentials are dropped."""
    raw = url.strip()
    if raw.startswith("git@"):
        host, sep, path = raw[4:].partition(":")
        if not sep or not host or not path:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Git remote URL is not a project address that can be shared.",
                retryable=False,
            )
        raw = f"ssh://git@{host}/{path.lstrip('/')}"
    parsed = urllib.parse.urlparse(raw)
    host = (parsed.hostname or "").lower().rstrip(".")
    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if not host or not path or any(part in ("", ".", "..") for part in path.split("/")):
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Git remote URL is not a project address that can be shared.",
            retryable=False,
        )
    hostport = f"{host}:{parsed.port}" if parsed.port else host
    return f"{hostport}/{path}"


def detect_project_remote(repo_path: Path) -> str:
    proc = run_safe_git(
        ["config", "--get", "remote.origin.url"],
        cwd=repo_path,
        check=False,
    )
    url = proc.stdout.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0 or not url:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="The checkout needs a git remote named origin before its memory can be shared.",
            retryable=False,
        )
    return project_remote_key(url)


def _read_team(share_dir: Path) -> dict[str, object]:
    path = share_dir / TEAM_FILE
    if share_dir.is_symlink() or not path.is_file() or path.is_symlink():
        raise CodeContextError(
            code="STORE_NOT_INITIALIZED",
            message="Shared memory is missing team.json.",
            retryable=False,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CodeContextError(
            code="STORE_NOT_INITIALIZED",
            message="Shared memory team.json could not be read.",
            retryable=False,
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("repository_id"), str)
        or not isinstance(payload.get("project_remote"), str)
        or not isinstance(payload.get("generation"), int)
        or not isinstance(payload.get("published_at"), str)
    ):
        raise CodeContextError(
            code="STORE_VERSION_UNSUPPORTED",
            message="Shared memory team.json has an unsupported shape.",
            retryable=False,
        )
    return payload


def _store_generation(store: Store) -> int:
    with store.connect_read() as conn:
        row = conn.execute("SELECT generation FROM store_meta WHERE singleton = 1").fetchone()
    if row is None:
        raise CodeContextError(
            code="STORE_NOT_INITIALIZED",
            message="Store is missing store_meta.",
            retryable=False,
        )
    return int(row["generation"])


def _report_for_existing(share_dir: Path, team: dict[str, object]) -> ExportReport:
    manifest = (share_dir / "manifest.json").read_bytes()
    return ExportReport(
        repository_id=str(team["repository_id"]),
        revision=team.get("active_revision") if isinstance(team.get("active_revision"), str) else None,
        generation=int(team["generation"]),
        access_epoch=0,
        destination=str(share_dir),
        include_git_objects=(share_dir / "repository.git").is_dir(),
        file_count=0,
        total_bytes=0,
        manifest_sha256=sha256_bytes(manifest),
        warnings=["Shared memory is already at this generation."],
    )


def _replace_tree(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Refusing to replace a symlinked share directory.",
            retryable=False,
        )
    backup = destination.parent / f".{destination.name}.replaced"
    if backup.exists() or backup.is_symlink():
        shutil.rmtree(backup)
    os.rename(destination, backup)
    try:
        os.rename(source, destination)
    except Exception:
        os.rename(backup, destination)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _write_team(directory: Path, payload: dict[str, object]) -> None:
    (directory / TEAM_FILE).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _local_team(store: Store) -> dict[str, object] | None:
    path = store.store_path / TEAM_FILE
    if not path.is_file() or path.is_symlink():
        return None
    return _read_team(store.store_path)


def execute_publish(store: Store, destination: Path) -> ExportReport:
    if not store.config.repo_path:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Join this memory to a checkout before publishing an update.",
            retryable=False,
        )
    remote_key = detect_project_remote(Path(store.config.repo_path))
    local_gen = _store_generation(store)
    dest = destination.expanduser()
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or not dest.is_dir():
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Share destination must be a real directory.",
                retryable=False,
            )
        current = _read_team(dest)
        if current["repository_id"] != store.config.repository_id:
            raise CodeContextError(
                code="REPOSITORY_MISMATCH",
                message="Share directory belongs to a different project memory.",
                retryable=False,
            )
        if current["project_remote"] != remote_key:
            raise CodeContextError(
                code="REPOSITORY_MISMATCH",
                message="Share directory is for a different git remote.",
                retryable=False,
            )
        local_team = _local_team(store)
        local_published = str(local_team["published_at"]) if local_team else None
        share_published = str(current["published_at"])
        # ponytail: UTC timestamps order publishes; skewed clocks can reorder them. Upgrade: a shared counter.
        share_is_newer = local_published is not None and (
            share_published > local_published
            or (
                share_published == local_published
                and int(current["generation"]) != int(local_team["generation"])
            )
        )
        unseen_share = local_published is None and int(current["generation"]) != local_gen
        if share_is_newer or unseen_share:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Shared memory is newer. Join it before publishing.",
                retryable=False,
            )
        same_snapshot = int(current["generation"]) == local_gen and (
            local_published is None or local_published == share_published
        )
        if same_snapshot:
            if local_published is None:
                _write_team(store.store_path, current)
            return _report_for_existing(dest, current)

    parent = dest.parent.expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{dest.name}.publishing"
    if staging.exists() or staging.is_symlink():
        shutil.rmtree(staging)
    report = execute_export(store, ExportRequest(destination=staging, include_git_objects=True))
    team_payload = {
        "active_revision": report.revision,
        "generation": report.generation,
        "project_remote": remote_key,
        "published_at": datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "repository_id": report.repository_id,
        "schema_version": 1,
    }
    _write_team(staging, team_payload)
    if dest.exists():
        _replace_tree(staging, dest)
    else:
        os.rename(staging, dest)
    _write_team(store.store_path, team_payload)
    return report.model_copy(update={"destination": str(dest)})


def _copy_share(share_dir: Path, store_path: Path) -> None:
    if store_path.exists() or store_path.is_symlink():
        if store_path.is_symlink():
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Refusing to replace a symlinked store path.",
                retryable=False,
            )
        shutil.rmtree(store_path)
    shutil.copytree(share_dir, store_path, symlinks=False)
    manifest = store_path / "manifest.json"
    if manifest.exists() and not manifest.is_symlink():
        manifest.unlink()


def _rebind(store_path: Path, repo_path: Path) -> None:
    resolved_repo, common_real, st_dev, st_ino, _obj_fmt = inspect_local_git_repository(repo_path)
    store_json_path = store_path / "store.json"
    raw = json.loads(store_json_path.read_text(encoding="utf-8"))
    raw["read_only_archive"] = False
    raw["repo_path"] = str(resolved_repo)
    store_json_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    store = Store.open(store_path, read_only=False)
    try:
        with store.write_transaction(bump_generation=False) as conn:
            conn.execute(
                """
                UPDATE store_meta
                SET local_repository_path = ?,
                    common_dir_realpath = ?,
                    common_dir_device = ?,
                    common_dir_inode = ?,
                    principal_id = NULL
                WHERE singleton = 1
                """,
                (str(resolved_repo), str(common_real), st_dev, st_ino),
            )
    finally:
        store.close()


def execute_join(store_path: Path, share_dir: Path, repo_path: Path) -> None:
    raw_share = share_dir.expanduser()
    if raw_share.is_symlink():
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Refusing to join through a symlinked share directory.",
            retryable=False,
        )
    share = raw_share.resolve()
    team = _read_team(share)
    remote_key = detect_project_remote(repo_path)
    if team["project_remote"] != remote_key:
        raise CodeContextError(
            code="REPOSITORY_MISMATCH",
            message="This checkout's origin does not match the shared memory.",
            retryable=False,
        )
    resolved_repo, _common, _dev, _ino, _fmt = inspect_local_git_repository(repo_path)
    raw_store = store_path.expanduser()
    if raw_store.is_symlink() or (raw_store.exists() and not raw_store.is_dir()):
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Store path must be a real directory.",
            retryable=False,
        )
    dest = raw_store.resolve()
    is_dot = dest.name == ".codecontext" and dest.parent == resolved_repo
    if not is_dot and (
        dest == resolved_repo
        or resolved_repo in dest.parents
        or share == dest
        or share in dest.parents
        or dest in share.parents
    ):
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Join the shared memory outside the checkout, at <repo>/.codecontext, and not inside the share directory.",
            retryable=False,
        )

    if (dest / "store.json").is_file():
        raw = json.loads((dest / "store.json").read_text(encoding="utf-8"))
        if raw.get("repository_id") != team["repository_id"]:
            raise CodeContextError(
                code="REPOSITORY_MISMATCH",
                message="Local store belongs to a different project memory.",
                retryable=False,
            )
        existing = Store.open(dest, read_only=True)
        try:
            local_gen = _store_generation(existing)
        finally:
            existing.close()
        local_published = None
        local_team = dest / TEAM_FILE
        if local_team.is_file() and not local_team.is_symlink():
            local_published = str(_read_team(dest)["published_at"])
        share_published = str(team["published_at"])
        if local_published is not None and local_published > share_published:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Shared memory is older than this store. Publish the newer memory.",
                retryable=False,
            )
        if local_published == share_published and local_gen >= int(team["generation"]):
            _rebind(dest, repo_path)
            return

    _copy_share(share, dest)
    _rebind(dest, repo_path)
