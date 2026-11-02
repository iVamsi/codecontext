"""Team snapshot publish and join.

The shared directory is a scrubbed snapshot. Each developer joins it into
their own store and rebinds that store to their checkout. A later publish
replaces the snapshot only after this store has joined the latest publish.
"""

import functools
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import urllib.parse
import zlib
from pathlib import Path

from codecontext.archive import execute_export, sanitized_bare_git_config
from codecontext.deadline import remaining_timeout
from codecontext.errors import CodeContextError
from codecontext.git import run_safe_git
from codecontext.identity import (
    canonical_json,
    datetime_to_us,
    parse_iso_utc,
    sha256_bytes,
    us_to_iso_z,
)
from codecontext.models import ExportReport, ExportRequest, TeamMergeReport
from codecontext.store import (
    SCHEMA_VERSION,
    Store,
    get_all_migrations,
    inspect_local_git_repository,
    normalize_gitlab_base_url,
    now_us,
)
from codecontext.syntax import normalize_search_text

TEAM_FILE = "team.json"
SWAP_LOCK_TIMEOUT_SECONDS = 5.0


def _published_us(team: dict) -> int:
    # us_to_iso_z drops a zero fraction, so the strings do not sort lexically; compare instants.
    try:
        return datetime_to_us(parse_iso_utc(str(team["published_at"])))
    except ValueError as exc:
        raise CodeContextError(
            code="STORE_VERSION_UNSUPPORTED",
            message="Shared memory team.json has an invalid published_at timestamp.",
            retryable=False,
        ) from exc


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
    return project_remote_key(_origin_url(repo_path))


def _origin_url(repo_path: Path) -> str:
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
    return url


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
        # A leftover backup means an earlier swap stopped halfway; it may hold the only good copy.
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=(
                f"An earlier replace stopped halfway and left {backup}. Inspect it, "
                f"copy out anything you need, delete {backup}, then retry."
            ),
            retryable=False,
        )
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
        local_published = _published_us(local_team) if local_team else None
        share_published = _published_us(current)
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
        "published_at": us_to_iso_z(now_us()),
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


def _refuse_symlink(path: Path, message: str) -> None:
    if path.is_symlink():
        raise CodeContextError(code="INVALID_REQUEST", message=message, retryable=False)


def _invalid_share(message: str) -> CodeContextError:
    return CodeContextError(code="INVALID_REQUEST", message=message, retryable=False)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _read_share_manifest(share_dir: Path) -> dict[str, str]:
    """The manifest's rel path to SHA-256 map. Copies are checked against it, not the share."""
    manifest_path = share_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise _invalid_share("Shared memory is missing manifest.json.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _invalid_share("Shared memory manifest.json could not be read.") from exc
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in files.items()
    ):
        raise _invalid_share("Shared memory manifest.json has an unsupported shape.")
    return files


def _copy_verified(src: Path, dst: Path, expected: str | None, rel: str) -> None:
    """Copy src to dst and keep it only if the copy matches the manifest."""
    if expected is None:
        raise _invalid_share(f"Shared memory file {rel} is not listed in manifest.json.")
    shutil.copyfile(src, dst)
    # Hash the copy, not the share: the share can change while it is read.
    if _file_sha256(dst) != expected:
        dst.unlink(missing_ok=True)
        raise _invalid_share(f"Shared memory file {rel} does not match manifest.json.")


def _copy_regular_file(src: Path, dst: Path, files: dict[str, str], rel: str) -> None:
    _refuse_symlink(src, "Shared snapshot contains a symlink.")
    if not src.is_file():
        raise CodeContextError(
            code="STORE_NOT_INITIALIZED",
            message="Shared snapshot is missing a required file.",
            retryable=False,
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    _copy_verified(src, dst, files.get(rel), rel)


@functools.lru_cache(maxsize=1)
def _known_triggers() -> frozenset[tuple[str, str]]:
    conn = sqlite3.connect(":memory:")
    try:
        for _, sql, _ in get_all_migrations():
            conn.executescript(sql)
        return frozenset(conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"))
    finally:
        conn.close()


def _open_share_db(db_path: Path) -> sqlite3.Connection:
    """Read-only connection to a share database that passed the schema and trigger checks."""
    try:
        conn = sqlite3.connect(f"file:{urllib.parse.quote(str(db_path))}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise _invalid_share("Shared memory database could not be opened.") from exc
    try:
        try:
            conn.execute("PRAGMA trusted_schema=OFF")
            triggers = set(conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"))
        except sqlite3.Error as exc:
            raise _invalid_share("Shared memory database could not be read.") from exc
        if triggers != _known_triggers():
            raise _invalid_share(
                "Shared memory database triggers do not match the ones codecontext creates."
            )
        _require_supported_share_schema(conn)
    except BaseException:
        conn.close()
        raise
    return conn


_LOOSE_OBJECT = re.compile(r"[0-9a-f]{2}/[0-9a-f]{38}(?:[0-9a-f]{24})?")
_PACK_FILE = re.compile(r"pack/[^/]+\.pack")


def _object_file_is_intact(path: Path, rel: str, obj_fmt: str) -> bool:
    """Whether a local file under objects/ passes its own checksum. Other kinds count as intact.

    The manifest is written by the share, so it cannot say a local object is wrong.
    """
    if _LOOSE_OBJECT.fullmatch(rel):
        try:
            data = zlib.decompress(path.read_bytes())
        except zlib.error:
            return False
        return hashlib.new(obj_fmt, data).hexdigest() == rel.replace("/", "")
    if rel.startswith("pack/") and rel.endswith((".pack", ".idx")):
        trailer = hashlib.new(obj_fmt).digest_size
        digest = hashlib.new(obj_fmt)
        with path.open("rb") as f:
            remaining = os.fstat(f.fileno()).st_size - trailer
            if remaining < 0:
                return False
            while remaining:
                chunk = f.read(min(remaining, 1 << 20))
                if not chunk:
                    return False
                digest.update(chunk)
                remaining -= len(chunk)
            return digest.digest() == f.read(trailer)
    return True


def _import_pack(src: Path, dest_file: Path, expected: str, rel: str, git_dir: Path) -> None:
    """Copy a share pack and index it here, so object IDs come from the pack's content."""
    # A private subdirectory: git looks for packs only directly under objects/pack.
    incoming = Path(tempfile.mkdtemp(prefix=".incoming-", dir=dest_file.parent))
    try:
        pack = incoming / "share.pack"
        idx = incoming / "share.idx"
        _copy_verified(src, pack, expected, rel)
        proc = run_safe_git(["index-pack", "-o", str(idx), str(pack)], git_dir=git_dir, timeout=300.0, check=False)
        if proc.returncode != 0:
            raise _invalid_share(f"Shared memory file {rel} is not a valid git pack.")
        # Git uses a pack only once its index exists, so the index goes in last.
        os.replace(pack, dest_file)
        os.replace(idx, dest_file.with_suffix(".idx"))
    finally:
        shutil.rmtree(incoming, ignore_errors=True)


def _copy_regular_tree(
    src: Path, dst: Path, files: dict[str, str], rel_prefix: str, obj_fmt: str
) -> None:
    _refuse_symlink(src, "Shared snapshot contains a symlink.")
    in_objects = rel_prefix.endswith("/objects/")
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
        current = dst / Path(dirpath).relative_to(src)
        kept: list[str] = []
        for name in dirnames:
            child = Path(dirpath) / name
            _refuse_symlink(child, "Shared snapshot contains a symlink.")
            kept.append(name)
            (current / name).mkdir(parents=True, exist_ok=True)
        dirnames[:] = kept
        for name in filenames:
            child = Path(dirpath) / name
            _refuse_symlink(child, "Shared snapshot contains a symlink.")
            if not child.is_file():
                continue
            rel_in_tree = child.relative_to(src).as_posix()
            is_pack_file = in_objects and _PACK_FILE.fullmatch(rel_in_tree) is not None
            if in_objects and not is_pack_file and not _LOOSE_OBJECT.fullmatch(rel_in_tree):
                # Only objects whose IDs are checked here are taken. Everything else under objects/
                # is share-authored metadata git would trust: alternates, pack indexes, commit-graph.
                continue
            rel = rel_prefix + rel_in_tree
            expected = files.get(rel)
            if expected is None:
                raise _invalid_share(f"Shared memory file {rel} is not listed in manifest.json.")
            dest_file = current / name
            if (
                dest_file.is_file()
                and not dest_file.is_symlink()
                and _object_file_is_intact(dest_file, rel_in_tree, obj_fmt)
                and (not is_pack_file or dest_file.with_suffix(".idx").is_file())
            ):
                continue
            if is_pack_file:
                _import_pack(child, dest_file, expected, rel, dst.parent)
                continue
            # Readers never see a half-written object: write beside it, then swap it in.
            fd, tmp_name = tempfile.mkstemp(prefix=f".{name}.", dir=current)
            os.close(fd)
            try:
                _copy_verified(child, Path(tmp_name), expected, rel)
                if in_objects and not _object_file_is_intact(Path(tmp_name), rel_in_tree, obj_fmt):
                    raise _invalid_share(f"Shared memory object {rel} does not match its object ID.")
                os.replace(tmp_name, dest_file)
            except BaseException:
                Path(tmp_name).unlink(missing_ok=True)
                raise


def _import_share(share_dir: Path, store_path: Path, team: dict[str, object]) -> None:
    """Copy only the snapshot database, identity, and git objects. Runtime dirs are created empty."""
    files = _read_share_manifest(share_dir)
    store_path.mkdir(parents=True, exist_ok=True)
    _copy_regular_file(share_dir / "memory.sqlite3", store_path / "memory.sqlite3", files, "memory.sqlite3")
    share_db = _open_share_db(store_path / "memory.sqlite3")
    try:
        row = share_db.execute("SELECT object_format FROM store_meta WHERE singleton = 1").fetchone()
    finally:
        share_db.close()
    obj_fmt = "sha256" if row and row[0] == "sha256" else "sha1"
    _copy_regular_file(share_dir / "store.json", store_path / "store.json", files, "store.json")
    # team.json is not in the manifest; write the copy _read_team already validated.
    _write_team(store_path, team)
    overview = share_dir / "OVERVIEW.md"
    if overview.is_file() and not overview.is_symlink():
        _copy_regular_file(overview, store_path / "OVERVIEW.md", files, "OVERVIEW.md")

    src_git = share_dir / "repository.git"
    if src_git.exists():
        _refuse_symlink(src_git, "Shared snapshot contains a symlink.")
        dest_git = store_path / "repository.git"
        dest_git.mkdir(parents=True, exist_ok=True)
        # Packs are indexed with git as they arrive, so the git dir must be valid first.
        (dest_git / "config").write_text(sanitized_bare_git_config(obj_fmt), encoding="utf-8")
        for name in ("HEAD", "packed-refs"):
            src = src_git / name
            if src.exists():
                _copy_regular_file(src, dest_git / name, files, f"repository.git/{name}")
        for name in ("refs", "objects"):
            src = src_git / name
            if src.exists():
                _copy_regular_tree(src, dest_git / name, files, f"repository.git/{name}/", obj_fmt)
    (store_path / "tmp").mkdir(parents=True, exist_ok=True)


def _ensure_checkout_ignore(store_dir: Path, final_dest: Path, resolved_repo: Path) -> None:
    if final_dest.name != ".codecontext" or final_dest.parent != resolved_repo:
        return
    ignore = store_dir / ".gitignore"
    if ignore.is_symlink():
        ignore.unlink()
    if not ignore.exists():
        ignore.write_text("*\n", encoding="utf-8")


def _dir_is_empty(path: Path) -> bool:
    return next(path.iterdir(), None) is None


def _matching_store(dest: Path, repository_id: str) -> bool:
    store_json = dest / "store.json"
    db_file = dest / "memory.sqlite3"
    if not store_json.exists() and not db_file.exists():
        return False
    if store_json.is_symlink() or db_file.is_symlink() or not store_json.is_file() or not db_file.is_file():
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Refusing to replace a directory that is not a CodeContext store.",
            retryable=False,
        )
    try:
        raw = json.loads(store_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Refusing to replace a directory that is not a CodeContext store.",
            retryable=False,
        ) from exc
    if not isinstance(raw, dict) or raw.get("repository_id") != repository_id:
        raise CodeContextError(
            code="REPOSITORY_MISMATCH",
            message="Local store belongs to a different project memory.",
            retryable=False,
        )
    return True


def _db_file_stamps(store_path: Path) -> tuple[tuple[int, int] | None, ...]:
    """`(size, mtime_ns)` of the database and its WAL. Many writes leave `generation` alone, so
    join compares these too before replacing a store it read earlier."""
    stamps: list[tuple[int, int] | None] = []
    for name in ("memory.sqlite3", "memory.sqlite3-wal"):
        try:
            st = os.stat(store_path / name)
        except FileNotFoundError:
            stamps.append(None)
        else:
            stamps.append((st.st_size, st.st_mtime_ns))
    return tuple(stamps)


def _https_port(port: int | None) -> int | None:
    return None if port == 443 else port


def _bind_providers_to_origin(raw: dict, origin_url: str) -> None:
    """Tie share-authored provider fields to this checkout's origin and drop the issue allowlist.

    Sync calls these providers with the joiner's token, so the share must not pick the repository.
    """
    origin_key = project_remote_key(origin_url)
    origin_hostport, _, origin_path = origin_key.partition("/")
    origin = f"{origin_hostport.split(':')[0]}/{origin_path}".lower()
    github = raw.get("github_repository")
    if github:
        if f"github.com/{github}".lower() != origin:
            raise _invalid_share("Shared memory names a different GitHub repository than this checkout's origin.")
        raw["github_repository"] = origin_path
    gitlab = raw.get("gitlab_repository")
    if gitlab or raw.get("gitlab_base_url"):
        base = urllib.parse.urlparse(normalize_gitlab_base_url(raw.get("gitlab_base_url") or "https://gitlab.com"))
        host = (base.hostname or "").lower().rstrip(".")
        try:
            port = base.port
        except ValueError as exc:
            raise _invalid_share("Shared memory has a malformed GitLab base URL.") from exc
        port = _https_port(port)
        # An https origin pins the web port. An ssh origin's port is the ssh port, so any https port goes.
        port_ok = True
        if origin_url.strip().lower().startswith("https://"):
            origin_port = urllib.parse.urlparse(origin_url.strip()).port
            port_ok = port == _https_port(origin_port)
        if (
            not gitlab
            or f"{host}/{gitlab}".lower() != origin
            or not port_ok
            or base.path
            or base.username
            or base.query
            or base.fragment
        ):
            raise _invalid_share("Shared memory names a different GitLab project than this checkout's origin.")
        raw["gitlab_base_url"] = f"https://{host}" + (f":{port}" if port else "")
        raw["gitlab_repository"] = origin_path
    raw["allowed_issue_repositories"] = []


def _rebind(
    store_path: Path, repo_path: Path, *, floor_generation: int = 0, share_origin_url: str | None = None
) -> None:
    """Point the store at `repo_path` and move its generation past `floor_generation`.

    Graph and ancestor caches are keyed by path, revision and generation, so a database swapped
    in at a path must not reuse its predecessor's generation. Pass `share_origin_url` when
    store.json was just copied from a share.
    """
    resolved_repo, common_real, st_dev, st_ino, _obj_fmt = inspect_local_git_repository(repo_path)
    store_json_path = store_path / "store.json"
    raw = json.loads(store_json_path.read_text(encoding="utf-8"))
    if share_origin_url is not None:
        _bind_providers_to_origin(raw, share_origin_url)
    raw["read_only_archive"] = False
    raw["repo_path"] = str(resolved_repo)
    tmp_json = store_json_path.with_name(".store.json.tmp")
    tmp_json.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_json, store_json_path)
    store = Store.open(store_path, read_only=False)
    try:
        with store.write_transaction(bump_generation=True) as conn:
            conn.execute(
                """
                UPDATE store_meta
                SET local_repository_path = ?,
                    common_dir_realpath = ?,
                    common_dir_device = ?,
                    common_dir_inode = ?,
                    principal_id = NULL,
                    generation = MAX(generation, ?)
                WHERE singleton = 1
                """,
                (str(resolved_repo), str(common_real), st_dev, st_ino, floor_generation),
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
    origin_url = _origin_url(repo_path)
    remote_key = project_remote_key(origin_url)
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

    local_gen: int | None = None
    local_files: tuple[tuple[int, int] | None, ...] | None = None
    if dest.exists():
        if not _dir_is_empty(dest):
            if not _matching_store(dest, str(team["repository_id"])):
                raise CodeContextError(
                    code="INVALID_REQUEST",
                    message="Refusing to replace a nonempty directory that is not this project's store.",
                    retryable=False,
                )
            existing = Store.open(dest, read_only=True)
            try:
                local_gen = _store_generation(existing)
            finally:
                existing.close()
            local_files = _db_file_stamps(dest)
            local_published = None
            local_team = dest / TEAM_FILE
            if local_team.is_file() and not local_team.is_symlink():
                local_published = _published_us(_read_team(dest))
            share_published = _published_us(team)
            if local_published is not None and local_published > share_published:
                raise CodeContextError(
                    code="INVALID_REQUEST",
                    message="Shared memory is older than this store. Publish the newer memory.",
                    retryable=False,
                )
            if local_published == share_published and local_gen >= int(team["generation"]):
                _rebind(dest, repo_path)
                _ensure_checkout_ignore(dest, dest, resolved_repo)
                return

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}.joining-", dir=dest.parent))
    try:
        _import_share(share, staging, team)
        _rebind(staging, repo_path, floor_generation=local_gen or 0, share_origin_url=origin_url)
        _ensure_checkout_ignore(staging, dest, resolved_repo)
        if dest.exists():
            if _dir_is_empty(dest):
                dest.rmdir()
                os.rename(staging, dest)
            else:
                existing = Store.open(dest, read_only=True)
                try:
                    with existing.acquire_writer_lock(
                        timeout_seconds=remaining_timeout(SWAP_LOCK_TIMEOUT_SECONDS), allow_read_only=True
                    ):
                        if _store_generation(existing) != local_gen or _db_file_stamps(dest) != local_files:
                            raise CodeContextError(
                                code="STORE_BUSY",
                                message="This store changed while joining, so nothing was replaced. Run join again.",
                                retryable=True,
                            )
                        _replace_tree(staging, dest)
                finally:
                    existing.close()
        else:
            os.rename(staging, dest)
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging, ignore_errors=True)
        raise


MERGE_TABLE_ORDER: tuple[str, ...] = (
    "sources",
    "source_versions",
    "evidence",
    "model_runs",
    "commits",
    "changes",
    "hunks",
    "extractions",
    "symbols",
    "syntax_occurrences",
    "episodes",
    "claims",
    "summaries",
    "claim_evidence",
    "dependencies",
    "relations",
)

# Record kinds a link row can hang off, with the table and key that hold them.
_RECORD_TABLES: dict[str, tuple[str, str]] = {
    "sources": ("source", "id"),
    "evidence": ("evidence", "id"),
    "commits": ("commit", "oid"),
    "changes": ("change", "id"),
    "symbols": ("symbol", "id"),
    "episodes": ("episode", "id"),
    "claims": ("claim", "id"),
    "summaries": ("summary", "id"),
}

# Link table -> (owner kind, owner id, other end kind, other end id) columns. claim_evidence only
# links claims, so its kind columns are None.
_LINK_OWNERS: dict[str, tuple[str | None, str, str | None, str | None]] = {
    "claim_evidence": (None, "claim_id", None, None),
    "dependencies": ("dependent_kind", "dependent_id", "input_kind", "input_id"),
    "relations": ("from_kind", "from_id", "to_kind", "to_id"),
}

# Links decide whether these records are shown and what backs them, so a share may only link
# the ones it brought in this merge.
_DERIVED_KINDS = frozenset({"claim", "summary"})


def _new_link_rows(rows: list, cols: list[str], table: str, new_ids: dict[str, set[str]]) -> list:
    """Link rows a share may add: owned by a claim or summary new from the share, or for other
    owners (commits, symbols and the like, which the joiner usually has from git) with either end new."""
    kind_col, id_col, other_kind_col, other_id_col = _LINK_OWNERS[table]
    try:
        id_at = cols.index(f'"{id_col}"')
        kind_at = cols.index(f'"{kind_col}"') if kind_col else None
        other_id_at = cols.index(f'"{other_id_col}"') if other_id_col else None
        other_kind_at = cols.index(f'"{other_kind_col}"') if other_kind_col else None
    except ValueError:
        return []
    kept = []
    for r in rows:
        kind = "claim" if kind_at is None else r[kind_at]
        if r[id_at] in new_ids.get(kind, ()):
            kept.append(r)
        elif (
            kind not in _DERIVED_KINDS
            and other_kind_at is not None
            and other_id_at is not None
            and r[other_id_at] in new_ids.get(r[other_kind_at], ())
        ):
            kept.append(r)
    return kept


def _merge_columns(local: sqlite3.Connection, share: sqlite3.Connection, table: str) -> list[str]:
    """Columns present in both databases, double-quoted. Column names from the share are untrusted."""
    local_cols = {row[1] for row in local.execute(f'PRAGMA table_info("{table}")')}
    return [
        '"' + row[1].replace('"', '""') + '"'
        for row in share.execute(f'PRAGMA table_info("{table}")')
        if row[1] in local_cols
    ]


def _require_supported_share_schema(share: sqlite3.Connection) -> None:
    try:
        row = share.execute("SELECT schema_version FROM store_meta WHERE singleton = 1").fetchone()
    except sqlite3.Error as exc:
        raise CodeContextError(
            code="INVALID_REQUEST", message="Shared memory has no readable store_meta.", retryable=False
        ) from exc
    if row is None or not isinstance(row[0], int) or row[0] > SCHEMA_VERSION:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Shared memory was written by a newer version of codecontext. Upgrade before merging.",
            retryable=False,
        )


def _text_list(raw: object, field: str) -> list[str]:
    try:
        items = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise _invalid_share(f"Shared memory claim has malformed {field}.") from exc
    if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
        raise _invalid_share(f"Shared memory claim {field} must be a list of strings.")
    return items


def _index_merged_records(conn: sqlite3.Connection, share: sqlite3.Connection) -> None:
    """Add search rows for ready claims and summaries that came from the share."""
    claim_ids = [r[0] for r in share.execute("SELECT id FROM claims WHERE state = 'ready'")]
    for claim_id in claim_ids:
        row = conn.execute(
            "SELECT episode_id, statement, rationale, alternatives_json, constraints_json, support, lifecycle"
            " FROM claims WHERE id = ? AND state = 'ready'",
            (claim_id,),
        ).fetchone()
        if row is None:
            continue
        episode_id, statement, rationale, alts, constraints, support, lifecycle = row
        meta = canonical_json(
            {
                "display_title": statement[:200],
                "episode_id": episode_id,
                "lifecycle": lifecycle,
                "snippet": (rationale or statement)[:320],
                "support": support,
            }
        )
        conn.execute(
            "INSERT OR IGNORE INTO search_documents"
            " (record_kind, record_id, chunk_ordinal, title, identifiers, body, metadata_json)"
            " VALUES ('decision', ?, 0, ?, ?, ?, ?)",
            (
                claim_id,
                normalize_search_text(statement),
                normalize_search_text(" ".join(_text_list(constraints, "constraints_json"))),
                normalize_search_text(
                    f"{statement} {rationale or ''} {' '.join(_text_list(alts, 'alternatives_json'))}"
                ),
                meta,
            ),
        )
    summary_ids = [r[0] for r in share.execute("SELECT id FROM summaries WHERE state = 'ready'")]
    for summary_id in summary_ids:
        row = conn.execute(
            "SELECT scope_kind, scope_id, revision, text FROM summaries WHERE id = ? AND state = 'ready'",
            (summary_id,),
        ).fetchone()
        if row is None:
            continue
        scope_kind, scope_id, revision, text = row
        meta = canonical_json(
            {
                "display_title": f"Summary ({scope_kind}: {scope_id})",
                "revision": revision,
                "scope_id": scope_id,
                "scope_kind": scope_kind,
                "snippet": text[:320],
            }
        )
        conn.execute(
            "INSERT OR IGNORE INTO search_documents"
            " (record_kind, record_id, chunk_ordinal, title, identifiers, body, metadata_json)"
            " VALUES ('summary', ?, 0, ?, ?, ?, ?)",
            (
                summary_id,
                normalize_search_text(f"{scope_kind} {scope_id}"),
                normalize_search_text(scope_id),
                normalize_search_text(text),
                meta,
            ),
        )


def execute_team_merge(store_path: Path, share_dir: Path, repo_path: Path) -> TeamMergeReport:
    """Non-destructive record-level union merge of a shared team memory into an existing local store."""
    raw_share = share_dir.expanduser()
    if raw_share.is_symlink():
        raise CodeContextError(code="INVALID_REQUEST", message="Refusing to join through a symlinked share directory.", retryable=False)
    share = raw_share.resolve()
    team = _read_team(share)
    remote_key = detect_project_remote(repo_path)
    if team["project_remote"] != remote_key:
        raise CodeContextError(code="REPOSITORY_MISMATCH", message="This checkout's origin does not match the shared memory.", retryable=False)

    dest = store_path.expanduser().resolve()
    if not dest.exists() or not (dest / "memory.sqlite3").is_file():
        execute_join(store_path, share_dir, repo_path)
        store = Store.open(dest, read_only=True)
        try:
            return TeamMergeReport(
                repository_id=str(team["repository_id"]),
                generation=_store_generation(store),
                git_objects_fetched=True,
                sources_merged=0,
                source_versions_merged=0,
                claims_merged=0,
                summaries_merged=0,
            )
        finally:
            store.close()

    if not _matching_store(dest, str(team["repository_id"])):
        raise CodeContextError(code="REPOSITORY_MISMATCH", message="Local store belongs to a different project memory.", retryable=False)

    share_db_path = share / "memory.sqlite3"
    if not share_db_path.is_file() or share_db_path.is_symlink():
        raise CodeContextError(code="STORE_NOT_INITIALIZED", message="Shared memory is missing memory.sqlite3.", retryable=False)

    files = _read_share_manifest(share)
    (dest / "tmp").mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix=".share-merge-", dir=dest / "tmp"))
    try:
        db_copy = private / "memory.sqlite3"
        _copy_regular_file(share_db_path, db_copy, files, "memory.sqlite3")
        share_conn = _open_share_db(db_copy)
    except BaseException:
        shutil.rmtree(private, ignore_errors=True)
        raise
    try:
        local_store = Store.open(dest, read_only=False)
        try:
            with local_store.connect_read() as conn:
                row = conn.execute("SELECT object_format FROM store_meta WHERE singleton = 1").fetchone()
            obj_fmt = "sha256" if row and row[0] == "sha256" else "sha1"
            share_git_objs = share / "repository.git" / "objects"
            local_git_objs = dest / "repository.git" / "objects"
            git_objs_copied = False
            if share_git_objs.is_dir() and local_git_objs.is_dir():
                _copy_regular_tree(
                    share_git_objs, local_git_objs, files, "repository.git/objects/", obj_fmt
                )
                git_objs_copied = True

            try:
                with local_store.write_transaction(bump_generation=True) as conn:
                    # symbols reference their parent symbol; rows arrive in arbitrary order.
                    conn.execute("PRAGMA defer_foreign_keys = ON")
                    counts: dict[str, int] = {}
                    new_ids: dict[str, set[str]] = {}
                    for tbl in MERGE_TABLE_ORDER:
                        cols = _merge_columns(conn, share_conn, tbl)
                        if not cols:
                            continue
                        col_names = ", ".join(cols)
                        placeholders = ", ".join("?" for _ in cols)
                        c_before = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                        rows = share_conn.execute(f"SELECT {col_names} FROM {tbl}").fetchall()
                        if tbl in _LINK_OWNERS:
                            rows = _new_link_rows(rows, cols, tbl, new_ids)
                        max_rowid = conn.execute(f"SELECT COALESCE(MAX(rowid), 0) FROM {tbl}").fetchone()[0]
                        if rows:
                            conn.executemany(
                                f"INSERT OR IGNORE INTO {tbl} ({col_names}) VALUES ({placeholders})",
                                rows,
                            )
                        counts[tbl] = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0] - c_before
                        if tbl in _RECORD_TABLES:
                            kind, key = _RECORD_TABLES[tbl]
                            new_ids[kind] = {
                                r[0] for r in conn.execute(f"SELECT {key} FROM {tbl} WHERE rowid > ?", (max_rowid,))
                            }
                    _index_merged_records(conn, share_conn)
            except sqlite3.IntegrityError as exc:
                raise _invalid_share(
                    "Shared memory has records that point at rows it does not contain. Nothing was merged."
                ) from exc

            local_team_file = dest / TEAM_FILE
            if local_team_file.is_file():
                _write_team(dest, team)

            return TeamMergeReport(
                repository_id=str(team["repository_id"]),
                generation=_store_generation(local_store),
                git_objects_fetched=git_objs_copied,
                sources_merged=counts.get("sources", 0),
                source_versions_merged=counts.get("source_versions", 0),
                claims_merged=counts.get("claims", 0),
                summaries_merged=counts.get("summaries", 0),
            )
        finally:
            local_store.close()
    finally:
        share_conn.close()
        shutil.rmtree(private, ignore_errors=True)
