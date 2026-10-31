import base64
import fcntl
import functools
import json
import os
import shutil
import sqlite3
import subprocess
import threading
import time
import urllib.parse
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codecontext._deps import verify_runtime_dependencies
from codecontext.deadline import check_deadline, expired, remaining_timeout
from codecontext.errors import CodeContextError
from codecontext.identity import (
    canonical_json,
    datetime_to_us,
    default_token_estimator,
    make_gap_id,
    make_job_id,
    sha256_bytes,
    us_to_iso_z,
    validate_github_repo_name,
    validate_gitlab_project_path,
)
from codecontext.models import (
    GAP_CODE_REGISTRY,
    Budget,
    Capabilities,
    Coverage,
    CoveragePart,
    Expand,
    Gap,
    GapCount,
    InitRequest,
    OmittedCounts,
    QueueCounts,
    Status,
)

SCHEMA_VERSION = 2
MIN_READ_ONLY_SCHEMA_VERSION = 1
MIGRATIONS_DIR = Path(__file__).parent / "migrations"
MIGRATION_FILE = MIGRATIONS_DIR / "001_initial.sql"


def get_migration_sql_and_hash() -> tuple[str, str]:
    raw = MIGRATION_FILE.read_bytes()
    return raw.decode("utf-8"), sha256_bytes(raw)


@functools.lru_cache(maxsize=1)
def get_all_migrations() -> tuple[tuple[int, str, str], ...]:
    items: list[tuple[int, str, str]] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        prefix = path.name.split("_", 1)[0]
        if not prefix.isdigit():
            continue
        ver = int(prefix)
        raw = path.read_bytes()
        items.append((ver, raw.decode("utf-8"), sha256_bytes(raw)))
    items.sort(key=lambda t: t[0])
    return tuple(items)


def now_us() -> int:
    return datetime_to_us(datetime.now(UTC))


@dataclass(frozen=True)
class StoreConfig:
    format_version: int
    repository_id: str
    read_only_archive: bool
    repo_path: str | None
    github_repository: str | None
    allowed_issue_repositories: list[str]
    gitlab_repository: str | None = None
    gitlab_base_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "allowed_issue_repositories": self.allowed_issue_repositories,
            "format_version": self.format_version,
            "github_repository": self.github_repository,
            "read_only_archive": self.read_only_archive,
            "repo_path": self.repo_path,
            "repository_id": self.repository_id,
        }
        if self.gitlab_repository is not None:
            out["gitlab_repository"] = self.gitlab_repository
        if self.gitlab_base_url is not None:
            out["gitlab_base_url"] = self.gitlab_base_url
        return out


@dataclass(frozen=True)
class ReadSnapshot:
    conn: sqlite3.Connection
    repository_id: str
    revision: str | None
    generation: int
    access_epoch: int
    access_state: str
    object_format: str
    github_name: str | None
    gitlab_name: str | None = None
    gitlab_base_url: str | None = None
    workspace_base_revision: str | None = None


def _parse_github_remote_url(raw_url: str) -> str | None:
    url = raw_url.strip()
    if not url:
        return None
    for prefix in ("git@github.com:", "ssh://git@github.com/", "https://github.com/", "http://github.com/"):
        if url.startswith(prefix):
            rest = url[len(prefix) :].strip("/")
            if rest.endswith(".git"):
                rest = rest[:-4]
            try:
                return validate_github_repo_name(rest)
            except CodeContextError:
                return None
    return None


def detect_github_repository_from_git(repo_path: Path) -> str | None:
    """Detect `OWNER/REPO` from `remote.origin.url` when pointing to github.com."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
    }
    try:
        proc = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "config", "--get", "remote.origin.url"],
            cwd=repo_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return _parse_github_remote_url(proc.stdout.strip())
    except OSError:
        pass
    return None


def inspect_local_git_repository(repo_path: Path) -> tuple[Path, Path, int, int, str]:
    """Validate and inspect local Git repository, returning:
    (resolved_repo_path, common_dir_realpath, st_dev, st_ino, object_format).
    """
    resolved = repo_path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Repository path does not exist or is not a directory: {repo_path}",
            retryable=False,
        )
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "LC_ALL": "C",
    }
    try:
        proc = subprocess.run(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "rev-parse",
                "--git-common-dir",
                "--show-object-format",
            ],
            cwd=resolved,
            env=env,
            capture_output=True,
            text=True,
            timeout=remaining_timeout(10),
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        check_deadline()
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Path is not a valid Git repository: {resolved}",
            retryable=False,
        ) from exc

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Git rev-parse did not return common-dir and object-format.",
            retryable=False,
        )
    common_dir_raw = Path(lines[0])
    if not common_dir_raw.is_absolute():
        common_dir_raw = resolved / common_dir_raw
    common_dir_real = common_dir_raw.resolve()
    st = common_dir_real.stat()
    obj_fmt = lines[1].lower()
    if obj_fmt not in ("sha1", "sha256"):
        raise CodeContextError(
            code="STORE_VERSION_UNSUPPORTED",
            message=f"Unsupported Git object format: {obj_fmt!r}",
            retryable=False,
        )
    return resolved, common_dir_real, int(st.st_dev), int(st.st_ino), obj_fmt


def normalize_github_repo(repo: str | None) -> str | None:
    if repo is None:
        return None
    return validate_github_repo_name(repo.strip())


def normalize_gitlab_repo(repo: str | None) -> str | None:
    if repo is None:
        return None
    return validate_gitlab_project_path(repo.strip())


def normalize_gitlab_base_url(url: str | None) -> str | None:
    if url is None:
        return None
    cleaned = url.strip().rstrip("/")
    if not cleaned.startswith("https://"):
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="GitLab base URL must use https://.",
            retryable=False,
        )
    return cleaned


def normalize_allowlist(repos: list[str]) -> list[str]:
    cleaned = [validate_github_repo_name(r.strip()) for r in repos]
    by_lower: dict[str, str] = {}
    for item in cleaned:
        low = item.lower()
        if low in by_lower:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message=f"Duplicate repository in allowed_issue_repositories: {item!r}",
                retryable=False,
            )
        by_lower[low] = item
    return [by_lower[k] for k in sorted(by_lower.keys())]


def encode_cursor(
    *,
    query_fingerprint: str,
    generation: int,
    access_epoch: int,
    offset: int,
    extra: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "e": access_epoch,
        "g": generation,
        "o": offset,
        "q": query_fingerprint,
    }
    if extra:
        payload["x"] = extra
    raw = canonical_json(payload).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_and_validate_cursor(
    cursor: str,
    *,
    expected_query_fingerprint: str,
    current_generation: int,
    current_access_epoch: int,
) -> tuple[int, dict[str, Any]]:
    if not cursor or len(cursor) > 8192:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Invalid cursor length.",
            retryable=False,
        )
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Cursor payload is not a dict")
        q_fp = data["q"]
        gen = int(data["g"])
        epoch = int(data["e"])
        offset = int(data["o"])
        extra = data.get("x") or {}
    except Exception as exc:
        raise CodeContextError(
            code="CURSOR_STALE",
            message="Malformed or unreadable pagination cursor.",
            retryable=False,
        ) from exc

    if offset < 0 or not isinstance(extra, dict):
        raise CodeContextError(
            code="CURSOR_STALE",
            message="Invalid cursor offset or state.",
            retryable=False,
        )
    if epoch != current_access_epoch:
        raise CodeContextError(
            code="CURSOR_STALE",
            message="Pagination cursor invalidated by access epoch change.",
            retryable=False,
            details={"reason": "access_epoch_changed"},
        )
    if gen != current_generation:
        raise CodeContextError(
            code="CURSOR_STALE",
            message="Pagination cursor invalidated by store generation update.",
            retryable=False,
            details={"reason": "generation_changed"},
        )
    if q_fp != expected_query_fingerprint:
        raise CodeContextError(
            code="CURSOR_STALE",
            message="Pagination cursor does not match current query parameters.",
            retryable=False,
            details={"reason": "query_mismatch"},
        )
    return offset, extra


@contextmanager
def _acquire_file_lock(lock_path: Path, timeout_seconds: float = 10.0) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.monotonic() + timeout_seconds
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise CodeContextError(
                        code="STORE_BUSY",
                        message="Another process holds the store lock.",
                        retryable=True,
                    ) from exc
                time.sleep(0.02)
        yield
    finally:
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


class Store:
    def __init__(
        self,
        store_path: Path,
        config: StoreConfig,
        *,
        read_only: bool,
    ) -> None:
        self.store_path = store_path
        self.config = config
        self.read_only = read_only or config.read_only_archive
        self.db_path = store_path / "memory.sqlite3"
        self.git_dir = store_path / "repository.git"
        self.store_json_path = store_path / "store.json"
        self.tmp_dir = store_path / "tmp"
        self._lock_path = store_path / "tmp" / "writer.lock"
        self._closed = False
        self.schema_version = SCHEMA_VERSION
        self._tls = threading.local()
        self._close_callbacks: list[Any] = []

    def register_close_callback(self, fn: Any) -> None:
        self._close_callbacks.append(fn)

    def _ensure_open(self) -> None:
        if self._closed:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Store instance is closed.",
                retryable=False,
            )

    def close(self) -> None:
        if self._closed:
            return
        for cb in self._close_callbacks:
            try:
                cb()
            except Exception:
                pass
        self._close_callbacks.clear()
        self._closed = True

    @classmethod
    def initialize(cls, store_path: Path, request: InitRequest) -> "Store":
        verify_runtime_dependencies()
        resolved_store = store_path.expanduser().resolve()
        resolved_repo, common_real, st_dev, st_ino, obj_fmt = inspect_local_git_repository(
            request.repo_path
        )

        is_dot_codecontext = (
            resolved_store.name == ".codecontext" and resolved_store.parent == resolved_repo
        )
        if not is_dot_codecontext and (
            resolved_store == resolved_repo
            or resolved_repo in resolved_store.parents
            or resolved_store == common_real
            or common_real in resolved_store.parents
        ):
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Runtime store path must be outside the indexed repository checkout or at <repo>/.codecontext.",
                retryable=False,
            )

        norm_gh = normalize_github_repo(request.github_repository)
        norm_gl = normalize_gitlab_repo(request.gitlab_repository)
        norm_gl_base = normalize_gitlab_base_url(request.gitlab_base_url)
        norm_allow = normalize_allowlist(request.allowed_issue_repositories)

        store_json = resolved_store / "store.json"
        db_file = resolved_store / "memory.sqlite3"

        if store_json.exists() or db_file.exists():
            if not (store_json.exists() and db_file.exists()):
                raise CodeContextError(
                    code="STORE_NOT_INITIALIZED",
                    message="Store directory is partially initialized or corrupted.",
                    retryable=False,
                )
            existing = cls.open(resolved_store, read_only=False)
            try:
                if existing.config.read_only_archive:
                    raise CodeContextError(
                        code="REPOSITORY_MISMATCH",
                        message="Cannot initialize over a read-only archive store.",
                        retryable=False,
                    )
                with existing.connect_read() as conn:
                    row = conn.execute(
                        "SELECT common_dir_realpath, common_dir_device, common_dir_inode, "
                        "github_name, object_format FROM store_meta WHERE singleton = 1"
                    ).fetchone()
                    if not row:
                        raise CodeContextError(
                            code="STORE_NOT_INITIALIZED",
                            message="Missing store_meta row.",
                            retryable=False,
                        )
                    db_common_real, db_dev, db_ino, db_gh, db_fmt = row

                # Tolerate st_dev changes across OS remounts when realpath and st_ino match
                same_common = (
                    db_common_real == str(common_real)
                    and int(db_ino) == st_ino
                    and db_fmt == obj_fmt
                )
                if same_common and int(db_dev) != st_dev:
                    with existing.write_transaction(bump_generation=False) as wconn:
                        wconn.execute(
                            "UPDATE store_meta SET common_dir_device = ? WHERE singleton = 1",
                            (st_dev,),
                        )
                same_gh = (existing.config.github_repository or "").lower() == (norm_gh or "").lower()
                same_gl = (existing.config.gitlab_repository or "").lower() == (norm_gl or "").lower()
                same_allow = [x.lower() for x in existing.config.allowed_issue_repositories] == [
                    x.lower() for x in norm_allow
                ]
                if not (same_common and same_gh and same_gl and same_allow):
                    raise CodeContextError(
                        code="REPOSITORY_MISMATCH",
                        message="Store is already bound to a different repository or configuration.",
                        retryable=False,
                    )
                return existing
            except Exception:
                existing.close()
                raise

        created_dir = not resolved_store.exists()
        if resolved_store.exists():
            entries = [p for p in resolved_store.iterdir() if p.name != "tmp"]
            if entries:
                raise CodeContextError(
                    code="INVALID_REQUEST",
                    message="Target store directory is non-empty and not an initialized CodeContext store.",
                    retryable=False,
                )

        resolved_store.mkdir(parents=True, exist_ok=True)
        (resolved_store / "tmp").mkdir(parents=True, exist_ok=True)
        lock_file = resolved_store / "tmp" / "writer.lock"

        with _acquire_file_lock(lock_file, timeout_seconds=remaining_timeout(10.0)):
            # Re-check after lock acquisition in case a concurrent caller finished initialization
            if store_json.exists() and db_file.exists():
                return cls.open(resolved_store, read_only=False)

            try:
                git_dir = resolved_store / "repository.git"
                empty_tpl = resolved_store / "tmp" / "_empty_git_tpl"
                empty_tpl.mkdir(parents=True, exist_ok=True)

                init_args = ["git", "-c", "core.fsmonitor=false", "init", "--bare", f"--template={empty_tpl}"]
                if obj_fmt == "sha256":
                    init_args.append("--object-format=sha256")
                init_args.append(str(git_dir))
                env = {
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"),
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_GLOBAL": "/dev/null",
                    "GIT_TERMINAL_PROMPT": "0",
                    "LC_ALL": "C",
                }
                subprocess.run(init_args, env=env, capture_output=True, check=True, timeout=15)
                subprocess.run(
                    ["git", "--git-dir", str(git_dir), "config", "gc.auto", "0"],
                    env=env,
                    capture_output=True,
                    check=True,
                    timeout=10,
                )

                repo_uuid = str(uuid.uuid4())
                config = StoreConfig(
                    format_version=1,
                    repository_id=repo_uuid,
                    read_only_archive=False,
                    repo_path=str(resolved_repo),
                    github_repository=norm_gh,
                    allowed_issue_repositories=norm_allow,
                    gitlab_repository=norm_gl,
                    gitlab_base_url=norm_gl_base,
                )

                migrations = get_all_migrations()
                created_us = now_us()

                conn = sqlite3.connect(str(db_file), timeout=5.0, isolation_level=None)
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA foreign_keys=ON")
                    conn.execute("PRAGMA busy_timeout=5000")
                    for _, mig_sql, _ in migrations:
                        conn.executescript(mig_sql)
                    conn.execute("BEGIN IMMEDIATE")
                    for ver, _, mig_hash in migrations:
                        conn.execute(
                            "INSERT INTO schema_migrations (version, applied_at_us, migration_sha256) VALUES (?, ?, ?)",
                            (ver, created_us, mig_hash),
                        )
                    conn.execute(
                        """
                        INSERT INTO store_meta (
                            singleton, repository_id, schema_version, object_format,
                            local_repository_path, common_dir_realpath, common_dir_device, common_dir_inode,
                            github_repository_id, github_name, gitlab_project_id, gitlab_name, gitlab_base_url,
                            principal_id, active_revision, workspace_base_revision,
                            generation, access_epoch, access_state, created_at_us
                        ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?, NULL, NULL, NULL, 0, 0, 'allowed', ?)
                        """,
                        (
                            repo_uuid,
                            SCHEMA_VERSION,
                            obj_fmt,
                            str(resolved_repo),
                            str(common_real),
                            st_dev,
                            st_ino,
                            norm_gh,
                            norm_gl,
                            norm_gl_base,
                            created_us,
                        ),
                    )
                    conn.execute("COMMIT")
                finally:
                    conn.close()

                tmp_json = resolved_store / "tmp" / f"store.json.{uuid.uuid4().hex}"
                tmp_json.write_text(
                    json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(tmp_json, store_json)
                if is_dot_codecontext:
                    gitignore_file = resolved_store / ".gitignore"
                    if not gitignore_file.exists():
                        gitignore_file.write_text("*\n", encoding="utf-8")
            except Exception:
                # Clean up partial artifacts so retries are not blocked
                for fname in ("store.json", "memory.sqlite3", "memory.sqlite3-wal", "memory.sqlite3-shm", ".gitignore"):
                    (resolved_store / fname).unlink(missing_ok=True)
                shutil.rmtree(resolved_store / "repository.git", ignore_errors=True)
                if created_dir:
                    shutil.rmtree(resolved_store / "tmp", ignore_errors=True)
                    try:
                        resolved_store.rmdir()
                    except OSError:
                        pass
                raise

        return cls(resolved_store, config, read_only=False)

    @classmethod
    def open(cls, store_path: Path, *, read_only: bool = False) -> "Store":
        resolved_store = store_path.expanduser().resolve()
        store_json = resolved_store / "store.json"
        db_file = resolved_store / "memory.sqlite3"
        if not resolved_store.exists() or not store_json.exists() or not db_file.exists():
            raise CodeContextError(
                code="STORE_NOT_INITIALIZED",
                message=f"Store is not initialized at {resolved_store}",
                retryable=False,
            )

        try:
            raw_cfg = json.loads(store_json.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CodeContextError(
                code="STORE_NOT_INITIALIZED",
                message="Failed to parse store.json.",
                retryable=False,
            ) from exc

        required_keys = {
            "format_version",
            "repository_id",
            "read_only_archive",
            "repo_path",
            "github_repository",
            "allowed_issue_repositories",
        }
        optional_keys = {"gitlab_repository", "gitlab_base_url"}
        if (
            not isinstance(raw_cfg, dict)
            or not required_keys.issubset(raw_cfg.keys())
            or not set(raw_cfg.keys()).issubset(required_keys | optional_keys)
        ):
            raise CodeContextError(
                code="STORE_VERSION_UNSUPPORTED",
                message="Invalid store.json format or unexpected keys.",
                retryable=False,
            )
        if raw_cfg["format_version"] != 1:
            raise CodeContextError(
                code="STORE_VERSION_UNSUPPORTED",
                message=f"Unsupported store format_version: {raw_cfg['format_version']}",
                retryable=False,
            )

        config = StoreConfig(
            format_version=int(raw_cfg["format_version"]),
            repository_id=str(raw_cfg["repository_id"]),
            read_only_archive=bool(raw_cfg["read_only_archive"]),
            repo_path=raw_cfg["repo_path"],
            github_repository=raw_cfg["github_repository"],
            allowed_issue_repositories=list(raw_cfg["allowed_issue_repositories"]),
            gitlab_repository=raw_cfg.get("gitlab_repository"),
            gitlab_base_url=raw_cfg.get("gitlab_base_url"),
        )
        effective_read_only = read_only or config.read_only_archive
        verify_runtime_dependencies()

        instance = cls(resolved_store, config, read_only=effective_read_only)
        instance._validate_and_migrate_schema()
        if not effective_read_only:
            instance.recover_abandoned_jobs()
        return instance

    def _open_raw_connection(self, *, for_write: bool) -> sqlite3.Connection:
        self._ensure_open()
        check_deadline()
        if for_write and self.read_only:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Cannot perform write operation on a read-only store.",
                retryable=False,
            )
        try:
            if for_write:
                conn = sqlite3.connect(str(self.db_path), timeout=remaining_timeout(5.0), isolation_level=None)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("PRAGMA cache_size=-32000")
                conn.execute(f"PRAGMA busy_timeout={int(remaining_timeout(5.0) * 1000)}")
                conn.set_progress_handler(lambda: int(expired()), 1000)
                return conn
            else:
                quoted = urllib.parse.quote(str(self.db_path))
                uri = f"file:{quoted}?mode=ro"
                conn = sqlite3.connect(uri, uri=True, timeout=remaining_timeout(5.0), isolation_level=None)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("PRAGMA cache_size=-32000")
                conn.set_progress_handler(lambda: int(expired()), 1000)
                return conn
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                raise CodeContextError(
                    code="STORE_BUSY",
                    message="SQLite database is busy.",
                    retryable=True,
                ) from exc
            raise CodeContextError(
                code="IO_ERROR",
                message="Failed to open SQLite connection.",
                retryable=False,
            ) from exc

    def _validate_and_migrate_schema(self) -> None:
        all_migs = get_all_migrations()
        mig_by_ver = {ver: (sql, sha) for ver, sql, sha in all_migs}

        conn = self._open_raw_connection(for_write=False)
        try:
            try:
                mig_rows = conn.execute(
                    "SELECT version, migration_sha256 FROM schema_migrations ORDER BY version ASC"
                ).fetchall()
                meta = conn.execute(
                    "SELECT repository_id, schema_version, object_format FROM store_meta WHERE singleton = 1"
                ).fetchone()
            except sqlite3.OperationalError as exc:
                raise CodeContextError(
                    code="STORE_VERSION_UNSUPPORTED",
                    message="Database schema is missing required tables.",
                    retryable=False,
                ) from exc

            if not mig_rows or not meta:
                raise CodeContextError(
                    code="STORE_VERSION_UNSUPPORTED",
                    message="Database migration or store_meta is missing.",
                    retryable=False,
                )

            db_schema_ver = int(meta["schema_version"])
            if db_schema_ver < MIN_READ_ONLY_SCHEMA_VERSION or db_schema_ver > SCHEMA_VERSION:
                raise CodeContextError(
                    code="STORE_VERSION_UNSUPPORTED",
                    message=f"Unsupported store_meta schema_version: {db_schema_ver}",
                    retryable=False,
                )

            if len(mig_rows) != db_schema_ver:
                raise CodeContextError(
                    code="STORE_VERSION_UNSUPPORTED",
                    message="Inconsistent schema_migrations table length vs store_meta.schema_version.",
                    retryable=False,
                )

            for expected_ver, row in enumerate(mig_rows, start=1):
                row_ver = int(row["version"])
                if row_ver != expected_ver or row_ver not in mig_by_ver:
                    raise CodeContextError(
                        code="STORE_VERSION_UNSUPPORTED",
                        message=f"Unsupported schema migration version: {row_ver}",
                        retryable=False,
                    )
                if row["migration_sha256"] != mig_by_ver[row_ver][1]:
                    raise CodeContextError(
                        code="STORE_VERSION_UNSUPPORTED",
                        message=f"Schema migration checksum mismatch at version {row_ver}.",
                        retryable=False,
                    )

            if meta["repository_id"] != self.config.repository_id:
                raise CodeContextError(
                    code="REPOSITORY_MISMATCH",
                    message="store.json repository_id does not match SQLite store_meta.",
                    retryable=False,
                )
            self.schema_version = db_schema_ver
        finally:
            conn.close()

        if self.schema_version < SCHEMA_VERSION and not self.read_only:
            self._apply_pending_migrations(mig_by_ver)

    def _apply_pending_migrations(self, mig_by_ver: dict[int, tuple[str, str]]) -> None:
        with self.acquire_writer_lock(timeout_seconds=remaining_timeout(10.0)):
            conn = self._open_raw_connection(for_write=True)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                cur_row = conn.execute("SELECT schema_version FROM store_meta WHERE singleton = 1").fetchone()
                cur_ver = int(cur_row["schema_version"]) if cur_row else self.schema_version
                for next_ver in range(cur_ver + 1, SCHEMA_VERSION + 1):
                    mig_sql, mig_hash = mig_by_ver[next_ver]
                    applied_us = now_us()
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        for stmt in (s.strip() for s in mig_sql.split(";")):
                            if not stmt or stmt.upper().startswith("PRAGMA "):
                                continue
                            conn.execute(stmt)
                        conn.execute(
                            "INSERT INTO schema_migrations (version, applied_at_us, migration_sha256) VALUES (?, ?, ?)",
                            (next_ver, applied_us, mig_hash),
                        )
                        conn.execute(
                            "UPDATE store_meta SET schema_version = ? WHERE singleton = 1",
                            (next_ver,),
                        )
                        conn.execute("COMMIT")
                    except Exception:
                        try:
                            conn.execute("ROLLBACK")
                        except sqlite3.Error:
                            pass
                        raise
                self.schema_version = SCHEMA_VERSION
            finally:
                conn.close()

    @contextmanager
    def acquire_writer_lock(
        self,
        timeout_seconds: float = 5.0,
        *,
        allow_read_only: bool = False,
    ) -> Iterator[None]:
        self._ensure_open()
        if self.read_only and not allow_read_only:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Store is opened in read-only mode.",
                retryable=False,
            )
        if getattr(self._tls, "lock_depth", 0) > 0:
            self._tls.lock_depth += 1
            try:
                yield
            finally:
                self._tls.lock_depth -= 1
            return

        with _acquire_file_lock(self._lock_path, timeout_seconds=timeout_seconds):
            self._tls.lock_depth = 1
            try:
                yield
            finally:
                self._tls.lock_depth = 0

    @contextmanager
    def connect_read(self) -> Iterator[sqlite3.Connection]:
        conn = self._open_raw_connection(for_write=False)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def read_snapshot(self, *, allow_blocked_for_owner: bool = False) -> Iterator[ReadSnapshot]:
        conn = self._open_raw_connection(for_write=False)
        try:
            conn.execute("BEGIN DEFERRED")
            if self.schema_version >= 2:
                meta = conn.execute(
                    "SELECT repository_id, active_revision, generation, access_epoch, "
                    "access_state, object_format, github_name, gitlab_name, gitlab_base_url, "
                    "workspace_base_revision FROM store_meta WHERE singleton = 1"
                ).fetchone()
            else:
                meta = conn.execute(
                    "SELECT repository_id, active_revision, generation, access_epoch, "
                    "access_state, object_format, github_name, NULL AS gitlab_name, "
                    "NULL AS gitlab_base_url, NULL AS workspace_base_revision "
                    "FROM store_meta WHERE singleton = 1"
                ).fetchone()
            if not meta:
                raise CodeContextError(
                    code="STORE_NOT_INITIALIZED",
                    message="Missing store_meta row.",
                    retryable=False,
                )
            if meta["access_state"] == "blocked" and not allow_blocked_for_owner:
                raise CodeContextError(
                    code="ACCESS_BLOCKED",
                    message="Repository access has been revoked or blocked for this store.",
                    retryable=False,
                )
            yield ReadSnapshot(
                conn=conn,
                repository_id=str(meta["repository_id"]),
                revision=meta["active_revision"],
                generation=int(meta["generation"]),
                access_epoch=int(meta["access_epoch"]),
                access_state=str(meta["access_state"]),
                object_format=str(meta["object_format"]),
                github_name=meta["github_name"],
                gitlab_name=meta["gitlab_name"],
                gitlab_base_url=meta["gitlab_base_url"],
                workspace_base_revision=meta["workspace_base_revision"],
            )
        finally:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            conn.close()

    def verify_access_epoch_and_authorize(self, expected_epoch: int) -> None:
        """Immediately before response emission, check access_epoch and access_state through a fresh lightweight connection."""
        with self.connect_read() as conn:
            row = conn.execute(
                "SELECT access_epoch, access_state FROM store_meta WHERE singleton = 1"
            ).fetchone()
            if not row:
                raise CodeContextError(
                    code="STORE_NOT_INITIALIZED",
                    message="Store metadata missing during authorization check.",
                    retryable=False,
                )
            if row["access_state"] == "blocked":
                raise CodeContextError(
                    code="ACCESS_BLOCKED",
                    message="Store access was blocked before response authorization.",
                    retryable=False,
                )
            if int(row["access_epoch"]) != expected_epoch:
                raise CodeContextError(
                    code="CURSOR_STALE",
                    message="Access epoch changed before response authorization.",
                    retryable=True,
                    details={"reason": "access_epoch_changed"},
                )

    @contextmanager
    def write_transaction(
        self,
        *,
        bump_generation: bool = True,
        bump_access_epoch: bool = False,
        lock_timeout: float = 5.0,
    ) -> Iterator[sqlite3.Connection]:
        active_conn: sqlite3.Connection | None = getattr(self._tls, "write_conn", None)
        if active_conn is not None:
            depth = int(getattr(self._tls, "tx_depth", 1))
            sp_name = f"cc_sp_{depth}"
            self._tls.tx_depth = depth + 1
            active_conn.execute(f"SAVEPOINT {sp_name}")
            try:
                yield active_conn
                if bump_generation:
                    self._tls.pending_bump_gen = True
                if bump_access_epoch:
                    self._tls.pending_bump_epoch = True
                active_conn.execute(f"RELEASE SAVEPOINT {sp_name}")
            except Exception:
                try:
                    active_conn.execute(f"ROLLBACK TO SAVEPOINT {sp_name}")
                    active_conn.execute(f"RELEASE SAVEPOINT {sp_name}")
                except sqlite3.Error:
                    pass
                raise
            finally:
                self._tls.tx_depth = depth
            return

        with self.acquire_writer_lock(timeout_seconds=remaining_timeout(lock_timeout)):
            conn = self._open_raw_connection(for_write=True)
            try:
                try:
                    conn.execute("BEGIN IMMEDIATE")
                except sqlite3.OperationalError as exc:
                    if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                        raise CodeContextError(
                            code="STORE_BUSY",
                            message="Timed out acquiring SQLite write transaction.",
                            retryable=True,
                        ) from exc
                    raise
                self._tls.write_conn = conn
                self._tls.tx_depth = 1
                self._tls.pending_bump_gen = bump_generation
                self._tls.pending_bump_epoch = bump_access_epoch
                yield conn
                if self._tls.pending_bump_gen:
                    conn.execute("UPDATE store_meta SET generation = generation + 1 WHERE singleton = 1")
                if self._tls.pending_bump_epoch:
                    conn.execute("UPDATE store_meta SET access_epoch = access_epoch + 1 WHERE singleton = 1")
                conn.execute("COMMIT")
            except Exception as exc:
                conn.set_progress_handler(None, 0)
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                if isinstance(exc, sqlite3.OperationalError) and (
                    "locked" in str(exc).lower() or "busy" in str(exc).lower()
                ):
                    raise CodeContextError(
                        code="STORE_BUSY",
                        message="SQLite busy timeout expired during transaction.",
                        retryable=True,
                    ) from exc
                raise
            finally:
                self._tls.write_conn = None
                self._tls.tx_depth = 0
                conn.close()

    def recover_abandoned_jobs(self) -> int:
        """Reset abandoned running jobs to pending on writable startup."""
        with self.connect_read() as rconn:
            has_running = rconn.execute("SELECT 1 FROM jobs WHERE state = 'running' LIMIT 1").fetchone()
        if not has_running:
            return 0
        with self.write_transaction(bump_generation=False) as conn:
            cur = conn.execute(
                "UPDATE jobs SET state = 'pending', updated_at_us = ? WHERE state = 'running'",
                (now_us(),),
            )
            return cur.rowcount

    # --- Narrow record and invalidation helpers ---

    @staticmethod
    def invalidate_dependents_recursive(
        conn: sqlite3.Connection,
        initial_inputs: list[tuple[str, str]],
    ) -> set[tuple[str, str]]:
        """Iteratively traverse reverse dependencies and mark dependent claims/summaries stale,
        removing their ready search documents.
        Guards against derivation cycles.
        """
        queue: deque[tuple[str, str]] = deque(initial_inputs)
        visited_inputs: set[tuple[str, str]] = set(initial_inputs)
        invalidated_dependents: set[tuple[str, str]] = set()

        while queue:
            in_kind, in_id = queue.popleft()
            rows = conn.execute(
                "SELECT dependent_kind, dependent_id FROM dependencies WHERE input_kind = ? AND input_id = ?",
                (in_kind, in_id),
            ).fetchall()
            for row in rows:
                dep = (str(row["dependent_kind"]), str(row["dependent_id"]))
                if dep in visited_inputs:
                    raise CodeContextError(
                        code="IO_ERROR",
                        message=f"Derivation cycle detected involving {dep[0]}:{dep[1]}",
                        retryable=False,
                    )
                if dep not in invalidated_dependents:
                    invalidated_dependents.add(dep)
                    dep_kind, dep_id = dep
                    if dep_kind == "claim":
                        conn.execute(
                            "UPDATE claims SET state = 'stale' WHERE id = ? AND state = 'ready'",
                            (dep_id,),
                        )
                        conn.execute(
                            "DELETE FROM search_documents WHERE record_kind = 'decision' AND record_id = ?",
                            (dep_id,),
                        )
                        conn.execute(
                            "UPDATE relations SET state = 'stale' WHERE from_kind = 'claim' AND from_id = ? AND state = 'active'",
                            (dep_id,),
                        )
                    elif dep_kind == "summary":
                        conn.execute(
                            "UPDATE summaries SET state = 'stale' WHERE id = ? AND state = 'ready'",
                            (dep_id,),
                        )
                        conn.execute(
                            "DELETE FROM search_documents WHERE record_kind = 'summary' AND record_id = ?",
                            (dep_id,),
                        )
                    visited_inputs.add(dep)
                    queue.append(dep)
        return invalidated_dependents

    @staticmethod
    def upsert_gap(
        conn: sqlite3.Connection,
        *,
        repository_id: str,
        code: str,
        scope: str,
        message: str,
        retryable: bool,
        record_id: str | None = None,
        active: bool = True,
        observed_us: int | None = None,
    ) -> str:
        if code not in GAP_CODE_REGISTRY:
            raise ValueError(f"Unknown gap code: {code}")
        ts = observed_us if observed_us is not None else now_us()
        gap_id = make_gap_id(repository_id=repository_id, code=code, scope=scope, record_id=record_id)
        existing = conn.execute(
            "SELECT first_seen_at_us FROM coverage_gaps WHERE id = ?",
            (gap_id,),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE coverage_gaps
                SET message = ?, retryable = ?, active = ?, last_seen_at_us = ?
                WHERE id = ?
                """,
                (message[:1000], 1 if retryable else 0, 1 if active else 0, ts, gap_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO coverage_gaps (
                    id, code, scope, record_id, message, retryable, active, first_seen_at_us, last_seen_at_us
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    gap_id,
                    code,
                    scope[:1000],
                    record_id,
                    message[:1000],
                    1 if retryable else 0,
                    1 if active else 0,
                    ts,
                    ts,
                ),
            )
        return gap_id

    @staticmethod
    def resolve_gaps_in_scope(conn: sqlite3.Connection, *, scope: str, code: str | None = None) -> None:
        ts = now_us()
        if code is not None:
            conn.execute(
                "UPDATE coverage_gaps SET active = 0, last_seen_at_us = ? WHERE scope = ? AND code = ? AND active = 1",
                (ts, scope, code),
            )
        else:
            conn.execute(
                "UPDATE coverage_gaps SET active = 0, last_seen_at_us = ? WHERE scope = ? AND active = 1",
                (ts, scope),
            )

    @staticmethod
    def enqueue_job(
        conn: sqlite3.Connection,
        *,
        kind: str,
        owner_id: str,
        priority: int,
        cursor: dict[str, Any] | None = None,
    ) -> str:
        job_id = make_job_id(kind=kind, owner_id=owner_id)
        ts = now_us()
        existing = conn.execute(
            "SELECT id, state, priority FROM jobs WHERE kind = ? AND owner_id = ?",
            (kind, owner_id),
        ).fetchone()
        if existing:
            new_prio = min(int(existing["priority"]), priority)
            new_state = "pending" if existing["state"] in ("complete", "retryable_error") else existing["state"]
            conn.execute(
                "UPDATE jobs SET priority = ?, state = ?, updated_at_us = ? WHERE id = ?",
                (new_prio, new_state, ts, job_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO jobs (id, kind, owner_id, priority, state, cursor_json, attempts, last_error_code, created_at_us, updated_at_us)
                VALUES (?, ?, ?, ?, 'pending', ?, 0, NULL, ?, ?)
                """,
                (job_id, kind, owner_id, priority, canonical_json(cursor or {}), ts, ts),
            )
        return job_id

    @staticmethod
    def get_queue_counts(conn: sqlite3.Connection) -> QueueCounts:
        rows = conn.execute("SELECT state, COUNT(*) AS cnt FROM jobs GROUP BY state").fetchall()
        counts = {row["state"]: int(row["cnt"]) for row in rows}
        return QueueCounts(
            pending=counts.get("pending", 0),
            running=counts.get("running", 0),
            complete=counts.get("complete", 0),
            retryable_error=counts.get("retryable_error", 0),
            blocked=counts.get("blocked", 0),
        )

    @staticmethod
    def build_coverage_and_gaps(
        conn: sqlite3.Connection,
        *,
        max_detailed_gaps: int = 20,
    ) -> tuple[Coverage, list[Gap], int]:
        """Compute Coverage and bounded detailed active Gap list.
        Returns (coverage, detailed_gaps, omitted_gap_count).
        """
        cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(store_meta)").fetchall()}
        if "gitlab_name" in cols:
            meta = conn.execute(
                "SELECT active_revision, github_name, gitlab_name FROM store_meta WHERE singleton = 1"
            ).fetchone()
            gitlab_name = meta["gitlab_name"] if meta else None
        else:
            meta = conn.execute(
                "SELECT active_revision, github_name FROM store_meta WHERE singleton = 1"
            ).fetchone()
            gitlab_name = None
        active_rev = meta["active_revision"] if meta else None
        github_name = meta["github_name"] if meta else None

        gap_rows = conn.execute(
            """
            SELECT id, code, scope, record_id, message, retryable
            FROM coverage_gaps
            WHERE active = 1
            ORDER BY scope ASC, code ASC, id ASC
            """
        ).fetchall()
        total_active_gaps = len(gap_rows)
        first_gap_id = str(gap_rows[0]["id"]) if gap_rows else None

        code_counts: dict[str, int] = {}
        git_gaps = 0
        github_gaps = 0
        structure_gaps = 0
        enrichment_gaps = 0
        for r in gap_rows:
            c = str(r["code"])
            sc = str(r["scope"])
            code_counts[c] = code_counts.get(c, 0) + 1
            if sc.startswith("git:") or c in ("GIT_HISTORY_PENDING", "SHALLOW_HISTORY", "MISSING_GIT_OBJECT", "MISSING_PARENT"):
                git_gaps += 1
            elif sc.startswith(("github:", "gitlab:")) or c in ("GITHUB_BACKFILL_PENDING", "SOURCE_UNAVAILABLE", "SOURCE_UNVERIFIED", "API_RESULT_CAP", "RECONCILIATION_PENDING"):
                github_gaps += 1
            elif sc.startswith("blob:") or c in ("STRUCTURE_PENDING", "UNSUPPORTED_SYNTAX", "PARSE_ERROR", "BLOB_TOO_LARGE", "BINARY_CONTENT", "TEXT_DECODE_FAILED"):
                structure_gaps += 1
            elif sc.startswith("episode:") or sc.startswith("summary:") or c in ("ENRICHMENT_PENDING", "EPISODE_INSPECTION_PENDING", "DERIVED_ENRICHMENT_PENDING", "MODEL_FAILED", "MODEL_INVALID_OUTPUT", "BUDGET_EXHAUSTED"):
                enrichment_gaps += 1

        gap_count_list = [
            GapCount(code=k, count=v) for k, v in sorted(code_counts.items())
        ][:100]

        detailed_gaps = [
            Gap(
                id=str(r["id"]),
                code=str(r["code"]),
                scope=str(r["scope"]),
                message=str(r["message"]),
                retryable=bool(r["retryable"]),
                record_id=r["record_id"],
            )
            for r in gap_rows[:max_detailed_gaps]
        ]
        omitted_gaps = max(0, total_active_gaps - len(detailed_gaps))

        ref_rows = conn.execute(
            "SELECT name FROM refs WHERE active = 1 ORDER BY name ASC"
        ).fetchall()
        all_refs = [str(r["name"]) for r in ref_rows]
        selected_refs = all_refs[:20]
        omitted_ref_count = max(0, len(all_refs) - len(selected_refs))

        total_commits = int(conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0])
        completed_commits = int(
            conn.execute("SELECT COUNT(*) FROM commits WHERE diff_state = 'complete'").fetchone()[0]
        )
        pending_git_jobs = int(
            conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE kind LIKE 'git_%' AND state IN ('pending', 'running', 'retryable_error')"
            ).fetchone()[0]
        )
        git_ckpt = conn.execute(
            "SELECT MAX(last_success_at_us) AS ls, MAX(last_reconciled_at_us) AS lr FROM checkpoints WHERE stream_key LIKE 'git:%'"
        ).fetchone()
        git_last_success = us_to_iso_z(int(git_ckpt["ls"])) if git_ckpt and git_ckpt["ls"] is not None else None
        git_reconciled = us_to_iso_z(int(git_ckpt["lr"])) if git_ckpt and git_ckpt["lr"] is not None else None

        if active_rev is None and total_commits == 0:
            git_state: Any = "not_started"
            git_known_total: int | None = None
        elif pending_git_jobs > 0 or completed_commits < total_commits or git_gaps > 0:
            git_state = "partial"
            git_known_total = total_commits if pending_git_jobs == 0 else None
        else:
            git_state = "complete"
            git_known_total = total_commits

        git_part = CoveragePart(
            state=git_state,
            completed=completed_commits,
            known_total=git_known_total,
            unit="commits",
            last_success_at=git_last_success,
            reconciled_at=git_reconciled,
            gap_count=git_gaps,
        )

        # Remote discussion coverage (GitHub or GitLab; excludes localprovider='git' docs)
        remote_sources = int(
            conn.execute("SELECT COUNT(*) FROM sources WHERE provider IN ('github', 'gitlab')").fetchone()[0]
        )
        pending_gh_jobs = int(
            conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE (kind LIKE 'github_%' OR kind LIKE 'gitlab_%') "
                "AND state IN ('pending', 'running', 'retryable_error')"
            ).fetchone()[0]
        )
        blocked_gh_jobs = int(
            conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE (kind LIKE 'github_%' OR kind LIKE 'gitlab_%') AND state = 'blocked'"
            ).fetchone()[0]
        )
        gh_ckpt = conn.execute(
            "SELECT MAX(last_success_at_us) AS ls, MIN(last_reconciled_at_us) AS lr "
            "FROM checkpoints WHERE stream_key LIKE 'github:%' OR stream_key LIKE 'gitlab:%'"
        ).fetchone()
        gh_last_success = us_to_iso_z(int(gh_ckpt["ls"])) if gh_ckpt and gh_ckpt["ls"] is not None else None
        gh_reconciled = us_to_iso_z(int(gh_ckpt["lr"])) if gh_ckpt and gh_ckpt["lr"] is not None else None

        has_remote_provider = bool(github_name or gitlab_name or remote_sources > 0 or gh_last_success is not None)
        if not has_remote_provider:
            gh_state: Any = "unavailable"
            gh_known_total: int | None = 0
        elif blocked_gh_jobs > 0 and remote_sources == 0:
            gh_state = "blocked"
            gh_known_total = None
        elif remote_sources == 0 and pending_gh_jobs == 0 and gh_last_success is None:
            gh_state = "not_started"
            gh_known_total = None
        elif pending_gh_jobs > 0 or blocked_gh_jobs > 0 or github_gaps > 0:
            gh_state = "partial"
            gh_known_total = None
        else:
            gh_state = "complete"
            gh_known_total = remote_sources

        gh_part = CoveragePart(
            state=gh_state,
            completed=remote_sources,
            known_total=gh_known_total,
            unit="records",
            last_success_at=gh_last_success,
            reconciled_at=gh_reconciled,
            gap_count=github_gaps,
        )

        snap_total = int(conn.execute("SELECT COUNT(*) FROM snapshot_files").fetchone()[0])
        snap_done = int(
            conn.execute(
                "SELECT COUNT(*) FROM snapshot_files WHERE content_state IN ('indexed', 'text_only', 'metadata_only')"
            ).fetchone()[0]
        )
        excluded_files = int(
            conn.execute(
                "SELECT COUNT(*) FROM snapshot_files WHERE content_state = 'metadata_only'"
            ).fetchone()[0]
        )
        fps = [
            str(r["extractor_fingerprint"])
            for r in conn.execute(
                "SELECT DISTINCT e.extractor_fingerprint FROM extractions e "
                "JOIN snapshot_files sf ON sf.extraction_id = e.id "
                "ORDER BY e.extractor_fingerprint ASC LIMIT 10"
            ).fetchall()
        ]
        if active_rev is None:
            struct_state: Any = "not_started"
            struct_known: int | None = None
        elif snap_done < snap_total or structure_gaps > 0:
            struct_state = "partial"
            struct_known = snap_total
        else:
            struct_state = "complete"
            struct_known = snap_total

        struct_part = CoveragePart(
            state=struct_state,
            completed=snap_done,
            known_total=struct_known,
            unit="blobs",
            last_success_at=git_last_success,
            reconciled_at=None,
            gap_count=structure_gaps,
        )

        total_eps = int(conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])
        inspected_eps = int(
            conn.execute("SELECT COUNT(*) FROM episodes WHERE state = 'inspected'").fetchone()[0]
        )
        run_ckpt = conn.execute(
            "SELECT MAX(finished_at_us) AS fin FROM model_runs WHERE state = 'complete'"
        ).fetchone()
        enrich_last = us_to_iso_z(int(run_ckpt["fin"])) if run_ckpt and run_ckpt["fin"] is not None else None
        if total_eps == 0 or (inspected_eps == 0 and enrich_last is None):
            enrich_state: Any = "not_started"
            enrich_known: int | None = total_eps if total_eps > 0 else None
        elif inspected_eps < total_eps or enrichment_gaps > 0:
            enrich_state = "partial"
            enrich_known = total_eps
        else:
            enrich_state = "complete"
            enrich_known = total_eps

        enrich_part = CoveragePart(
            state=enrich_state,
            completed=inspected_eps,
            known_total=enrich_known,
            unit="segments",
            last_success_at=enrich_last,
            reconciled_at=None,
            gap_count=enrichment_gaps,
        )

        coverage = Coverage(
            git=git_part,
            github=gh_part,
            structure=struct_part,
            enrichment=enrich_part,
            selected_refs=selected_refs,
            selected_ref_count=len(all_refs),
            omitted_ref_count=omitted_ref_count,
            excluded_file_count=excluded_files,
            parser_fingerprints=fps,
            active_gap_count=total_active_gaps,
            gap_counts=gap_count_list,
            first_gap_id=first_gap_id,
        )
        return coverage, detailed_gaps, omitted_gaps

    def status(
        self,
        *,
        token_counter: Any = default_token_estimator,
        estimator_id: str = "utf8_bytes_div3_v1",
        count_kind: str = "estimated",
    ) -> Status:
        """Compute Status with fixed 6,000-token ceiling, at most 10 detailed gaps, and gap_ expansion handles."""
        try:
            from codecontext.syntax import get_supported_syntax_languages

            syntax_langs = get_supported_syntax_languages()
        except Exception:
            syntax_langs = [
                "bash",
                "c",
                "c_sharp",
                "cpp",
                "go",
                "gradle",
                "graphql",
                "java",
                "javascript",
                "kotlin",
                "php",
                "proto",
                "python",
                "ruby",
                "rust",
                "sql",
                "swift",
                "tsx",
                "typescript",
            ]

        with self.read_snapshot() as snap:
            conn = snap.conn
            q_counts = self.get_queue_counts(conn)
            snap_file_cnt = int(conn.execute("SELECT COUNT(*) FROM snapshot_files").fetchone()[0])
            commit_cnt = int(conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0])
            source_cnt = int(conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0])
            gh_source_cnt = int(
                conn.execute("SELECT COUNT(*) FROM sources WHERE provider = 'github'").fetchone()[0]
            )
            gl_source_cnt = int(
                conn.execute("SELECT COUNT(*) FROM sources WHERE provider = 'gitlab'").fetchone()[0]
            )
            claim_cnt = int(
                conn.execute("SELECT COUNT(*) FROM claims WHERE state = 'ready'").fetchone()[0]
            )
            sum_cnt = int(
                conn.execute("SELECT COUNT(*) FROM summaries WHERE state = 'ready'").fetchone()[0]
            )
            emb_cnt = 0
            if self.schema_version >= 2:
                try:
                    emb_cnt = int(conn.execute("SELECT COUNT(*) FROM document_embeddings").fetchone()[0])
                except sqlite3.OperationalError:
                    emb_cnt = 0

            has_git_objs = (self.git_dir / "objects").exists()
            caps = Capabilities(
                syntax_languages=syntax_langs,
                text_fallback=True,
                git_history=commit_cnt > 0,
                github_history=gh_source_cnt > 0,
                gitlab_history=gl_source_cnt > 0,
                offline_git_objects=has_git_objs,
                enriched_claims_available=claim_cnt > 0,
                semantic_embeddings_available=emb_cnt > 0,
            )

            max_gaps = 10
            while max_gaps >= 0:
                cov, gaps, omitted_gaps = self.build_coverage_and_gaps(
                    conn, max_detailed_gaps=max_gaps
                )
                expand_list: list[Expand] = []
                if cov.first_gap_id is not None:
                    expand_list.append(
                        Expand(
                            id=cov.first_gap_id,
                            reason="Inspect active coverage gaps starting from first_gap_id",
                        )
                    )
                truncated = omitted_gaps > 0
                used_est = 0
                for _ in range(3):
                    candidate = Status(
                        repository_id=snap.repository_id,
                        revision=snap.revision,
                        generation=snap.generation,
                        access_epoch=snap.access_epoch,
                        coverage=cov,
                        gaps=gaps,
                        capabilities=caps,
                        access_state="allowed" if snap.access_state == "allowed" else "blocked",
                        read_only=self.read_only,
                        github_repository=snap.github_name,
                        gitlab_repository=snap.gitlab_name,
                        object_format="sha256" if snap.object_format == "sha256" else "sha1",
                        jobs=q_counts,
                        snapshot_file_count=snap_file_cnt,
                        commit_count=commit_cnt,
                        source_count=source_cnt,
                        claim_count=claim_cnt,
                        summary_count=sum_cnt,
                        budget=Budget(
                            requested_tokens=6000,
                            used_tokens=used_est,
                            estimator=estimator_id,
                            count_kind="tokenizer" if count_kind == "tokenizer" else "estimated",
                            truncated=truncated,
                            omitted_counts=OmittedCounts(gaps=omitted_gaps),
                        ),
                        expand=expand_list,
                    )
                    serialized = candidate.model_dump_json()
                    new_used = int(token_counter(serialized))
                    if new_used == used_est:
                        break
                    used_est = new_used

                if used_est <= 6000 or max_gaps == 0:
                    break
                max_gaps -= 1

            expected_epoch = snap.access_epoch

        self.verify_access_epoch_and_authorize(expected_epoch)
        return candidate
