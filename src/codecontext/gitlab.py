import json
import os
import shutil
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any

import httpx

from codecontext.deadline import check_deadline, remaining_timeout
from codecontext.errors import CodeContextError
from codecontext.github import (
    MAX_HTTP_BODY_BYTES,
    USER_AGENT,
    _parse_opt_time_us,
    upsert_github_source_and_version,
)
from codecontext.identity import (
    canonical_json,
    canonical_sha256,
    make_episode_id,
    make_relation_id,
)
from codecontext.store import Store, now_us


def detect_gitlab_repository_from_git(repo_path: Path) -> tuple[str, str] | None:
    """Detect `(gitlab_base_url, project_path)` from `remote.origin.url` when pointing to a GitLab host."""
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
            timeout=remaining_timeout(5),
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return parse_gitlab_remote_url(proc.stdout.strip())
    except OSError:
        pass
    return None


def _host_and_port(parsed: urllib.parse.ParseResult) -> tuple[str, int | None] | None:
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host or parsed.username or parsed.password:
        return None
    return host, parsed.port


def _is_gitlab_host(host: str) -> bool:
    # ponytail: a DNS label equal to "gitlab" treats self-hosted hosts as GitLab. Upgrade: an explicit host allowlist.
    return host == "gitlab.com" or host.endswith(".gitlab.com") or "gitlab" in host.split(".")


def _https_origin(host: str, port: int | None) -> str:
    if port not in (None, 443):
        return f"https://{host}:{port}"
    return f"https://{host}"


def _gitlab_origin(url: str) -> tuple[str, int | None] | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        return None
    ident = _host_and_port(parsed)
    if ident is None or not _is_gitlab_host(ident[0]):
        return None
    return ident


def _norm_port(port: int | None) -> int:
    return 443 if port in (None, 443) else port


def _same_endpoint(left: tuple[str, int | None], right: tuple[str, int | None]) -> bool:
    return left[0] == right[0] and _norm_port(left[1]) == _norm_port(right[1])


def _env_token_allowed(host: str, port: int | None) -> bool:
    if (host == "gitlab.com" or host.endswith(".gitlab.com")) and _norm_port(port) == 443:
        return True
    raw = os.environ.get("GITLAB_HOST", "").strip()
    if not raw:
        return False
    configured = _gitlab_origin(raw if "://" in raw else f"https://{raw}")
    return configured is not None and _same_endpoint(configured, (host, port))


def parse_gitlab_remote_url(raw_url: str) -> tuple[str, str] | None:
    """Parse a GitLab SSH or HTTPS remote URL into `(https_base_url, namespace/project)`."""
    url = raw_url.strip()
    if not url or "github.com" in url.lower():
        return None
    if url.startswith("git@"):
        host, sep, proj = url[4:].partition(":")
        host = host.lower().rstrip(".")
        if not sep or "@" in host or not _is_gitlab_host(host):
            return None
        proj = proj.strip("/")
        if proj.endswith(".git"):
            proj = proj[:-4]
        if "/" in proj and ".." not in proj.split("/"):
            return (_https_origin(host, None), proj)
        return None

    ident = _gitlab_origin(url)
    if ident is None:
        return None
    proj = urllib.parse.urlparse(url).path.strip("/")
    if proj.endswith(".git"):
        proj = proj[:-4]
    if "/" in proj and ".." not in proj.split("/"):
        return (_https_origin(*ident), proj)
    return None


def resolve_gitlab_token(base_url: str = "https://gitlab.com") -> str | None:
    """Resolve a GitLab token for this exact host.

    `GITLAB_TOKEN` and `GLAB_TOKEN` are used only for gitlab.com, its subdomains,
    or the host named by `GITLAB_HOST`. Other hosts use `glab --host` or an MCP
    config whose URL host matches exactly.
    """
    ident = _gitlab_origin(base_url)
    if ident is None:
        return None
    host, port = ident
    if _env_token_allowed(host, port):
        env_tok = os.environ.get("GITLAB_TOKEN") or os.environ.get("GLAB_TOKEN")
        if env_tok:
            return env_tok.strip()

    glab_host = host if port in (None, 443) else f"{host}:{port}"
    glab_bin = shutil.which("glab") or "/opt/homebrew/bin/glab"
    if os.path.isfile(glab_bin) and os.access(glab_bin, os.X_OK):
        try:
            proc = subprocess.run(
                [glab_bin, "config", "get", "token", "--host", glab_host],
                capture_output=True,
                text=True,
                timeout=remaining_timeout(3),
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except OSError:
            pass

    for cfg_path in (Path.home() / ".cursor" / "mcp.json", Path.home() / ".claude.json"):
        try:
            if not cfg_path.is_file():
                continue
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            servers = data.get("mcpServers") if isinstance(data, dict) else None
            if not isinstance(servers, dict):
                continue
            for srv_name, srv_cfg in servers.items():
                if "gitlab" not in str(srv_name).lower() or not isinstance(srv_cfg, dict):
                    continue
                # Verify MCP server targets the same GitLab host (defaults to gitlab.com)
                srv_url = str(
                    srv_cfg.get("url")
                    or (srv_cfg.get("env") or {}).get("GITLAB_API_URL")
                    or "https://gitlab.com"
                )
                srv_ident = _gitlab_origin(srv_url if "://" in srv_url else f"https://{srv_url}")
                if srv_ident is None or not _same_endpoint(srv_ident, (host, port)):
                    continue
                env_map = srv_cfg.get("env")
                if isinstance(env_map, dict):
                    for k in ("GITLAB_PERSONAL_ACCESS_TOKEN", "GITLAB_TOKEN", "GLAB_TOKEN"):
                        if env_map.get(k):
                            return str(env_map[k]).strip()
                headers = srv_cfg.get("headers")
                if isinstance(headers, dict):
                    pt = headers.get("PRIVATE-TOKEN") or headers.get("private-token")
                    if pt:
                        return str(pt).strip()
                    auth_h = str(headers.get("Authorization") or headers.get("authorization") or "").strip()
                    if auth_h.lower().startswith("bearer "):
                        return auth_h[7:].strip()
        except Exception:
            pass

    return None


class GitLabClient:
    """Synchronous GET client for GitLab REST API v4 (`/api/v4/...`) with streaming 32 MiB limit."""

    def __init__(
        self,
        *,
        base_url: str = "https://gitlab.com",
        token: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        ident = _gitlab_origin(base_url)
        if ident is None:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="GitLab base_url must be https on a GitLab host.",
                retryable=False,
            )
        self.base_url = _https_origin(*ident)
        self.token = token if token is not None else resolve_gitlab_token(self.base_url)
        headers = {
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
        if self.token:
            headers["PRIVATE-TOKEN"] = self.token
        self._client = httpx.Client(
            base_url=f"{self.base_url}/api/v4",
            headers=headers,
            transport=transport,
            follow_redirects=False,
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
        )
        self.requests_attempted = 0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GitLabClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def get_json(self, path: str, *, remaining_requests: int = 1000) -> tuple[int, Any, str | None]:
        if remaining_requests <= 0:
            raise CodeContextError(
                code="BUDGET_EXHAUSTED",
                message="Sync request budget exhausted before GitLab HTTP call.",
                retryable=True,
            )
        self.requests_attempted += 1
        target = path if path.startswith("/") else f"/{path}"
        with self._client.stream("GET", target, timeout=remaining_timeout(30)) as resp:
            status = resp.status_code
            if status == 401:
                raise CodeContextError(
                    code="AUTH_REQUIRED",
                    message="GitLab authentication failed (401 Unauthorized).",
                    retryable=False,
                )
            if status == 429:
                raise CodeContextError(
                    code="RATE_LIMITED",
                    message="GitLab API rate limit exceeded (429).",
                    retryable=True,
                )
            if status == 403:
                raise CodeContextError(
                    code="ACCESS_BLOCKED",
                    message="GitLab API returned 403 Forbidden.",
                    retryable=False,
                )
            if status == 404:
                return (404, None, None)
            if status < 200 or status >= 300:
                raise CodeContextError(
                    code="IO_ERROR",
                    message=f"Unexpected GitLab HTTP status {status}.",
                    retryable=False,
                )
            chunks: list[bytes] = []
            total = 0
            for ch in resp.iter_bytes():
                check_deadline()
                total += len(ch)
                if total > MAX_HTTP_BODY_BYTES:
                    raise CodeContextError(
                        code="IO_ERROR",
                        message="GitLab response exceeded 32 MiB limit.",
                        retryable=False,
                    )
                chunks.append(ch)
            data = json.loads(b"".join(chunks).decode("utf-8"))
            raw_next = (resp.headers.get("X-Next-Page") or resp.headers.get("x-next-page") or "").strip()
            if raw_next:
                if not raw_next.isdigit() or int(raw_next) < 1:
                    raise CodeContextError(
                        code="IO_ERROR",
                        message=f"Invalid GitLab X-Next-Page header: {raw_next!r}",
                        retryable=False,
                    )
                next_page: str | None = raw_next
            else:
                next_page = None
            return (status, data, next_page)


def verify_and_bind_gitlab_project(
    store: Store,
    client: GitLabClient,
    *,
    project_path: str,
    remaining_requests: int = 1000,
) -> str:
    """Verify `/api/v4/user` when authenticated and `/api/v4/projects/{encoded_proj}`,
    enforcing principal binding (`store_meta.principal_id`) and immutable project ID.
    Returns `proj_id_str` (`gitlab:<numeric_id>`).
    """
    encoded_proj = urllib.parse.quote(project_path, safe="")
    start_reqs = client.requests_attempted

    with store.connect_read() as conn:
        meta = conn.execute(
            "SELECT github_repository_id, principal_id FROM store_meta WHERE singleton = 1"
        ).fetchone()
        assert meta is not None
        stored_proj_id = meta["github_repository_id"]
        stored_principal = meta["principal_id"]

    current_principal: str | None = None
    if client.token:
        st_u, user_data, _ = client.get_json("/user", remaining_requests=remaining_requests)
        if st_u == 404 or not isinstance(user_data, dict) or "id" not in user_data:
            raise CodeContextError(
                code="AUTH_REQUIRED",
                message="Unable to verify GitLab user principal via GET /api/v4/user.",
                retryable=False,
            )
        current_principal = f"gitlab:user:{user_data['id']}"
        if stored_principal is not None and stored_principal != current_principal:
            raise CodeContextError(
                code="REPOSITORY_MISMATCH",
                message="Configured GitLab principal differs from the principal bound to this store.",
                retryable=False,
            )
    elif stored_principal is not None:
        raise CodeContextError(
            code="AUTH_REQUIRED",
            message="Store is bound to an authenticated principal; GITLAB_TOKEN is required.",
            retryable=False,
        )

    rem = remaining_requests - (client.requests_attempted - start_reqs)
    status, proj_data, _ = client.get_json(f"/projects/{encoded_proj}", remaining_requests=rem)
    if status == 404 or not isinstance(proj_data, dict) or "id" not in proj_data:
        raise CodeContextError(
            code="ACCESS_BLOCKED",
            message=f"GitLab project {project_path!r} is unavailable or inaccessible.",
            retryable=False,
        )

    proj_num_id = int(proj_data["id"])
    visibility = str(proj_data.get("visibility") or "public").lower()
    if visibility != "public" and current_principal is None:
        raise CodeContextError(
            code="AUTH_REQUIRED",
            message="Non-public GitLab project requires a verified user principal.",
            retryable=False,
        )

    if stored_proj_id is not None and int(stored_proj_id) != proj_num_id:
        raise CodeContextError(
            code="REPOSITORY_MISMATCH",
            message=f"GitLab project numeric ID changed ({stored_proj_id} != {proj_num_id}).",
            retryable=False,
        )

    with store.write_transaction(bump_generation=False) as conn:
        conn.execute(
            """
            UPDATE store_meta
            SET github_repository_id = ?,
                principal_id = COALESCE(principal_id, ?)
            WHERE singleton = 1
            """,
            (proj_num_id, current_principal),
        )

    return f"gitlab:{proj_num_id}"


def _gitlab_cursor(store: Store, key: str) -> dict[str, Any]:
    with store.connect_read() as conn:
        row = conn.execute(
            "SELECT cursor_json FROM checkpoints WHERE stream_key = ?",
            (key,),
        ).fetchone()
    if row is None:
        return {}
    try:
        payload = json.loads(str(row["cursor_json"]))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_gitlab_cursor(store: Store, key: str, payload: dict[str, Any]) -> None:
    with store.write_transaction(bump_generation=False) as conn:
        conn.execute(
            """
            INSERT INTO checkpoints (stream_key, cursor_json, last_success_at_us)
            VALUES (?, ?, ?)
            ON CONFLICT(stream_key) DO UPDATE SET
                cursor_json = excluded.cursor_json,
                last_success_at_us = excluded.last_success_at_us
            """,
            (key, canonical_json(payload), now_us()),
        )


def collect_gitlab_merge_requests_and_issues(
    store: Store,
    client: GitLabClient,
    *,
    project_path: str,
    remaining_requests: int = 1000,
) -> tuple[int, int, int, bool]:
    """Collect GitLab project metadata, Merge Requests (`/merge_requests`), MR discussions/notes
    (`/merge_requests/{iid}/discussions`), and Issues (`/issues`) into the CodeContext store.
    Returns `(requests_used, sources_added, versions_added, completed)`.
    """
    encoded_proj = urllib.parse.quote(project_path, safe="")
    start_reqs = client.requests_attempted
    sources_added = 0
    versions_added = 0
    repo_uuid = store.config.repository_id

    def budget_left() -> int:
        return remaining_requests - (client.requests_attempted - start_reqs)

    proj_id_str = verify_and_bind_gitlab_project(
        store,
        client,
        project_path=project_path,
        remaining_requests=budget_left(),
    )
    stream_key = f"gitlab:{proj_id_str}:collection"
    cursor = _gitlab_cursor(store, stream_key)
    if cursor.get("phase") == "done":
        cursor = {}

    phase = str(cursor.get("phase") or "merge_requests")
    page = str(cursor.get("page") or "1")
    skip_iid = cursor.get("skip_iid") if isinstance(cursor.get("skip_iid"), int) else None
    discussion_iid = cursor.get("discussion_iid") if isinstance(cursor.get("discussion_iid"), int) else None
    discussion_page = str(cursor.get("discussion_page") or "1")

    def paused(payload: dict[str, Any]) -> tuple[int, int, int, bool]:
        _save_gitlab_cursor(store, stream_key, payload)
        return (client.requests_attempted - start_reqs, sources_added, versions_added, False)

    # 1. Enumerate Merge Requests (`/projects/{id}/merge_requests?state=all&per_page=100`)
    if phase != "issues":
        passed_skip = skip_iid is None
        while page:
            if budget_left() <= 0:
                return paused({"phase": "merge_requests", "page": page, "skip_iid": skip_iid,
                               "discussion_iid": discussion_iid, "discussion_page": discussion_page})
            st_mr, mrs_data, next_p = client.get_json(
                f"/projects/{encoded_proj}/merge_requests?state=all&per_page=100&page={page}",
                remaining_requests=budget_left(),
            )
            if st_mr == 404 or not isinstance(mrs_data, list):
                break

            for mr in mrs_data:
                if not isinstance(mr, dict) or "iid" not in mr:
                    continue
                iid = int(mr["iid"])
                if not passed_skip:
                    if iid == skip_iid:
                        passed_skip = True
                    continue
                merge_sha = mr.get("merge_commit_sha") or mr.get("squash_commit_sha")
                head_sha = mr.get("sha")
                author_raw = mr.get("author") or {}
                author_dict = {
                    "id": author_raw.get("id"),
                    "login": author_raw.get("username") or author_raw.get("name"),
                }
                with store.write_transaction(bump_generation=True) as conn:
                    mr_src_id, mr_ver_id, s_a, v_a = upsert_github_source_and_version(
                        conn,
                        repository_id=repo_uuid,
                        repository_provider_id=proj_id_str,
                        kind="pull_request",
                        provider_id=str(iid),
                        parent_source_id=None,
                        title=mr.get("title"),
                        body=mr.get("description"),
                        author_dict=author_dict,
                        source_url=mr.get("web_url"),
                        authored_at_us=_parse_opt_time_us(mr.get("created_at")),
                        updated_at_us=_parse_opt_time_us(mr.get("updated_at")),
                        anchors={
                            "commit_id": head_sha,
                            "merge_commit_sha": merge_sha,
                            "number": iid,
                        },
                        lifecycle_fields={
                            "merged_at": mr.get("merged_at"),
                            "state": mr.get("state"),
                        },
                        raw_payload=mr,
                    )
                    if s_a:
                        sources_added += 1
                    if v_a:
                        versions_added += 1

                    ep_id = make_episode_id(repository_id=repo_uuid, pr_source_id=mr_src_id)
                    cand_c = merge_sha or head_sha
                    ep_c_oid: str | None = None
                    if cand_c:
                        has_c = conn.execute("SELECT 1 FROM commits WHERE oid = ?", (cand_c,)).fetchone()
                        if has_c:
                            ep_c_oid = str(cand_c)
                    conn.execute(
                        """
                        INSERT INTO episodes (id, kind, primary_source_id, commit_oid, input_fingerprint, state)
                        VALUES (?, 'pr', ?, ?, ?, 'pending')
                        ON CONFLICT(id) DO UPDATE SET
                            commit_oid = COALESCE(episodes.commit_oid, excluded.commit_oid),
                            input_fingerprint = excluded.input_fingerprint
                        """,
                        (ep_id, mr_src_id, ep_c_oid, mr_ver_id),
                    )
                    for c_sha in [s for s in (merge_sha, head_sha) if s]:
                        conn.execute(
                            "DELETE FROM episodes WHERE kind = 'commit' AND commit_oid = ? AND state = 'pending'",
                            (c_sha,),
                        )
                        rel_id = make_relation_id(
                            from_kind="commit",
                            from_id=c_sha,
                            to_kind="source",
                            to_id=mr_src_id,
                            relation_kind="associated_pr",
                            method="gitlab_mr_association",
                        )
                        conn.execute(
                            """
                            INSERT INTO relations (
                                id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
                            ) VALUES (?, 'commit', ?, 'source', ?, 'associated_pr', 'gitlab_mr_association', 'exact', NULL, '{}', 'active')
                            ON CONFLICT(id) DO NOTHING
                            """,
                            (rel_id, c_sha, mr_src_id),
                        )

                disc_page = discussion_page if discussion_iid == iid else "1"
                while disc_page:
                    if budget_left() <= 0:
                        return paused(
                            {
                                "discussion_iid": iid,
                                "discussion_page": disc_page,
                                "page": page,
                                "phase": "merge_requests",
                                "skip_iid": skip_iid,
                            }
                        )
                    st_d, discs_data, next_d = client.get_json(
                        f"/projects/{encoded_proj}/merge_requests/{iid}/discussions?per_page=100&page={disc_page}",
                        remaining_requests=budget_left(),
                    )
                    if st_d == 200 and isinstance(discs_data, list):
                        with store.write_transaction(bump_generation=True) as conn:
                            for disc in discs_data:
                                if not isinstance(disc, dict):
                                    continue
                                notes = disc.get("notes")
                                if not isinstance(notes, list):
                                    continue
                                for note in notes:
                                    if not isinstance(note, dict) or note.get("system") is True or "id" not in note:
                                        continue
                                    n_auth = note.get("author") or {}
                                    pos = note.get("position")
                                    is_diff_note = isinstance(pos, dict) and bool(pos.get("new_path") or pos.get("old_path"))
                                    n_kind = "review_comment" if is_diff_note else "issue_comment"
                                    n_anchors = (
                                        {
                                            "commit_id": pos.get("head_sha"),
                                            "line": pos.get("new_line") or pos.get("old_line"),
                                            "original_commit_id": pos.get("start_sha") or pos.get("base_sha"),
                                            "original_line": pos.get("old_line") or pos.get("new_line"),
                                            "path": pos.get("new_path") or pos.get("old_path"),
                                            "side": "RIGHT" if pos.get("new_line") else "LEFT",
                                        }
                                        if is_diff_note and isinstance(pos, dict)
                                        else {"mr_iid": iid}
                                    )
                                    _, _, sa, va = upsert_github_source_and_version(
                                        conn,
                                        repository_id=repo_uuid,
                                        repository_provider_id=proj_id_str,
                                        kind=n_kind,
                                        provider_id=str(note["id"]),
                                        parent_source_id=mr_src_id,
                                        title=None,
                                        body=note.get("body"),
                                        author_dict={
                                            "id": n_auth.get("id"),
                                            "login": n_auth.get("username") or n_auth.get("name"),
                                        },
                                        source_url=None,
                                        authored_at_us=_parse_opt_time_us(note.get("created_at")),
                                        updated_at_us=_parse_opt_time_us(note.get("updated_at")),
                                        anchors=n_anchors,
                                        lifecycle_fields={},
                                        raw_payload=note,
                                    )
                                    if sa:
                                        sources_added += 1
                                    if va:
                                        versions_added += 1

                            child_vers = conn.execute(
                                "SELECT current_version_id FROM sources WHERE id = ? OR parent_source_id = ? ORDER BY id ASC",
                                (mr_src_id, mr_src_id),
                            ).fetchall()
                            ep_fp = canonical_sha256(
                                [str(r["current_version_id"]) for r in child_vers if r["current_version_id"]]
                            )
                            conn.execute(
                                "UPDATE episodes SET input_fingerprint = ? WHERE id = ?",
                                (ep_fp, ep_id),
                            )
                    disc_page = next_d if st_d == 200 else None
                skip_iid = iid
                discussion_iid = None
                discussion_page = "1"
                _save_gitlab_cursor(
                    store,
                    stream_key,
                    {"page": page, "phase": "merge_requests", "skip_iid": iid},
                )

            page = next_p
            skip_iid = None
            passed_skip = True
            if page:
                _save_gitlab_cursor(store, stream_key, {"page": page, "phase": "merge_requests"})
        phase = "issues"
        page = "1"

    # 2. Enumerate GitLab Issues (`/projects/{id}/issues?state=all&per_page=100`)
    iss_page = page if phase == "issues" else "1"
    while iss_page:
        if budget_left() <= 0:
            return paused({"page": iss_page, "phase": "issues"})
        st_i, issues_data, next_ip = client.get_json(
            f"/projects/{encoded_proj}/issues?state=all&per_page=100&page={iss_page}",
            remaining_requests=budget_left(),
        )
        if st_i == 404 or not isinstance(issues_data, list):
            break
        with store.write_transaction(bump_generation=True) as conn:
            for iss in issues_data:
                if not isinstance(iss, dict) or "iid" not in iss:
                    continue
                i_auth = iss.get("author") or {}
                _, _, sa, va = upsert_github_source_and_version(
                    conn,
                    repository_id=repo_uuid,
                    repository_provider_id=proj_id_str,
                    kind="issue",
                    provider_id=str(iss["iid"]),
                    parent_source_id=None,
                    title=iss.get("title"),
                    body=iss.get("description"),
                    author_dict={
                        "id": i_auth.get("id"),
                        "login": i_auth.get("username") or i_auth.get("name"),
                    },
                    source_url=iss.get("web_url"),
                    authored_at_us=_parse_opt_time_us(iss.get("created_at")),
                    updated_at_us=_parse_opt_time_us(iss.get("updated_at")),
                    anchors={"number": int(iss["iid"]), "repository": project_path},
                    lifecycle_fields={"state": iss.get("state")},
                    raw_payload=iss,
                )
                if sa:
                    sources_added += 1
                if va:
                    versions_added += 1
        iss_page = next_ip
        if iss_page:
            _save_gitlab_cursor(store, stream_key, {"page": iss_page, "phase": "issues"})

    _save_gitlab_cursor(store, stream_key, {"phase": "done"})
    return (client.requests_attempted - start_reqs, sources_added, versions_added, True)
