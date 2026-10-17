import json
import os
import urllib.parse
from pathlib import Path
from typing import Any

import httpx

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
    import subprocess

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
            return parse_gitlab_remote_url(proc.stdout.strip())
    except OSError:
        pass
    return None


def parse_gitlab_remote_url(raw_url: str) -> tuple[str, str] | None:
    """Parse a GitLab SSH or HTTPS remote URL into `(https_base_url, namespace/project)`."""
    url = raw_url.strip()
    if not url or "github.com" in url.lower():
        return None
    if url.startswith("git@"):
        # git@gitlab.com:group/sub/repo.git
        colon_idx = url.find(":")
        if colon_idx == -1:
            return None
        host = url[4:colon_idx]
        if "gitlab" not in host.lower():
            return None
        proj = url[colon_idx + 1 :].strip("/")
        if proj.endswith(".git"):
            proj = proj[:-4]
        if "/" in proj:
            return (f"https://{host}", proj)
        return None

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in ("http", "https") and parsed.netloc and "gitlab" in parsed.netloc.lower():
        proj = parsed.path.strip("/")
        if proj.endswith(".git"):
            proj = proj[:-4]
        if "/" in proj:
            return (f"{parsed.scheme}://{parsed.netloc}", proj)
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
        self.base_url = base_url.rstrip("/")
        self.token = token if token is not None else os.environ.get("GITLAB_TOKEN")
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
        with self._client.stream("GET", target) as resp:
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
                total += len(ch)
                if total > MAX_HTTP_BODY_BYTES:
                    raise CodeContextError(
                        code="IO_ERROR",
                        message="GitLab response exceeded 32 MiB limit.",
                        retryable=False,
                    )
                chunks.append(ch)
            data = json.loads(b"".join(chunks).decode("utf-8"))
            next_page = resp.headers.get("X-Next-Page") or resp.headers.get("x-next-page")
            return (status, data, next_page if next_page and next_page.strip() else None)


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

    status, proj_data, _ = client.get_json(
        f"/projects/{encoded_proj}", remaining_requests=budget_left()
    )
    if status == 404 or not isinstance(proj_data, dict) or "id" not in proj_data:
        raise CodeContextError(
            code="ACCESS_BLOCKED",
            message=f"GitLab project {project_path!r} is unavailable or inaccessible.",
            retryable=False,
        )

    proj_id_str = f"gitlab:{proj_data['id']}"

    # 1. Enumerate Merge Requests (`/projects/{id}/merge_requests?state=all&per_page=100`)
    page = "1"
    while page:
        if budget_left() <= 0:
            return (client.requests_attempted - start_reqs, sources_added, versions_added, False)
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

            # Fetch MR discussions/notes (`/projects/{id}/merge_requests/{iid}/discussions`)
            if budget_left() > 0:
                st_d, discs_data, _ = client.get_json(
                    f"/projects/{encoded_proj}/merge_requests/{iid}/discussions?per_page=100",
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

        page = next_p

    # 2. Enumerate GitLab Issues (`/projects/{id}/issues?state=all&per_page=100`)
    iss_page = "1"
    while iss_page:
        if budget_left() <= 0:
            return (client.requests_attempted - start_reqs, sources_added, versions_added, False)
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

    with store.write_transaction(bump_generation=False) as conn:
        conn.execute(
            """
            INSERT INTO checkpoints (stream_key, cursor_json, last_success_at_us)
            VALUES (?, ?, ?)
            ON CONFLICT(stream_key) DO UPDATE SET cursor_json = excluded.cursor_json, last_success_at_us = excluded.last_success_at_us
            """,
            (f"github:{proj_id_str}:issues_all", canonical_json({"done": True}), now_us()),
        )

    return (client.requests_attempted - start_reqs, sources_added, versions_added, True)
