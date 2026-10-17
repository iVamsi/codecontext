import json
import os
import random
import re
import sqlite3
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

import httpx

from codecontext.errors import CodeContextError
from codecontext.identity import (
    byte_span_to_lines,
    canonical_json,
    canonical_sha256,
    datetime_to_us,
    hash_source_version_payload,
    make_episode_id,
    make_ev_id,
    make_relation_id,
    make_src_id,
    make_ver_id,
    parse_iso_utc,
    sha256_bytes,
)
from codecontext.store import Store, now_us
from codecontext.syntax import chunk_utf8_bytes, normalize_search_text

GITHUB_API_BASE = "https://api.github.com"
GITHUB_API_VERSION = "2026-03-10"
USER_AGENT = "codecontext/0.1.0"
MAX_HTTP_BODY_BYTES = 32 * 1024 * 1024  # 32 MiB
MAX_SOURCE_TEXT_BYTES = 4 * 1024 * 1024  # 4 MiB
PR_COMMITS_CAP = 250
PR_FILES_CAP = 3000

LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')
ISSUE_REF_RE = re.compile(r"(?:(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+))?#(?P<num>[1-9]\d*)")


def parse_link_next(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        m = LINK_NEXT_RE.search(part)
        if m:
            url = m.group(1).strip()
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme != "https" or parsed.netloc != "api.github.com":
                raise CodeContextError(
                    code="IO_ERROR",
                    message=f"Refusing cross-host pagination link: {parsed.netloc!r}",
                    retryable=False,
                )
            return url
    return None


def _parse_opt_time_us(val: Any) -> int | None:
    if not val or not isinstance(val, str):
        return None
    try:
        dt = parse_iso_utc(val)
        return datetime_to_us(dt)
    except Exception:
        return None


def extract_issue_references(text: str | None, default_repo: str) -> list[tuple[str, int]]:
    """Extract `(owner/repo, issue_number)` references from text."""
    if not text:
        return []
    refs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for m in ISSUE_REF_RE.finditer(text):
        repo = m.group("repo") or default_repo
        num = int(m.group("num"))
        key = (repo.lower(), num)
        if key not in seen:
            seen.add(key)
            refs.append((repo, num))
    return refs


@dataclass
class HttpFetchResult:
    status_code: int
    data: Any
    etag: str | None
    last_modified: str | None
    next_url: str | None
    requests_used: int
    rate_limit_reset_at: float | None = None
    body_too_large: bool = False


class GitHubClient:
    """Synchronous GET client for `https://api.github.com` with streaming 32 MiB enforcement,
    ETag/Last-Modified support, budget accounting, and rate-limit handling.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep_fn: Any = time.sleep,
    ) -> None:
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN")
        self._sleep_fn = sleep_fn
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._client = httpx.Client(
            base_url=GITHUB_API_BASE,
            headers=headers,
            transport=transport,
            follow_redirects=False,
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
        )
        self.requests_attempted = 0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GitHubClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def get_json(
        self,
        url_or_path: str,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
        remaining_requests: int = 1000,
        deadline_monotonic: float | None = None,
    ) -> HttpFetchResult:
        if remaining_requests <= 0:
            raise CodeContextError(
                code="BUDGET_EXHAUSTED",
                message="Sync request budget exhausted before HTTP call.",
                retryable=True,
            )

        if url_or_path.startswith("https://"):
            parsed = urllib.parse.urlparse(url_or_path)
            if parsed.scheme != "https" or parsed.netloc != "api.github.com":
                raise CodeContextError(
                    code="INVALID_REQUEST",
                    message="GitHubClient only permits https://api.github.com URLs.",
                    retryable=False,
                )
            target = url_or_path
        else:
            target = url_or_path if url_or_path.startswith("/") else f"/{url_or_path}"

        req_headers: dict[str, str] = {}
        if etag:
            req_headers["If-None-Match"] = etag
        if last_modified:
            req_headers["If-Modified-Since"] = last_modified

        attempts_this_call = 0
        max_attempts = min(3, remaining_requests)

        while attempts_this_call < max_attempts:
            if deadline_monotonic is not None:
                rem_sec = deadline_monotonic - time.monotonic()
                if rem_sec <= 0:
                    raise CodeContextError(
                        code="BUDGET_EXHAUSTED",
                        message="Sync time budget exhausted before HTTP call.",
                        retryable=True,
                    )
                req_timeout = httpx.Timeout(
                    connect=min(10.0, rem_sec),
                    read=min(30.0, rem_sec),
                    write=min(10.0, rem_sec),
                    pool=min(10.0, rem_sec),
                )
            else:
                req_timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)

            attempts_this_call += 1
            self.requests_attempted += 1

            try:
                with self._client.stream("GET", target, headers=req_headers, timeout=req_timeout) as resp:
                    status = resp.status_code

                    if status == 304:
                        return HttpFetchResult(
                            status_code=304,
                            data=None,
                            etag=resp.headers.get("ETag") or etag,
                            last_modified=resp.headers.get("Last-Modified") or last_modified,
                            next_url=None,
                            requests_used=attempts_this_call,
                        )

                    if status == 401:
                        raise CodeContextError(
                            code="AUTH_REQUIRED",
                            message="GitHub authentication failed (401 Unauthorized).",
                            retryable=False,
                        )

                    if status in (429, 403):
                        retry_after = resp.headers.get("Retry-After")
                        rl_rem = resp.headers.get("X-RateLimit-Remaining")
                        rl_reset = resp.headers.get("X-RateLimit-Reset")
                        is_rate_limit = (
                            status == 429
                            or retry_after is not None
                            or rl_rem == "0"
                        )
                        if is_rate_limit:
                            wait_sec = 60.0
                            if retry_after is not None and retry_after.isdigit():
                                wait_sec = float(int(retry_after))
                            elif rl_reset is not None and rl_reset.isdigit():
                                wait_sec = max(1.0, float(int(rl_reset)) - time.time())
                            raise CodeContextError(
                                code="RATE_LIMITED",
                                message="GitHub API rate limit exceeded.",
                                retryable=True,
                                details={"wait_seconds": int(wait_sec), "retry_at_offset": int(wait_sec)},
                            )
                        raise CodeContextError(
                            code="ACCESS_BLOCKED",
                            message="GitHub API returned 403 Forbidden.",
                            retryable=False,
                            details={"status_code": 403},
                        )

                    if status in (502, 503, 504):
                        if attempts_this_call < max_attempts:
                            delay = float(attempts_this_call) + random.uniform(0.0, 0.25)
                            if deadline_monotonic is None or (time.monotonic() + delay) < deadline_monotonic:
                                self._sleep_fn(delay)
                                continue
                        raise CodeContextError(
                            code="IO_ERROR",
                            message=f"GitHub API transient server error ({status}).",
                            retryable=True,
                            details={"status_code": status, "attempts": attempts_this_call},
                        )

                    if status == 404:
                        return HttpFetchResult(
                            status_code=404,
                            data=None,
                            etag=None,
                            last_modified=None,
                            next_url=None,
                            requests_used=attempts_this_call,
                        )

                    if status < 200 or status >= 300:
                        raise CodeContextError(
                            code="IO_ERROR",
                            message=f"Unexpected GitHub HTTP status {status}.",
                            retryable=False,
                            details={"status_code": status},
                        )

                    # Stream response up to MAX_HTTP_BODY_BYTES (32 MiB)
                    chunks: list[bytes] = []
                    total_bytes = 0
                    too_large = False
                    for chunk in resp.iter_bytes():
                        total_bytes += len(chunk)
                        if total_bytes > MAX_HTTP_BODY_BYTES:
                            too_large = True
                            break
                        chunks.append(chunk)

                    if too_large:
                        return HttpFetchResult(
                            status_code=status,
                            data=None,
                            etag=resp.headers.get("ETag"),
                            last_modified=resp.headers.get("Last-Modified"),
                            next_url=parse_link_next(resp.headers.get("Link")),
                            requests_used=attempts_this_call,
                            body_too_large=True,
                        )

                    raw_body = b"".join(chunks)
                    try:
                        parsed_json = json.loads(raw_body.decode("utf-8"))
                    except Exception as exc:
                        raise CodeContextError(
                            code="IO_ERROR",
                            message="Malformed JSON response from GitHub API.",
                            retryable=False,
                        ) from exc

                    return HttpFetchResult(
                        status_code=status,
                        data=parsed_json,
                        etag=resp.headers.get("ETag"),
                        last_modified=resp.headers.get("Last-Modified"),
                        next_url=parse_link_next(resp.headers.get("Link")),
                        requests_used=attempts_this_call,
                    )

            except httpx.TransportError as exc:
                if attempts_this_call < max_attempts:
                    delay = float(attempts_this_call) + random.uniform(0.0, 0.25)
                    if deadline_monotonic is None or (time.monotonic() + delay) < deadline_monotonic:
                        self._sleep_fn(delay)
                        continue
                raise CodeContextError(
                    code="IO_ERROR",
                    message="HTTP transport error communicating with GitHub API.",
                    retryable=True,
                    details={"attempts": attempts_this_call},
                ) from exc

        raise CodeContextError(
            code="IO_ERROR",
            message="Exhausted HTTP retry attempts.",
            retryable=True,
        )


def verify_and_bind_github_repository(
    store: Store,
    client: GitHubClient,
    *,
    remaining_requests: int = 1000,
    deadline_monotonic: float | None = None,
) -> tuple[str, str]:
    """Verify `GET /user` when authenticated and `GET /repos/{owner}/{repo}`.
    Enforces principal binding and immutable numeric repository ID.
    Returns `(repository_provider_id, repo_full_name)`.
    """
    if not store.config.github_repository:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Store has no configured github_repository.",
            retryable=False,
        )

    with store.connect_read() as conn:
        meta = conn.execute(
            "SELECT repository_id, github_repository_id, github_name, principal_id FROM store_meta WHERE singleton = 1"
        ).fetchone()
        assert meta is not None
        repo_uuid = str(meta["repository_id"])
        stored_gh_id = meta["github_repository_id"]
        stored_principal = meta["principal_id"]

    current_principal: str | None = None
    used = 0
    if client.token:
        user_res = client.get_json(
            "/user",
            remaining_requests=remaining_requests - used,
            deadline_monotonic=deadline_monotonic,
        )
        used += user_res.requests_used
        if user_res.status_code == 404 or not isinstance(user_res.data, dict) or "id" not in user_res.data:
            raise CodeContextError(
                code="AUTH_REQUIRED",
                message="Unsupported token type (e.g. installation token) or unable to verify principal via GET /user.",
                retryable=False,
            )
        current_principal = f"github:user:{user_res.data['id']}"
        if stored_principal is not None and stored_principal != current_principal:
            raise CodeContextError(
                code="REPOSITORY_MISMATCH",
                message="Configured GitHub principal differs from the principal bound to this store.",
                retryable=False,
            )
    else:
        if stored_principal is not None:
            raise CodeContextError(
                code="AUTH_REQUIRED",
                message="Store is bound to an authenticated principal; GITHUB_TOKEN is required.",
                retryable=False,
            )

    gh_name = store.config.github_repository
    repo_res = client.get_json(
        f"/repos/{gh_name}",
        remaining_requests=remaining_requests - used,
        deadline_monotonic=deadline_monotonic,
    )
    if repo_res.status_code == 404:
        with store.write_transaction(bump_generation=True) as conn:
            Store.upsert_gap(
                conn,
                repository_id=repo_uuid,
                code="SOURCE_UNAVAILABLE",
                scope=f"github:repo:{gh_name}",
                message=f"GitHub repository {gh_name} returned 404 (deleted or inaccessible).",
                retryable=True,
            )
        raise CodeContextError(
            code="ACCESS_BLOCKED",
            message=f"GitHub repository {gh_name} is unavailable or inaccessible (404).",
            retryable=False,
        )

    if not isinstance(repo_res.data, dict) or "id" not in repo_res.data:
        raise CodeContextError(
            code="IO_ERROR",
            message="Invalid repository payload from GitHub API.",
            retryable=False,
        )

    gh_num_id = int(repo_res.data["id"])
    full_name = str(repo_res.data.get("full_name") or gh_name)
    is_private = bool(repo_res.data.get("private", False))

    if is_private and current_principal is None:
        raise CodeContextError(
            code="AUTH_REQUIRED",
            message="Private GitHub repository requires a verified user principal.",
            retryable=False,
        )

    if stored_gh_id is not None and int(stored_gh_id) != gh_num_id:
        raise CodeContextError(
            code="REPOSITORY_MISMATCH",
            message=f"GitHub repository numeric ID changed ({stored_gh_id} != {gh_num_id}).",
            retryable=False,
        )

    with store.write_transaction(bump_generation=False) as conn:
        conn.execute(
            """
            UPDATE store_meta
            SET github_repository_id = ?,
                github_name = ?,
                principal_id = COALESCE(principal_id, ?)
            WHERE singleton = 1
            """,
            (gh_num_id, full_name, current_principal),
        )

    return str(gh_num_id), full_name


def upsert_github_source_and_version(
    conn: sqlite3.Connection,
    *,
    repository_id: str,
    repository_provider_id: str,
    kind: str,
    provider_id: str,
    parent_source_id: str | None,
    title: str | None,
    body: str | None,
    author_dict: dict[str, Any] | None,
    source_url: str | None,
    authored_at_us: int | None,
    updated_at_us: int | None,
    anchors: dict[str, Any] | None,
    lifecycle_fields: dict[str, Any] | None,
    raw_payload: dict[str, Any],
    observed_us: int | None = None,
) -> tuple[str, str, bool, bool]:
    """Normalize and persist a GitHub source and immutable `source_version` following Storage §4 Source refresh recipe.
    Returns `(src_id, ver_id, source_added, version_added)`.
    """
    ts = observed_us if observed_us is not None else now_us()
    src_id = make_src_id(
        provider="github",
        repository_provider_id=repository_provider_id,
        kind=kind,
        provider_id=provider_id,
    )

    # Check 4 MiB discussion body limit
    body_bytes = body.encode("utf-8") if body is not None else b""
    oversized = len(body_bytes) > MAX_SOURCE_TEXT_BYTES
    stored_body = None if oversized else body
    stored_raw_payload = (
        {"id": provider_id, "oversized": True} if oversized else raw_payload
    )

    author_id = str(author_dict["id"]) if author_dict and author_dict.get("id") is not None else None
    author_login = str(author_dict["login"]) if author_dict and author_dict.get("login") is not None else None
    author_json_str = (
        canonical_json({"actor_id": author_id, "display_name": author_login, "provider": "github"})
        if (author_id or author_login)
        else None
    )

    content_hash = hash_source_version_payload(
        title=title,
        body=stored_body,
        author_id=author_id,
        author_name=author_login,
        anchors=anchors,
        authored_at_us=authored_at_us,
        updated_at_us=updated_at_us,
        lifecycle_fields=lifecycle_fields,
    )
    ver_id = make_ver_id(source_id=src_id, content_sha256=content_hash)

    existing_src = conn.execute(
        "SELECT id, current_version_id FROM sources WHERE id = ?",
        (src_id,),
    ).fetchone()
    source_added = existing_src is None
    prev_ver_id = str(existing_src["current_version_id"]) if existing_src and existing_src["current_version_id"] else None

    if source_added:
        conn.execute(
            """
            INSERT INTO sources (
                id, provider, repository_provider_id, provider_id, kind,
                parent_source_id, current_version_id, availability, last_seen_at_us, last_reconciled_at_us
            ) VALUES (?, 'github', ?, ?, ?, ?, NULL, 'available', ?, NULL)
            """,
            (src_id, repository_provider_id, str(provider_id), kind, parent_source_id, ts),
        )
    else:
        conn.execute(
            "UPDATE sources SET availability = 'available', last_seen_at_us = ? WHERE id = ?",
            (ts, src_id),
        )

    if prev_ver_id == ver_id:
        conn.execute(
            "UPDATE source_versions SET last_observed_at_us = ? WHERE id = ?",
            (ts, ver_id),
        )
        return (src_id, ver_id, False, False)

    # Check if this immutable version already exists
    existing_ver = conn.execute(
        "SELECT id FROM source_versions WHERE id = ?",
        (ver_id,),
    ).fetchone()
    version_added = existing_ver is None
    if version_added:
        conn.execute(
            """
            INSERT INTO source_versions (
                id, source_id, content_sha256, title, body, raw_payload_json,
                author_json, source_url, authored_at_us, updated_at_us,
                first_observed_at_us, last_observed_at_us
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ver_id,
                src_id,
                content_hash,
                title,
                stored_body,
                canonical_json(stored_raw_payload),
                author_json_str,
                source_url,
                authored_at_us,
                updated_at_us,
                ts,
                ts,
            ),
        )
    else:
        conn.execute(
            "UPDATE source_versions SET last_observed_at_us = ? WHERE id = ?",
            (ts, ver_id),
        )

    # Update current_version_id on source
    conn.execute(
        "UPDATE sources SET current_version_id = ? WHERE id = ?",
        (ver_id, src_id),
    )

    if oversized:
        Store.upsert_gap(
            conn,
            repository_id=repository_id,
            code="SOURCE_BODY_TOO_LARGE",
            scope=f"github:source:{src_id}",
            message=f"Source {kind} {provider_id} body ({len(body_bytes)} bytes) exceeds 4 MiB limit; stored metadata only.",
            retryable=False,
            record_id=src_id,
            observed_us=ts,
        )

    # Register full-body discussion evidence locator for this version
    effective_bytes = stored_body.encode("utf-8") if stored_body is not None else b""
    l_s, l_e = byte_span_to_lines(effective_bytes, 0, len(effective_bytes))
    disc_anchor = {
        "git_oid": None,
        "kind": "discussion",
        "original_review_commit": (anchors or {}).get("original_commit_id"),
        "path": {"path": anchors["path"], "path_bytes_b64": None} if anchors and anchors.get("path") else None,
        "revision": (anchors or {}).get("commit_id"),
        "source_version_id": ver_id,
        "span": {
            "byte_end": len(effective_bytes),
            "byte_start": 0,
            "line_end": l_e,
            "line_start": l_s,
        },
        "underlying_record_id": src_id,
    }
    body_sha = sha256_bytes(effective_bytes)
    ev_id = make_ev_id(
        repository_id=repository_id,
        kind="discussion",
        source_version_id=ver_id,
        git_oid=None,
        anchor=disc_anchor,
        content_sha256=body_sha,
    )
    conn.execute(
        """
        INSERT INTO evidence (id, kind, source_version_id, git_oid, anchor_json, content_sha256, state)
        VALUES (?, 'discussion', ?, NULL, ?, ?, ?)
        ON CONFLICT(id) DO NOTHING
        """,
        (
            ev_id,
            ver_id,
            canonical_json(disc_anchor),
            body_sha,
            "oversized" if oversized else "available",
        ),
    )

    # If a previous version was replaced, recursively invalidate dependent claims/summaries
    # and mark previous version's FTS rows `current=false` (retaining version documents for cutoff search per T14)
    if prev_ver_id is not None and prev_ver_id != ver_id:
        Store.invalidate_dependents_recursive(conn, [("source_version", prev_ver_id)])
        prev_fts = conn.execute(
            "SELECT rowid, metadata_json FROM search_documents WHERE record_kind = 'history' AND record_id = ?",
            (prev_ver_id,),
        ).fetchall()
        for pf in prev_fts:
            m_dict = json.loads(str(pf["metadata_json"]))
            m_dict["current"] = False
            conn.execute(
                "UPDATE search_documents SET metadata_json = ? WHERE rowid = ?",
                (canonical_json(m_dict), int(pf["rowid"])),
            )

    # Index current source_version chunks in FTS `history` partition
    if not oversized and (title or stored_body):
        chunks = chunk_utf8_bytes(effective_bytes)
        disp_title = title or f"{kind} #{provider_id}"
        for ch in chunks:
            meta_json = canonical_json(
                {
                    "authored_at_us": authored_at_us,
                    "current": True,
                    "display_title": disp_title[:200],
                    "evidence_id": ev_id,
                    "kind": kind,
                    "path": anchors.get("path") if anchors else None,
                    "Snippet": ch.text[:320],
                    "snippet": ch.text[:320],
                    "source_id": src_id,
                    "updated_at_us": updated_at_us,
                    "version_id": ver_id,
                }
            )
            conn.execute(
                """
                INSERT INTO search_documents (record_kind, record_id, chunk_ordinal, title, identifiers, body, metadata_json)
                VALUES ('history', ?, ?, ?, ?, ?, ?)
                ON CONFLICT(record_kind, record_id, chunk_ordinal) DO UPDATE SET
                    title = excluded.title,
                    identifiers = excluded.identifiers,
                    body = excluded.body,
                    metadata_json = excluded.metadata_json
                """,
                (
                    ver_id,
                    ch.ordinal,
                    normalize_search_text(disp_title),
                    normalize_search_text(f"{kind} {provider_id} {author_login or ''}"),
                    normalize_search_text(f"{title or ''} {ch.text}"),
                    meta_json,
                ),
            )

    return (src_id, ver_id, source_added, version_added)


def ingest_pr_detail_and_children(
    store: Store,
    client: GitHubClient,
    *,
    repo_provider_id: str,
    repo_full_name: str,
    pr_number: int,
    associated_commit_oids: list[str] | None = None,
    remaining_requests: int = 1000,
    deadline_monotonic: float | None = None,
) -> tuple[int, int, int, bool]:
    """Fetch and checkpoint a PR (`/pulls/{number}`) and its paginated child streams:
    - `/issues/{number}/comments`
    - `/pulls/{number}/reviews`
    - `/pulls/{number}/comments` (inline review comments)
    - `/issues/{number}/timeline`
    - `/pulls/{number}/commits` (up to 250 cap)
    - `/pulls/{number}/files` (up to 3000 cap)
    - One-hop linked issues in the same repo or `allowed_issue_repositories`.
    Supports resuming from committed page checkpoints when interrupted by budget.
    Returns `(requests_used, sources_added, versions_added, completed)`.
    """
    repo_uuid = store.config.repository_id
    used_requests = 0
    sources_added = 0
    versions_added = 0

    def budget_left() -> int:
        return remaining_requests - used_requests

    # 1. Fetch `/pulls/{pr_number}` if not yet checkpointed or when refreshing
    pr_stream_key = f"github:{repo_provider_id}:pr:{pr_number}:detail"
    pr_res = client.get_json(
        f"/repos/{repo_full_name}/pulls/{pr_number}",
        remaining_requests=budget_left(),
        deadline_monotonic=deadline_monotonic,
    )
    used_requests += pr_res.requests_used

    if pr_res.status_code == 404:
        with store.write_transaction(bump_generation=True) as conn:
            Store.upsert_gap(
                conn,
                repository_id=repo_uuid,
                code="SOURCE_UNAVAILABLE",
                scope=f"github:pr:{pr_number}",
                message=f"PR #{pr_number} returned 404.",
                retryable=False,
            )
        return (used_requests, 0, 0, True)

    if pr_res.body_too_large or not isinstance(pr_res.data, dict):
        with store.write_transaction(bump_generation=True) as conn:
            Store.upsert_gap(
                conn,
                repository_id=repo_uuid,
                code="HTTP_BODY_TOO_LARGE" if pr_res.body_too_large else "IO_ERROR",
                scope=f"github:pr:{pr_number}",
                message=f"PR #{pr_number} detail response exceeded 32 MiB limit.",
                retryable=False,
            )
        return (used_requests, 0, 0, True)

    pr_data = pr_res.data
    head_sha = (pr_data.get("head") or {}).get("sha")
    base_sha = (pr_data.get("base") or {}).get("sha")
    merge_sha = pr_data.get("merge_commit_sha")
    pr_anchors = {
        "base_sha": base_sha,
        "commit_id": head_sha,
        "merge_commit_sha": merge_sha,
        "number": pr_number,
    }
    pr_lifecycle = {
        "merged_at": pr_data.get("merged_at"),
        "state": pr_data.get("state"),
    }

    with store.write_transaction(bump_generation=True) as conn:
        pr_src_id, pr_ver_id, s_add, v_add = upsert_github_source_and_version(
            conn,
            repository_id=repo_uuid,
            repository_provider_id=repo_provider_id,
            kind="pull_request",
            provider_id=str(pr_data.get("number") or pr_number),
            parent_source_id=None,
            title=pr_data.get("title"),
            body=pr_data.get("body"),
            author_dict=pr_data.get("user"),
            source_url=pr_data.get("html_url"),
            authored_at_us=_parse_opt_time_us(pr_data.get("created_at")),
            updated_at_us=_parse_opt_time_us(pr_data.get("updated_at")),
            anchors=pr_anchors,
            lifecycle_fields=pr_lifecycle,
            raw_payload=pr_data,
        )
        if s_add:
            sources_added += 1
        if v_add:
            versions_added += 1

        # Associate commits with PR and create/update PR episode
        ep_id = make_episode_id(repository_id=repo_uuid, pr_source_id=pr_src_id)
        candidate_ep_commit = merge_sha or head_sha
        ep_commit_oid: str | None = None
        if candidate_ep_commit:
            has_c = conn.execute(
                "SELECT 1 FROM commits WHERE oid = ?", (candidate_ep_commit,)
            ).fetchone()
            if has_c:
                ep_commit_oid = str(candidate_ep_commit)

        conn.execute(
            """
            INSERT INTO episodes (id, kind, primary_source_id, commit_oid, input_fingerprint, state)
            VALUES (?, 'pr', ?, ?, ?, 'pending')
            ON CONFLICT(id) DO UPDATE SET
                commit_oid = COALESCE(episodes.commit_oid, excluded.commit_oid),
                input_fingerprint = excluded.input_fingerprint,
                state = CASE WHEN episodes.input_fingerprint != excluded.input_fingerprint THEN 'pending' ELSE episodes.state END
            """,
            (ep_id, pr_src_id, ep_commit_oid, pr_ver_id),
        )
        all_assoc_shas = list(
            dict.fromkeys(
                [*(associated_commit_oids or []), *([merge_sha] if merge_sha else []), *([head_sha] if head_sha else [])]
            )
        )
        for c_sha in all_assoc_shas:
            if c_sha:
                # Remove standalone commit-only episode once a PR episode owns this commit
                conn.execute(
                    "DELETE FROM episodes WHERE kind = 'commit' AND commit_oid = ? AND state = 'pending'",
                    (c_sha,),
                )
                rel_id = make_relation_id(
                    from_kind="commit",
                    from_id=c_sha,
                    to_kind="source",
                    to_id=pr_src_id,
                    relation_kind="associated_pr",
                    method="github_pr_association",
                )
                conn.execute(
                    """
                    INSERT INTO relations (
                        id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
                    ) VALUES (?, 'commit', ?, 'source', ?, 'associated_pr', 'github_pr_association', 'exact', NULL, '{}', 'active')
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (rel_id, c_sha, pr_src_id),
                )

        ts = now_us()
        conn.execute(
            """
            INSERT INTO checkpoints (stream_key, cursor_json, etag, last_modified, last_success_at_us)
            VALUES (?, '{"done":true}', ?, ?, ?)
            ON CONFLICT(stream_key) DO UPDATE SET
                cursor_json = excluded.cursor_json,
                etag = excluded.etag,
                last_modified = excluded.last_modified,
                last_success_at_us = excluded.last_success_at_us
            """,
            (pr_stream_key, pr_res.etag, pr_res.last_modified, ts),
        )

    # Helper to paginate a child stream with page-by-page checkpointing
    child_streams = [
        ("issue_comments", f"/repos/{repo_full_name}/issues/{pr_number}/comments?per_page=100"),
        ("reviews", f"/repos/{repo_full_name}/pulls/{pr_number}/reviews?per_page=100"),
        ("review_comments", f"/repos/{repo_full_name}/pulls/{pr_number}/comments?per_page=100"),
        ("timeline", f"/repos/{repo_full_name}/issues/{pr_number}/timeline?per_page=100"),
        ("pr_commits", f"/repos/{repo_full_name}/pulls/{pr_number}/commits?per_page=100"),
        ("pr_files", f"/repos/{repo_full_name}/pulls/{pr_number}/files?per_page=100"),
    ]

    discovered_issue_refs: list[tuple[str, int]] = extract_issue_references(
        f"{pr_data.get('title') or ''} {pr_data.get('body') or ''}",
        repo_full_name,
    )

    for stream_kind, initial_path in child_streams:
        s_key = f"github:{repo_provider_id}:pr:{pr_number}:{stream_kind}"
        with store.connect_read() as r_conn:
            ckpt = r_conn.execute(
                "SELECT cursor_json FROM checkpoints WHERE stream_key = ?",
                (s_key,),
            ).fetchone()
        if ckpt:
            c_state = json.loads(str(ckpt["cursor_json"]))
            if c_state.get("done") is True:
                continue
            next_url: str | None = c_state.get("next_url") or initial_path
            seen_count = int(c_state.get("count", 0))
            started_from_first_page = next_url == initial_path
        else:
            next_url = initial_path
            seen_count = 0
            started_from_first_page = True
        stream_seen_ids: set[str] = set()

        while next_url is not None:
            if budget_left() <= 0:
                with store.write_transaction(bump_generation=True) as conn:
                    Store.upsert_gap(
                        conn,
                        repository_id=repo_uuid,
                        code="GITHUB_BACKFILL_PENDING",
                        scope=s_key,
                        message=f"PR #{pr_number} stream {stream_kind} paused by request budget.",
                        retryable=True,
                    )
                return (used_requests, sources_added, versions_added, False)

            page_res = client.get_json(
                next_url,
                remaining_requests=budget_left(),
                deadline_monotonic=deadline_monotonic,
            )
            used_requests += page_res.requests_used

            if page_res.status_code == 404:
                # Optional or missing endpoint on mock/server; mark done
                with store.write_transaction(bump_generation=False) as conn:
                    conn.execute(
                        """
                        INSERT INTO checkpoints (stream_key, cursor_json, last_success_at_us)
                        VALUES (?, '{"done":true}', ?)
                        ON CONFLICT(stream_key) DO UPDATE SET cursor_json = '{"done":true}', last_success_at_us = excluded.last_success_at_us
                        """,
                        (s_key, now_us()),
                    )
                break

            items = page_res.data if isinstance(page_res.data, list) else []
            seen_count += len(items)
            next_url = page_res.next_url

            with store.write_transaction(bump_generation=True) as conn:
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    if stream_kind == "issue_comments" and "id" in item:
                        ic_sid, _, s_a, v_a = upsert_github_source_and_version(
                            conn,
                            repository_id=repo_uuid,
                            repository_provider_id=repo_provider_id,
                            kind="issue_comment",
                            provider_id=str(item["id"]),
                            parent_source_id=pr_src_id,
                            title=None,
                            body=item.get("body"),
                            author_dict=item.get("user"),
                            source_url=item.get("html_url"),
                            authored_at_us=_parse_opt_time_us(item.get("created_at")),
                            updated_at_us=_parse_opt_time_us(item.get("updated_at")),
                            anchors={"pr_number": pr_number},
                            lifecycle_fields={},
                            raw_payload=item,
                        )
                        if s_a:
                            sources_added += 1
                        if v_a:
                            versions_added += 1
                        stream_seen_ids.add(ic_sid)
                        discovered_issue_refs.extend(
                            extract_issue_references(item.get("body"), repo_full_name)
                        )

                    elif stream_kind == "reviews" and "id" in item:
                        rv_sid, _, s_a, v_a = upsert_github_source_and_version(
                            conn,
                            repository_id=repo_uuid,
                            repository_provider_id=repo_provider_id,
                            kind="review",
                            provider_id=str(item["id"]),
                            parent_source_id=pr_src_id,
                            title=None,
                            body=item.get("body"),
                            author_dict=item.get("user"),
                            source_url=item.get("html_url"),
                            authored_at_us=_parse_opt_time_us(item.get("submitted_at")),
                            updated_at_us=_parse_opt_time_us(item.get("submitted_at")),
                            anchors={"commit_id": item.get("commit_id"), "pr_number": pr_number},
                            lifecycle_fields={"state": item.get("state")},
                            raw_payload=item,
                        )
                        if s_a:
                            sources_added += 1
                        if v_a:
                            versions_added += 1
                        stream_seen_ids.add(rv_sid)

                    elif stream_kind == "review_comments" and "id" in item:
                        rc_anchors = {
                            "commit_id": item.get("commit_id"),
                            "diff_hunk": item.get("diff_hunk"),
                            "in_reply_to_id": item.get("in_reply_to_id"),
                            "line": item.get("line"),
                            "original_commit_id": item.get("original_commit_id"),
                            "original_line": item.get("original_line"),
                            "original_position": item.get("original_position"),
                            "original_start_line": item.get("original_start_line"),
                            "path": item.get("path"),
                            "position": item.get("position"),
                            "pull_request_review_id": item.get("pull_request_review_id"),
                            "side": item.get("side", "RIGHT"),
                            "start_line": item.get("start_line"),
                            "start_side": item.get("start_side"),
                        }
                        rc_src_id, _, s_a, v_a = upsert_github_source_and_version(
                            conn,
                            repository_id=repo_uuid,
                            repository_provider_id=repo_provider_id,
                            kind="review_comment",
                            provider_id=str(item["id"]),
                            parent_source_id=pr_src_id,
                            title=None,
                            body=item.get("body"),
                            author_dict=item.get("user"),
                            source_url=item.get("html_url"),
                            authored_at_us=_parse_opt_time_us(item.get("created_at")),
                            updated_at_us=_parse_opt_time_us(item.get("updated_at")),
                            anchors=rc_anchors,
                            lifecycle_fields={},
                            raw_payload=item,
                        )
                        if s_a:
                            sources_added += 1
                        if v_a:
                            versions_added += 1
                        stream_seen_ids.add(rc_src_id)
                        if item.get("side") == "LEFT" and not base_sha:
                            Store.upsert_gap(
                                conn,
                                repository_id=repo_uuid,
                                code="REVIEW_COMPARISON_MISSING",
                                scope=f"github:review_comment:{item['id']}",
                                message=f"LEFT-side review comment {item['id']} is missing original base comparison.",
                                retryable=False,
                                record_id=rc_src_id,
                            )

                    elif stream_kind == "timeline":
                        ev_id_raw = item.get("id") or f"{pr_number}_{item.get('event')}_{item.get('created_at')}"
                        ev_type = str(item.get("event") or "unknown")
                        if ev_type in (
                            "referenced",
                            "cross-referenced",
                            "connected",
                            "disconnected",
                            "closed",
                            "reopened",
                            "review_dismissed",
                        ):
                            tl_sid, _, s_a, v_a = upsert_github_source_and_version(
                                conn,
                                repository_id=repo_uuid,
                                repository_provider_id=repo_provider_id,
                                kind="timeline_event",
                                provider_id=str(ev_id_raw),
                                parent_source_id=pr_src_id,
                                title=ev_type,
                                body=item.get("commit_id") or ev_type,
                                author_dict=item.get("actor"),
                                source_url=item.get("url"),
                                authored_at_us=_parse_opt_time_us(item.get("created_at")),
                                updated_at_us=_parse_opt_time_us(item.get("created_at")),
                                anchors={"commit_id": item.get("commit_id"), "event": ev_type},
                                lifecycle_fields={"event": ev_type},
                                raw_payload=item,
                            )
                            if s_a:
                                sources_added += 1
                            if v_a:
                                versions_added += 1
                            stream_seen_ids.add(tl_sid)

                    elif stream_kind == "pr_commits":
                        c_sha = item.get("sha")
                        if isinstance(c_sha, str) and c_sha:
                            rel_id = make_relation_id(
                                from_kind="commit",
                                from_id=c_sha,
                                to_kind="source",
                                to_id=pr_src_id,
                                relation_kind="associated_pr",
                                method="github_pr_commits_endpoint",
                            )
                            conn.execute(
                                """
                                INSERT INTO relations (
                                    id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
                                ) VALUES (?, 'commit', ?, 'source', ?, 'associated_pr', 'github_pr_commits_endpoint', 'exact', NULL, '{}', 'active')
                                ON CONFLICT(id) DO NOTHING
                                """,
                                (rel_id, c_sha, pr_src_id),
                            )

                # Check PR commits (250) and PR files (3000) endpoint caps
                if stream_kind == "pr_commits" and (
                    seen_count >= PR_COMMITS_CAP
                    or (isinstance(pr_data.get("commits"), int) and int(pr_data["commits"]) > PR_COMMITS_CAP)
                ):
                    Store.upsert_gap(
                        conn,
                        repository_id=repo_uuid,
                        code="API_RESULT_CAP",
                        scope=f"github:pr:{pr_number}:commits",
                        message=f"PR #{pr_number} commits endpoint reached the 250-commit REST API cap; recover remaining commits from Git.",
                        retryable=False,
                        record_id=pr_src_id,
                    )
                if stream_kind == "pr_files" and (
                    seen_count >= PR_FILES_CAP
                    or (isinstance(pr_data.get("changed_files"), int) and int(pr_data["changed_files"]) > PR_FILES_CAP)
                ):
                    Store.upsert_gap(
                        conn,
                        repository_id=repo_uuid,
                        code="API_RESULT_CAP",
                        scope=f"github:pr:{pr_number}:files",
                        message=f"PR #{pr_number} files endpoint reached the 3,000-file REST API cap; use Git diff-tree as authority.",
                        retryable=False,
                        record_id=pr_src_id,
                    )

                # Update episode input_fingerprint to reflect all current child versions
                child_ver_rows = conn.execute(
                    "SELECT current_version_id FROM sources WHERE id = ? OR parent_source_id = ? ORDER BY id ASC",
                    (pr_src_id, pr_src_id),
                ).fetchall()
                ep_fp = canonical_sha256(
                    [str(r["current_version_id"]) for r in child_ver_rows if r["current_version_id"]]
                )
                conn.execute(
                    """
                    UPDATE episodes
                    SET input_fingerprint = ?,
                        state = CASE WHEN input_fingerprint != ? THEN 'pending' ELSE state END
                    WHERE id = ?
                    """,
                    (ep_fp, ep_fp, ep_id),
                )

                ckpt_payload = (
                    {"count": seen_count, "done": True, "next_url": None}
                    if next_url is None
                    else {"count": seen_count, "done": False, "next_url": next_url}
                )
                conn.execute(
                    """
                    INSERT INTO checkpoints (stream_key, cursor_json, etag, last_modified, last_success_at_us)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(stream_key) DO UPDATE SET
                        cursor_json = excluded.cursor_json,
                        etag = excluded.etag,
                        last_modified = excluded.last_modified,
                        last_success_at_us = excluded.last_success_at_us
                    """,
                    (s_key, canonical_json(ckpt_payload), page_res.etag, page_res.last_modified, now_us()),
                )
                if next_url is None:
                    Store.resolve_gaps_in_scope(conn, scope=s_key, code="GITHUB_BACKFILL_PENDING")
                    # If this child stream was fetched from its first page through completion,
                    # any previously seen active child of this kind not in `stream_seen_ids` is marked `deleted`
                    stream_to_src_kind = {
                        "issue_comments": "issue_comment",
                        "reviews": "review",
                        "review_comments": "review_comment",
                        "timeline": "timeline_event",
                    }
                    sk = stream_to_src_kind.get(stream_kind)
                    if sk and started_from_first_page:
                        existing_children = conn.execute(
                            "SELECT id, current_version_id FROM sources WHERE parent_source_id = ? AND kind = ? AND availability = 'available'",
                            (pr_src_id, sk),
                        ).fetchall()
                        for ec in existing_children:
                            ec_id = str(ec["id"])
                            if ec_id not in stream_seen_ids:
                                conn.execute(
                                    "UPDATE sources SET availability = 'deleted', last_reconciled_at_us = ? WHERE id = ?",
                                    (now_us(), ec_id),
                                )
                                if ec["current_version_id"]:
                                    Store.invalidate_dependents_recursive(
                                        conn, [("source_version", str(ec["current_version_id"]))]
                                    )
                                    conn.execute(
                                        "DELETE FROM search_documents WHERE record_kind = 'history' AND record_id = ?",
                                        (str(ec["current_version_id"]),),
                                    )

    # Also scan stored child source_versions of this PR for issue references so resumed runs see refs from earlier pages
    with store.connect_read() as r_conn:
        stored_child_bodies = r_conn.execute(
            """
            SELECT sv.title, sv.body
            FROM sources s
            JOIN source_versions sv ON sv.id = s.current_version_id
            WHERE s.id = ? OR s.parent_source_id = ?
            """,
            (pr_src_id, pr_src_id),
        ).fetchall()
    for r in stored_child_bodies:
        discovered_issue_refs.extend(
            extract_issue_references(f"{r['title'] or ''} {r['body'] or ''}", repo_full_name)
        )

    # Fetch one-hop linked issues (same repo or in `allowed_issue_repositories`)
    allowed_set = {r.lower() for r in store.config.allowed_issue_repositories} | {repo_full_name.lower()}
    seen_issue_refs: set[tuple[str, int]] = set()
    for ref_repo, ref_num in discovered_issue_refs:
        ref_key = (ref_repo.lower(), ref_num)
        if ref_key in seen_issue_refs:
            continue
        seen_issue_refs.add(ref_key)
        if ref_num == pr_number and ref_repo.lower() == repo_full_name.lower():
            continue
        if ref_repo.lower() not in allowed_set:
            continue
        iss_ckpt_key = f"github:linked_issue:{ref_repo.lower()}:{ref_num}"
        with store.connect_read() as r_conn:
            iss_ckpt = r_conn.execute(
                "SELECT cursor_json FROM checkpoints WHERE stream_key = ?",
                (iss_ckpt_key,),
            ).fetchone()
        if iss_ckpt and json.loads(str(iss_ckpt["cursor_json"])).get("done") is True:
            continue
        if budget_left() <= 0:
            break
        iss_res = client.get_json(
            f"/repos/{ref_repo}/issues/{ref_num}",
            remaining_requests=budget_left(),
            deadline_monotonic=deadline_monotonic,
        )
        used_requests += iss_res.requests_used
        if iss_res.status_code == 200 and isinstance(iss_res.data, dict) and "pull_request" not in iss_res.data:
            iss_data = iss_res.data
            target_repo_prov_id = (
                repo_provider_id if ref_repo.lower() == repo_full_name.lower() else ref_repo.lower()
            )
            with store.write_transaction(bump_generation=True) as conn:
                iss_src_id, _, s_a, v_a = upsert_github_source_and_version(
                    conn,
                    repository_id=repo_uuid,
                    repository_provider_id=target_repo_prov_id,
                    kind="issue",
                    provider_id=str(iss_data.get("number") or ref_num),
                    parent_source_id=None,
                    title=iss_data.get("title"),
                    body=iss_data.get("body"),
                    author_dict=iss_data.get("user"),
                    source_url=iss_data.get("html_url"),
                    authored_at_us=_parse_opt_time_us(iss_data.get("created_at")),
                    updated_at_us=_parse_opt_time_us(iss_data.get("updated_at")),
                    anchors={"number": ref_num, "repository": ref_repo},
                    lifecycle_fields={"state": iss_data.get("state")},
                    raw_payload=iss_data,
                )
                if s_a:
                    sources_added += 1
                if v_a:
                    versions_added += 1
                rel_id = make_relation_id(
                    from_kind="source",
                    from_id=pr_src_id,
                    to_kind="source",
                    to_id=iss_src_id,
                    relation_kind="references_issue",
                    method="explicit_text_reference",
                )
                conn.execute(
                    """
                    INSERT INTO relations (
                        id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
                    ) VALUES (?, 'source', ?, 'source', ?, 'references_issue', 'explicit_text_reference', 'exact', NULL, '{}', 'active')
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (rel_id, pr_src_id, iss_src_id),
                )
                conn.execute(
                    """
                    INSERT INTO checkpoints (stream_key, cursor_json, last_success_at_us)
                    VALUES (?, '{"done":true}', ?)
                    ON CONFLICT(stream_key) DO UPDATE SET cursor_json = '{"done":true}', last_success_at_us = excluded.last_success_at_us
                    """,
                    (iss_ckpt_key, now_us()),
                )

    return (used_requests, sources_added, versions_added, True)


def collect_focused_github_evidence(
    store: Store,
    client: GitHubClient,
    *,
    commit_oids: list[str],
    remaining_requests: int = 1000,
    deadline_monotonic: float | None = None,
) -> tuple[int, int, int, bool]:
    """For focus commits, query `/commits/{sha}/pulls` ( up to 25 episodes ) and acquire their
    PR details, conversations, reviews, inline review comments, and one-hop allowed issues.
    Returns `(requests_used, sources_added, versions_added, all_complete)`.
    """
    if not store.config.github_repository:
        return (0, 0, 0, True)

    start_reqs = client.requests_attempted
    repo_prov_id, repo_full_name = verify_and_bind_github_repository(
        store,
        client,
        remaining_requests=remaining_requests,
        deadline_monotonic=deadline_monotonic,
    )
    used_requests = client.requests_attempted - start_reqs
    sources_added = 0
    versions_added = 0

    pr_to_commits: dict[int, list[str]] = {}
    all_complete = True

    for c_oid in commit_oids[:100]:
        if remaining_requests - used_requests <= 0:
            all_complete = False
            break
        assoc_key = f"github:{repo_prov_id}:commit_pulls:{c_oid}"
        with store.connect_read() as r_conn:
            ckpt = r_conn.execute(
                "SELECT cursor_json FROM checkpoints WHERE stream_key = ?",
                (assoc_key,),
            ).fetchone()
        if ckpt:
            c_state = json.loads(str(ckpt["cursor_json"]))
            if c_state.get("done") is True:
                for pnum in c_state.get("pr_numbers", []):
                    pr_to_commits.setdefault(int(pnum), []).append(c_oid)
                continue

        res = client.get_json(
            f"/repos/{repo_full_name}/commits/{c_oid}/pulls",
            remaining_requests=remaining_requests - used_requests,
            deadline_monotonic=deadline_monotonic,
        )
        used_requests += res.requests_used
        pr_nums: list[int] = []
        if res.status_code == 200 and isinstance(res.data, list):
            for pr_item in res.data:
                if isinstance(pr_item, dict) and "number" in pr_item:
                    pnum = int(pr_item["number"])
                    pr_nums.append(pnum)
                    pr_to_commits.setdefault(pnum, []).append(c_oid)

        with store.write_transaction(bump_generation=False) as conn:
            conn.execute(
                """
                INSERT INTO checkpoints (stream_key, cursor_json, last_success_at_us)
                VALUES (?, ?, ?)
                ON CONFLICT(stream_key) DO UPDATE SET cursor_json = excluded.cursor_json, last_success_at_us = excluded.last_success_at_us
                """,
                (assoc_key, canonical_json({"done": True, "pr_numbers": pr_nums}), now_us()),
            )

    for pr_num, assoc_commits in list(pr_to_commits.items())[:25]:
        if remaining_requests - used_requests <= 0:
            all_complete = False
            break
        u, s_a, v_a, done = ingest_pr_detail_and_children(
            store,
            client,
            repo_provider_id=repo_prov_id,
            repo_full_name=repo_full_name,
            pr_number=pr_num,
            associated_commit_oids=assoc_commits,
            remaining_requests=remaining_requests - used_requests,
            deadline_monotonic=deadline_monotonic,
        )
        used_requests += u
        sources_added += s_a
        versions_added += v_a
        if not done:
            all_complete = False
            break

    # Mark full GitHub backfill as pending until full enumeration completes in T13
    with store.write_transaction(bump_generation=True) as conn:
        full_ckpt = conn.execute(
            "SELECT cursor_json FROM checkpoints WHERE stream_key = ?",
            (f"github:{repo_prov_id}:issues_all",),
        ).fetchone()
        full_done = bool(full_ckpt and json.loads(str(full_ckpt["cursor_json"])).get("done") is True)
        if not full_done:
            Store.upsert_gap(
                conn,
                repository_id=store.config.repository_id,
                code="GITHUB_BACKFILL_PENDING",
                scope=f"github:{repo_prov_id}:backfill",
                message="Full repository GitHub issue/PR backfill is pending.",
                retryable=True,
            )

    return (used_requests, sources_added, versions_added, all_complete)


def ingest_issue_and_children(
    store: Store,
    client: GitHubClient,
    *,
    repo_provider_id: str,
    repo_full_name: str,
    issue_data: dict[str, Any],
    remaining_requests: int = 1000,
    deadline_monotonic: float | None = None,
    reconcile_pass_id: str | None = None,
    reconcile_stream_key: str | None = None,
) -> tuple[int, int, int, bool]:
    """Persist an issue and fetch its conversation comments (`/issues/{number}/comments`)
    and timeline (`/issues/{number}/timeline`), plus one-hop allowlisted issue references.
    Returns `(requests_used, sources_added, versions_added, completed)`.
    """
    repo_uuid = store.config.repository_id
    used_requests = 0
    sources_added = 0
    versions_added = 0
    issue_num = int(issue_data["number"])

    def budget_left() -> int:
        return remaining_requests - used_requests

    with store.write_transaction(bump_generation=True) as conn:
        iss_src_id, _, s_add, v_add = upsert_github_source_and_version(
            conn,
            repository_id=repo_uuid,
            repository_provider_id=repo_provider_id,
            kind="issue",
            provider_id=str(issue_num),
            parent_source_id=None,
            title=issue_data.get("title"),
            body=issue_data.get("body"),
            author_dict=issue_data.get("user"),
            source_url=issue_data.get("html_url"),
            authored_at_us=_parse_opt_time_us(issue_data.get("created_at")),
            updated_at_us=_parse_opt_time_us(issue_data.get("updated_at")),
            anchors={"number": issue_num, "repository": repo_full_name},
            lifecycle_fields={"state": issue_data.get("state")},
            raw_payload=issue_data,
        )
        if s_add:
            sources_added += 1
        if v_add:
            versions_added += 1
        if reconcile_pass_id and reconcile_stream_key:
            conn.execute(
                """
                INSERT INTO reconciliation_seen (stream_key, pass_id, source_id)
                VALUES (?, ?, ?)
                ON CONFLICT(stream_key, pass_id, source_id) DO NOTHING
                """,
                (reconcile_stream_key, reconcile_pass_id, iss_src_id),
            )

    discovered_refs = extract_issue_references(
        f"{issue_data.get('title') or ''} {issue_data.get('body') or ''}",
        repo_full_name,
    )

    child_streams = [
        ("issue_comments", f"/repos/{repo_full_name}/issues/{issue_num}/comments?per_page=100"),
        ("timeline", f"/repos/{repo_full_name}/issues/{issue_num}/timeline?per_page=100"),
    ]
    for stream_kind, initial_path in child_streams:
        s_key = f"github:{repo_provider_id}:issue:{issue_num}:{stream_kind}"
        with store.connect_read() as r_conn:
            ckpt = r_conn.execute(
                "SELECT cursor_json FROM checkpoints WHERE stream_key = ?", (s_key,)
            ).fetchone()
        if ckpt and reconcile_pass_id is None:
            c_state = json.loads(str(ckpt["cursor_json"]))
            if c_state.get("done") is True:
                continue
            next_url: str | None = c_state.get("next_url") or initial_path
        else:
            next_url = initial_path

        while next_url is not None:
            if budget_left() <= 0:
                return (used_requests, sources_added, versions_added, False)
            page_res = client.get_json(
                next_url,
                remaining_requests=budget_left(),
                deadline_monotonic=deadline_monotonic,
            )
            used_requests += page_res.requests_used
            if page_res.status_code == 404:
                break
            items = page_res.data if isinstance(page_res.data, list) else []
            next_url = page_res.next_url

            with store.write_transaction(bump_generation=True) as conn:
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    if stream_kind == "issue_comments" and "id" in item:
                        c_src_id, _, s_a, v_a = upsert_github_source_and_version(
                            conn,
                            repository_id=repo_uuid,
                            repository_provider_id=repo_provider_id,
                            kind="issue_comment",
                            provider_id=str(item["id"]),
                            parent_source_id=iss_src_id,
                            title=None,
                            body=item.get("body"),
                            author_dict=item.get("user"),
                            source_url=item.get("html_url"),
                            authored_at_us=_parse_opt_time_us(item.get("created_at")),
                            updated_at_us=_parse_opt_time_us(item.get("updated_at")),
                            anchors={"issue_number": issue_num},
                            lifecycle_fields={},
                            raw_payload=item,
                        )
                        if s_a:
                            sources_added += 1
                        if v_a:
                            versions_added += 1
                        if reconcile_pass_id and reconcile_stream_key:
                            conn.execute(
                                """
                                INSERT INTO reconciliation_seen (stream_key, pass_id, source_id)
                                VALUES (?, ?, ?)
                                ON CONFLICT(stream_key, pass_id, source_id) DO NOTHING
                                """,
                                (reconcile_stream_key, reconcile_pass_id, c_src_id),
                            )
                        discovered_refs.extend(
                            extract_issue_references(item.get("body"), repo_full_name)
                        )
                    elif stream_kind == "timeline":
                        ev_id_raw = item.get("id") or f"{issue_num}_{item.get('event')}_{item.get('created_at')}"
                        ev_type = str(item.get("event") or "unknown")
                        if ev_type in (
                            "referenced",
                            "cross-referenced",
                            "connected",
                            "disconnected",
                            "closed",
                            "reopened",
                        ):
                            t_src_id, _, s_a, v_a = upsert_github_source_and_version(
                                conn,
                                repository_id=repo_uuid,
                                repository_provider_id=repo_provider_id,
                                kind="timeline_event",
                                provider_id=str(ev_id_raw),
                                parent_source_id=iss_src_id,
                                title=ev_type,
                                body=item.get("commit_id") or ev_type,
                                author_dict=item.get("actor"),
                                source_url=item.get("url"),
                                authored_at_us=_parse_opt_time_us(item.get("created_at")),
                                updated_at_us=_parse_opt_time_us(item.get("created_at")),
                                anchors={"commit_id": item.get("commit_id"), "event": ev_type},
                                lifecycle_fields={"event": ev_type},
                                raw_payload=item,
                            )
                            if s_a:
                                sources_added += 1
                            if v_a:
                                versions_added += 1
                            if reconcile_pass_id and reconcile_stream_key:
                                conn.execute(
                                    """
                                    INSERT INTO reconciliation_seen (stream_key, pass_id, source_id)
                                    VALUES (?, ?, ?)
                                    ON CONFLICT(stream_key, pass_id, source_id) DO NOTHING
                                    """,
                                    (reconcile_stream_key, reconcile_pass_id, t_src_id),
                                )
                            # Platform-confirmed cross-referenced issue/PR link
                            src_issue = (item.get("source") or {}).get("issue")
                            if isinstance(src_issue, dict) and "number" in src_issue:
                                target_num = int(src_issue["number"])
                                target_kind = "pull_request" if "pull_request" in src_issue else "issue"
                                target_sid = make_src_id(
                                    provider="github",
                                    repository_provider_id=repo_provider_id,
                                    kind=target_kind,
                                    provider_id=str(target_num),
                                )
                                rel_id = make_relation_id(
                                    from_kind="source",
                                    from_id=iss_src_id,
                                    to_kind="source",
                                    to_id=target_sid,
                                    relation_kind="references_issue",
                                    method="platform_timeline_event",
                                )
                                conn.execute(
                                    """
                                    INSERT INTO relations (
                                        id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
                                    ) VALUES (?, 'source', ?, 'source', ?, 'references_issue', 'platform_timeline_event', 'exact', NULL, '{}', 'active')
                                    ON CONFLICT(id) DO NOTHING
                                    """,
                                    (rel_id, iss_src_id, target_sid),
                                )

                ckpt_payload = (
                    {"done": True, "next_url": None}
                    if next_url is None
                    else {"done": False, "next_url": next_url}
                )
                conn.execute(
                    """
                    INSERT INTO checkpoints (stream_key, cursor_json, etag, last_modified, last_success_at_us)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(stream_key) DO UPDATE SET
                        cursor_json = excluded.cursor_json,
                        etag = excluded.etag,
                        last_modified = excluded.last_modified,
                        last_success_at_us = excluded.last_success_at_us
                    """,
                    (s_key, canonical_json(ckpt_payload), page_res.etag, page_res.last_modified, now_us()),
                )

    # Fetch one-hop allowlisted external issues referenced in this issue
    allowed_set = {r.lower() for r in store.config.allowed_issue_repositories} | {repo_full_name.lower()}
    for ref_repo, ref_num in discovered_refs:
        if ref_num == issue_num and ref_repo.lower() == repo_full_name.lower():
            continue
        if ref_repo.lower() not in allowed_set:
            continue
        if budget_left() <= 0:
            break
        ext_res = client.get_json(
            f"/repos/{ref_repo}/issues/{ref_num}",
            remaining_requests=budget_left(),
            deadline_monotonic=deadline_monotonic,
        )
        used_requests += ext_res.requests_used
        if ext_res.status_code == 200 and isinstance(ext_res.data, dict) and "pull_request" not in ext_res.data:
            ext_data = ext_res.data
            ext_prov_id = (
                repo_provider_id if ref_repo.lower() == repo_full_name.lower() else ref_repo.lower()
            )
            with store.write_transaction(bump_generation=True) as conn:
                ext_src_id, _, s_a, v_a = upsert_github_source_and_version(
                    conn,
                    repository_id=repo_uuid,
                    repository_provider_id=ext_prov_id,
                    kind="issue",
                    provider_id=str(ext_data.get("number") or ref_num),
                    parent_source_id=None,
                    title=ext_data.get("title"),
                    body=ext_data.get("body"),
                    author_dict=ext_data.get("user"),
                    source_url=ext_data.get("html_url"),
                    authored_at_us=_parse_opt_time_us(ext_data.get("created_at")),
                    updated_at_us=_parse_opt_time_us(ext_data.get("updated_at")),
                    anchors={"number": ref_num, "repository": ref_repo},
                    lifecycle_fields={"state": ext_data.get("state")},
                    raw_payload=ext_data,
                )
                if s_a:
                    sources_added += 1
                if v_a:
                    versions_added += 1
                rel_id = make_relation_id(
                    from_kind="source",
                    from_id=iss_src_id,
                    to_kind="source",
                    to_id=ext_src_id,
                    relation_kind="references_issue",
                    method="explicit_text_reference",
                )
                conn.execute(
                    """
                    INSERT INTO relations (
                        id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
                    ) VALUES (?, 'source', ?, 'source', ?, 'references_issue', 'explicit_text_reference', 'exact', NULL, '{}', 'active')
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (rel_id, iss_src_id, ext_src_id),
                )

    return (used_requests, sources_added, versions_added, True)


def collect_full_and_incremental_github(
    store: Store,
    client: GitHubClient,
    *,
    reconcile: bool = False,
    remaining_requests: int = 1000,
    deadline_monotonic: float | None = None,
) -> tuple[int, int, int, bool]:
    """Execute full GitHub `/issues?state=all&sort=created&direction=asc` backfill,
    incremental comment update streams (`/issues/comments?sort=updated&direction=asc&since=...`
    and `/pulls/comments?sort=updated&direction=asc&since=...`), and `reconcile=True` full
    reconciliation pass with `reconciliation_seen` missing-source detection.
    Returns `(requests_used, sources_added, versions_added, completed)`.
    """
    if not store.config.github_repository:
        return (0, 0, 0, True)

    sync_start_us = now_us()
    start_reqs = client.requests_attempted
    repo_prov_id, repo_full_name = verify_and_bind_github_repository(
        store,
        client,
        remaining_requests=remaining_requests,
        deadline_monotonic=deadline_monotonic,
    )
    used_requests = client.requests_attempted - start_reqs
    sources_added = 0
    versions_added = 0
    repo_uuid = store.config.repository_id

    def budget_left() -> int:
        return remaining_requests - used_requests

    issues_stream_key = f"github:{repo_prov_id}:issues_all"
    with store.connect_read() as r_conn:
        ckpt = r_conn.execute(
            "SELECT cursor_json, high_watermark_us FROM checkpoints WHERE stream_key = ?",
            (issues_stream_key,),
        ).fetchone()

    pass_id: str | None = None
    initial_issues_url = f"/repos/{repo_full_name}/issues?state=all&sort=created&direction=asc&per_page=100"

    if reconcile:
        if ckpt:
            c_state = json.loads(str(ckpt["cursor_json"]))
            if c_state.get("reconciling") is True and c_state.get("pass_id"):
                pass_id = str(c_state["pass_id"])
                next_url: str | None = c_state.get("next_url") or initial_issues_url
            else:
                pass_id = f"pass_{sync_start_us}"
                next_url = initial_issues_url
                with store.write_transaction(bump_generation=False) as conn:
                    # Reset PR child checkpoints so reconcile re-enumerates all child streams from page 1
                    conn.execute(
                        "DELETE FROM checkpoints WHERE stream_key LIKE ?",
                        (f"github:{repo_prov_id}:pr:%",),
                    )
                    conn.execute(
                        "DELETE FROM checkpoints WHERE stream_key LIKE ?",
                        (f"github:{repo_prov_id}:issue:%",),
                    )
                    conn.execute(
                        """
                        INSERT INTO checkpoints (stream_key, cursor_json, last_success_at_us)
                        VALUES (?, ?, ?)
                        ON CONFLICT(stream_key) DO UPDATE SET cursor_json = excluded.cursor_json
                        """,
                        (
                            issues_stream_key,
                            canonical_json({"done": False, "next_url": next_url, "pass_id": pass_id, "reconciling": True}),
                            sync_start_us,
                        ),
                    )
        else:
            pass_id = f"pass_{sync_start_us}"
            next_url = initial_issues_url
            with store.write_transaction(bump_generation=False) as conn:
                conn.execute(
                    """
                    INSERT INTO checkpoints (stream_key, cursor_json, last_success_at_us)
                    VALUES (?, ?, ?)
                    """,
                    (
                        issues_stream_key,
                        canonical_json({"done": False, "next_url": next_url, "pass_id": pass_id, "reconciling": True}),
                        sync_start_us,
                    ),
                )
    else:
        if ckpt:
            c_state = json.loads(str(ckpt["cursor_json"]))
            next_url = None if c_state.get("done") is True else (c_state.get("next_url") or initial_issues_url)
        else:
            next_url = initial_issues_url

    # 1. Enumerate `/issues?state=all&sort=created&direction=asc`
    while next_url is not None:
        if budget_left() <= 0:
            return (used_requests, sources_added, versions_added, False)

        page_res = client.get_json(
            next_url,
            remaining_requests=budget_left(),
            deadline_monotonic=deadline_monotonic,
        )
        used_requests += page_res.requests_used
        if page_res.status_code == 404:
            next_url = None
            break

        items = page_res.data if isinstance(page_res.data, list) else []
        pending_items_on_page = list(items)

        all_children_done = True
        for item in pending_items_on_page:
            if not isinstance(item, dict) or "number" not in item:
                continue
            num = int(item["number"])
            if "pull_request" in item:
                # Normalize PR-shaped issue listing to one `pull_request` source via `/pulls/{number}`
                if budget_left() <= 0:
                    all_children_done = False
                    break
                u, s_a, v_a, pr_done = ingest_pr_detail_and_children(
                    store,
                    client,
                    repo_provider_id=repo_prov_id,
                    repo_full_name=repo_full_name,
                    pr_number=num,
                    remaining_requests=budget_left(),
                    deadline_monotonic=deadline_monotonic,
                )
                used_requests += u
                sources_added += s_a
                versions_added += v_a
                if pass_id:
                    with store.write_transaction(bump_generation=False) as conn:
                        pr_sid = make_src_id(
                            provider="github",
                            repository_provider_id=repo_prov_id,
                            kind="pull_request",
                            provider_id=str(num),
                        )
                        family = conn.execute(
                            "SELECT id FROM sources WHERE id = ? OR parent_source_id = ?",
                            (pr_sid, pr_sid),
                        ).fetchall()
                        for fr in family:
                            conn.execute(
                                """
                                INSERT INTO reconciliation_seen (stream_key, pass_id, source_id)
                                VALUES (?, ?, ?)
                                ON CONFLICT(stream_key, pass_id, source_id) DO NOTHING
                                """,
                                (issues_stream_key, pass_id, str(fr["id"])),
                            )
                if not pr_done:
                    all_children_done = False
                    break
            else:
                if budget_left() <= 0:
                    all_children_done = False
                    break
                u, s_a, v_a, iss_done = ingest_issue_and_children(
                    store,
                    client,
                    repo_provider_id=repo_prov_id,
                    repo_full_name=repo_full_name,
                    issue_data=item,
                    remaining_requests=budget_left(),
                    deadline_monotonic=deadline_monotonic,
                    reconcile_pass_id=pass_id,
                    reconcile_stream_key=issues_stream_key,
                )
                used_requests += u
                sources_added += s_a
                versions_added += v_a
                if not iss_done:
                    all_children_done = False
                    break

        if not all_children_done:
            # Keep `next_url` pointing at the current page so resume finishes remaining items on this page
            return (used_requests, sources_added, versions_added, False)

        next_url = page_res.next_url
        with store.write_transaction(bump_generation=True) as conn:
            ckpt_payload = {
                "done": next_url is None,
                "next_url": next_url,
                "pass_id": pass_id,
                "reconciling": bool(reconcile and next_url is not None),
            }
            conn.execute(
                """
                INSERT INTO checkpoints (stream_key, cursor_json, etag, last_modified, high_watermark_us, last_success_at_us)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(stream_key) DO UPDATE SET
                    cursor_json = excluded.cursor_json,
                    etag = excluded.etag,
                    last_modified = excluded.last_modified,
                    high_watermark_us = COALESCE(checkpoints.high_watermark_us, excluded.high_watermark_us),
                    last_success_at_us = excluded.last_success_at_us
                """,
                (issues_stream_key, canonical_json(ckpt_payload), page_res.etag, page_res.last_modified, sync_start_us, now_us()),
            )
            if next_url is None:
                Store.resolve_gaps_in_scope(
                    conn,
                    scope=f"github:{repo_prov_id}:backfill",
                    code="GITHUB_BACKFILL_PENDING",
                )

    # 2. If a full reconciliation sweep (`reconcile=True` and `pass_id` present) just completed:
    # mark any primary-repository sources NOT in `reconciliation_seen(issues_stream_key, pass_id)` as `deleted`
    # and invalidate their dependent claims/summaries!
    if reconcile and pass_id is not None:
        with store.write_transaction(bump_generation=True) as conn:
            unseen_rows = conn.execute(
                """
                SELECT s.id, s.current_version_id
                FROM sources s
                WHERE s.repository_provider_id = ?
                  AND s.availability = 'available'
                  AND s.id NOT IN (
                      SELECT source_id FROM reconciliation_seen WHERE stream_key = ? AND pass_id = ?
                  )
                """,
                (repo_prov_id, issues_stream_key, pass_id),
            ).fetchall()
            for ur in unseen_rows:
                u_sid = str(ur["id"])
                u_vid = ur["current_version_id"]
                conn.execute(
                    "UPDATE sources SET availability = 'deleted', last_reconciled_at_us = ? WHERE id = ?",
                    (sync_start_us, u_sid),
                )
                if u_vid:
                    Store.invalidate_dependents_recursive(conn, [("source_version", str(u_vid))])
                    conn.execute(
                        "DELETE FROM search_documents WHERE record_kind = 'history' AND record_id = ?",
                        (str(u_vid),),
                    )
            conn.execute(
                "UPDATE sources SET last_reconciled_at_us = ? WHERE repository_provider_id = ? AND availability = 'available'",
                (sync_start_us, repo_prov_id),
            )
            conn.execute(
                "UPDATE checkpoints SET last_reconciled_at_us = ? WHERE stream_key = ?",
                (sync_start_us, issues_stream_key),
            )
            conn.execute(
                "DELETE FROM reconciliation_seen WHERE stream_key = ? AND pass_id = ?",
                (issues_stream_key, pass_id),
            )

    # 3. Run repository-wide incremental comment update streams (`/issues/comments` and `/pulls/comments`)
    # using watermark minus 5 minutes overlap window
    five_min_us = 5 * 60 * 1_000_000
    for inc_kind, inc_endpoint in [
        ("issue_comments_inc", f"/repos/{repo_full_name}/issues/comments"),
        ("review_comments_inc", f"/repos/{repo_full_name}/pulls/comments"),
    ]:
        inc_key = f"github:{repo_prov_id}:{inc_kind}"
        with store.connect_read() as r_conn:
            inc_ckpt = r_conn.execute(
                "SELECT high_watermark_us FROM checkpoints WHERE stream_key = ?",
                (inc_key,),
            ).fetchone()
        wm_us = int(inc_ckpt["high_watermark_us"]) if inc_ckpt and inc_ckpt["high_watermark_us"] else None
        if wm_us is not None:
            from codecontext.identity import us_to_iso_z

            since_iso = us_to_iso_z(max(0, wm_us - five_min_us))
            inc_url: str | None = f"{inc_endpoint}?sort=updated&direction=asc&since={since_iso}&per_page=100"
        else:
            inc_url = f"{inc_endpoint}?sort=updated&direction=asc&per_page=100"

        while inc_url is not None:
            if budget_left() <= 0:
                return (used_requests, sources_added, versions_added, False)
            inc_res = client.get_json(
                inc_url,
                remaining_requests=budget_left(),
                deadline_monotonic=deadline_monotonic,
            )
            used_requests += inc_res.requests_used
            if inc_res.status_code == 404:
                break
            inc_items = inc_res.data if isinstance(inc_res.data, list) else []
            inc_url = inc_res.next_url

            with store.write_transaction(bump_generation=True) as conn:
                for item in inc_items:
                    if not isinstance(item, dict) or "id" not in item:
                        continue
                    c_id_str = str(item["id"])
                    target_kind = "issue_comment" if inc_kind == "issue_comments_inc" else "review_comment"
                    existing_s = conn.execute(
                        "SELECT id, parent_source_id FROM sources WHERE provider = 'github' AND repository_provider_id = ? AND kind = ? AND provider_id = ?",
                        (repo_prov_id, target_kind, c_id_str),
                    ).fetchone()
                    parent_sid = str(existing_s["parent_source_id"]) if existing_s and existing_s["parent_source_id"] else None
                    rc_anchors = (
                        {
                            "commit_id": item.get("commit_id"),
                            "diff_hunk": item.get("diff_hunk"),
                            "in_reply_to_id": item.get("in_reply_to_id"),
                            "line": item.get("line"),
                            "original_commit_id": item.get("original_commit_id"),
                            "original_line": item.get("original_line"),
                            "original_position": item.get("original_position"),
                            "original_start_line": item.get("original_start_line"),
                            "path": item.get("path"),
                            "position": item.get("position"),
                            "pull_request_review_id": item.get("pull_request_review_id"),
                            "side": item.get("side", "RIGHT"),
                            "start_line": item.get("start_line"),
                            "start_side": item.get("start_side"),
                        }
                        if target_kind == "review_comment"
                        else {}
                    )
                    _, new_vid, s_a, v_a = upsert_github_source_and_version(
                        conn,
                        repository_id=repo_uuid,
                        repository_provider_id=repo_prov_id,
                        kind=target_kind,
                        provider_id=c_id_str,
                        parent_source_id=parent_sid,
                        title=None,
                        body=item.get("body"),
                        author_dict=item.get("user"),
                        source_url=item.get("html_url"),
                        authored_at_us=_parse_opt_time_us(item.get("created_at")),
                        updated_at_us=_parse_opt_time_us(item.get("updated_at")),
                        anchors=rc_anchors,
                        lifecycle_fields={},
                        raw_payload=item,
                    )
                    if s_a:
                        sources_added += 1
                    if v_a:
                        versions_added += 1
                        if parent_sid:
                            conn.execute(
                                "UPDATE episodes SET state = 'pending' WHERE primary_source_id = ?",
                                (parent_sid,),
                            )

                if inc_url is None:
                    conn.execute(
                        """
                        INSERT INTO checkpoints (stream_key, cursor_json, high_watermark_us, last_success_at_us)
                        VALUES (?, '{"done":true}', ?, ?)
                        ON CONFLICT(stream_key) DO UPDATE SET
                            cursor_json = '{"done":true}',
                            high_watermark_us = excluded.high_watermark_us,
                            last_success_at_us = excluded.last_success_at_us
                        """,
                        (inc_key, sync_start_us, now_us()),
                    )

    return (used_requests, sources_added, versions_added, True)

