import json
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from codecontext.deadline import SyncDeadlineExceeded, check_deadline, expired, sync_deadline
from codecontext.errors import CodeContextError
from codecontext.git import GitMirror
from codecontext.github import (
    GitHubClient,
    collect_focused_github_evidence,
    collect_full_and_incremental_github,
)
from codecontext.gitlab import (
    GitLabClient,
    collect_gitlab_merge_requests_and_issues,
    detect_gitlab_repository_from_git,
)
from codecontext.identity import us_to_iso_z
from codecontext.lineage import (
    diff_first_parents,
    index_commit_changes_and_hunks,
    index_focused_path_history,
)
from codecontext.models import PathTarget, SymbolTarget, SyncReport, SyncRequest
from codecontext.store import Store, now_us
from codecontext.syntax import index_active_snapshot_files

COMMIT_DIFF_BATCH_SIZE = 32
MAX_DIFF_JOB_ATTEMPTS = 3


def execute_sync(
    store: Store,
    mirror: GitMirror,
    request: SyncRequest,
    *,
    github_transport: httpx.BaseTransport | None = None,
    github_token: str | None = None,
    gitlab_token: str | None = None,
) -> SyncReport:
    started = now_us()
    with store.connect_read() as conn:
        commits_before = conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
        extractions_before = conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0]
        sources_before = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        versions_before = conn.execute("SELECT COUNT(*) FROM source_versions").fetchone()[0]
    progress: dict[str, Any] = {"structure_complete": False}
    with sync_deadline(request.max_seconds):
        try:
            return _execute_sync(
                store,
                mirror,
                request,
                progress=progress,
                github_transport=github_transport,
                github_token=github_token,
                gitlab_token=gitlab_token,
            )
        except (SyncDeadlineExceeded, CodeContextError, sqlite3.OperationalError, subprocess.TimeoutExpired, httpx.TransportError) as exc:
            if not isinstance(exc, SyncDeadlineExceeded) and not expired():
                raise
    # The interrupted transaction rolls back; earlier file/commit checkpoints remain usable.
    with store.write_transaction() as conn:
        conn.execute("UPDATE jobs SET state = 'pending' WHERE state = 'running' AND kind = 'git_commit_diff'")
        gap_code = "GIT_HISTORY_PENDING" if progress["structure_complete"] else "STRUCTURE_PENDING"
        Store.upsert_gap(conn, repository_id=store.config.repository_id, code=gap_code,
                         scope="sync:deadline", message="Sync reached its time budget. Run sync to finish the requested work.", retryable=True)
        coverage, gaps, _ = Store.build_coverage_and_gaps(conn, max_detailed_gaps=20)
        meta = conn.execute("SELECT * FROM store_meta WHERE singleton = 1").fetchone()
        jobs = Store.get_queue_counts(conn)
        commits_added = conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0] - commits_before
        blobs_parsed = conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0] - extractions_before
        sources_added = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] - sources_before
        versions_added = conn.execute("SELECT COUNT(*) FROM source_versions").fetchone()[0] - versions_before
    finished = now_us()
    return SyncReport(repository_id=store.config.repository_id, revision=meta["active_revision"],
                      generation=meta["generation"] + 1, access_epoch=meta["access_epoch"],
                      coverage=coverage, gaps=gaps, state="partial", started_at=us_to_iso_z(started),
                      finished_at=us_to_iso_z(finished), elapsed_ms=(finished-started)//1000,
                      requests_attempted=getattr(progress.get("http_client"), "requests_attempted", 0),
                      commits_added=max(0, commits_added), blobs_parsed=max(0, blobs_parsed),
                      sources_added=max(0, sources_added), versions_added=max(0, versions_added),
                      cache_hits=0, jobs=jobs, stop_reason="time_budget")


def _drain_commit_diff_jobs(
    store: Store, mirror: GitMirror, deadline_mono: float, failed_this_sync: list[str]
) -> bool:
    """Process pending `git_commit_diff` jobs in batches that share one `cat-file --batch`
    session and one write transaction. Each job runs in its own savepoint, so a job that fails
    leaves no partial rows and does not stop the rest of the batch. A job that fails is skipped for
    the rest of this sync (`failed_this_sync`) and parks in `retryable_error` with an IO_ERROR gap
    after MAX_DIFF_JOB_ATTEMPTS, until a reconcile sync makes it pending again.
    Returns False when the time budget is exhausted before the queue empties.
    """
    while True:
        if time.monotonic() >= deadline_mono:
            return False
        with store.connect_read() as r_conn:
            jobs = r_conn.execute(
                """
                SELECT id, owner_id FROM jobs
                WHERE state = 'pending' AND kind = 'git_commit_diff'
                  AND id NOT IN (SELECT value FROM json_each(?))
                ORDER BY priority ASC, created_at_us ASC, id ASC
                LIMIT ?
                """,
                (json.dumps(failed_this_sync), COMMIT_DIFF_BATCH_SIZE),
            ).fetchall()
        if not jobs:
            return True
        with mirror.open_batch_session() as batch, store.write_transaction(bump_generation=True) as conn:
            p0_by_commit = diff_first_parents(
                conn, mirror, batch, [str(job["owner_id"]) for job in jobs]
            )
            for job in jobs:
                check_deadline()
                if time.monotonic() >= deadline_mono:
                    break
                job_id = str(job["id"])
                commit_oid = str(job["owner_id"])
                try:
                    with store.write_transaction(bump_generation=True) as job_conn:
                        index_commit_changes_and_hunks(
                            store,
                            mirror,
                            commit_oid,
                            batch=batch,
                            conn=job_conn,
                            p0_changes=p0_by_commit.get(commit_oid),
                        )
                        job_conn.execute(
                            "UPDATE jobs SET state = 'complete', updated_at_us = ? WHERE id = ?",
                            (now_us(), job_id),
                        )
                        Store.resolve_gaps_in_scope(
                            job_conn, scope=f"git:commit:{commit_oid}", code="IO_ERROR"
                        )
                except (SyncDeadlineExceeded, sqlite3.Error):
                    # Deadline interrupts and busy/locked errors are store-level, not this commit's fault.
                    raise
                except CodeContextError as exc:
                    failed_this_sync.append(job_id)
                    _record_diff_job_failure(store, conn, job_id, commit_oid, exc.code)
                except Exception as exc:
                    failed_this_sync.append(job_id)
                    # Only the type name: exception text may carry paths or secrets.
                    _record_diff_job_failure(
                        store, conn, job_id, commit_oid, "INTERNAL", detail=type(exc).__name__
                    )


def _record_diff_job_failure(
    store: Store,
    conn: sqlite3.Connection,
    job_id: str,
    commit_oid: str,
    error_code: str,
    *,
    detail: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE jobs SET attempts = attempts + 1, last_error_code = ?, updated_at_us = ?,
            state = CASE WHEN attempts + 1 >= ? THEN 'retryable_error' ELSE 'pending' END
        WHERE id = ?
        """,
        (error_code, now_us(), MAX_DIFF_JOB_ATTEMPTS, job_id),
    )
    state = conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()["state"]
    if state == "retryable_error":
        reason = f"{error_code}: {detail}" if detail else error_code
        Store.upsert_gap(
            conn,
            repository_id=store.config.repository_id,
            code="IO_ERROR",
            scope=f"git:commit:{commit_oid}",
            message=f"Diffing commit {commit_oid[:12]} failed {MAX_DIFF_JOB_ATTEMPTS} times ({reason}).",
            retryable=True,
        )


def _execute_sync(
    store: Store,
    mirror: GitMirror,
    request: SyncRequest,
    *,
    progress: dict[str, Any],
    github_transport: httpx.BaseTransport | None = None,
    github_token: str | None = None,
    gitlab_token: str | None = None,
) -> SyncReport:
    """Execute bounded, resumable synchronization for `request` without calling any model.
    Priority order per 02-ingestion-and-lineage §1:
    1. Validate identity, dependencies, credentials when needed, and requested revision.
    2. Import objects needed for requested commit; prepare file map, ancestry, and active syntax (prioritizing focus).
    3. For a focus target, find up to 100 relevant historical changes and collect up to 25 related GitHub episodes.
    4. Publish useful collected evidence and pending-enrichment status.
    5. Continue current-snapshot syntax, remaining Git history, and GitHub enumeration with 10:1 focus fairness.
    """
    started_us = now_us()
    start_mono = time.monotonic()
    deadline_mono = start_mono + float(request.max_seconds)
    failed_diff_jobs: list[str] = []

    with store.connect_read() as r_conn:
        meta = r_conn.execute(
            "SELECT repository_id, active_revision FROM store_meta WHERE singleton = 1"
        ).fetchone()
        assert meta is not None
        repo_id = str(meta["repository_id"])
        commits_before = int(r_conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0])

    active_rev = mirror.prepare_snapshot(
        request.revision,
        include_workspace=request.revision == "HEAD",
        include_untracked=request.include_untracked,
    )

    with store.connect_read() as r_conn:
        commits_after = int(r_conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0])
    commits_added = max(0, commits_after - commits_before)

    # Resolve focus path if supplied
    focus_path_bytes: bytes | None = None
    if isinstance(request.focus, PathTarget):
        focus_path_bytes = request.focus.raw_bytes()
    elif isinstance(request.focus, SymbolTarget):
        with store.connect_read() as r_conn:
            s_path = r_conn.execute(
                """
                SELECT sf.path FROM snapshot_files sf
                JOIN symbols s ON s.extraction_id = sf.extraction_id
                WHERE s.id = ? ORDER BY sf.path ASC LIMIT 1
                """,
                (request.focus.id,),
            ).fetchone()
            if s_path:
                focus_path_bytes = bytes(s_path["path"])

    # The focus file goes first so a short budget still covers the code the caller asked about.
    blobs_parsed, cache_hits = index_active_snapshot_files(
        store,
        mirror,
        focus_path=focus_path_bytes,
    )

    progress["structure_complete"] = True
    with store.write_transaction() as conn:
        Store.resolve_gaps_in_scope(conn, scope="sync:deadline")
        if request.reconcile:
            # Reconcile is the explicit retry for diff jobs that used up their attempts.
            conn.execute(
                "UPDATE jobs SET state = 'pending', attempts = 0, updated_at_us = ? "
                "WHERE kind = 'git_commit_diff' AND state = 'retryable_error'",
                (now_us(),),
            )
    check_deadline()
    requests_attempted = 0
    sources_added = 0
    versions_added = 0
    stop_reason: str = "scope_complete"

    if request.ingest_docs:
        from codecontext.docs_ingest import ingest_repository_docs

        d_src, d_ver = ingest_repository_docs(store, mirror, active_revision=active_rev)
        sources_added += d_src
        versions_added += d_ver

    focus_commits: list[str] = []
    if focus_path_bytes is not None:
        focus_commits = index_focused_path_history(
            store,
            mirror,
            active_revision=active_rev,
            focus_path=focus_path_bytes,
            max_changes=100,
        )
    else:
        with store.write_transaction(bump_generation=True) as conn:
            Store.enqueue_pending_diff_jobs(conn)

    if focus_path_bytes is None and not _drain_commit_diff_jobs(store, mirror, deadline_mono, failed_diff_jobs):
        stop_reason = "time_budget"

    gitlab_info: tuple[str, str] | None = None
    if not request.offline and not store.config.github_repository:
        if store.config.gitlab_repository:
            gitlab_info = (
                store.config.gitlab_base_url or "https://gitlab.com",
                store.config.gitlab_repository,
            )
        elif store.config.repo_path:
            gitlab_info = detect_gitlab_repository_from_git(Path(store.config.repo_path))

    if gitlab_info is not None and stop_reason == "scope_complete":
        gl_base, gl_proj = gitlab_info
        rem_reqs = request.max_requests - requests_attempted
        if rem_reqs > 0 and time.monotonic() < deadline_mono:
            with GitLabClient(base_url=gl_base, token=gitlab_token, transport=github_transport) as gl_client:
                progress["http_client"] = gl_client
                used, s_add, v_add, gl_done = collect_gitlab_merge_requests_and_issues(
                    store,
                    gl_client,
                    project_path=gl_proj,
                    remaining_requests=rem_reqs,
                )
                requests_attempted += used
                sources_added += s_add
                versions_added += v_add
                if not gl_done:
                    stop_reason = "request_budget"
    elif store.config.gitlab_repository and request.offline:
        with store.write_transaction(bump_generation=True) as conn:
            Store.upsert_gap(
                conn,
                repository_id=repo_id,
                code="GITHUB_BACKFILL_PENDING",
                scope=f"gitlab:{store.config.gitlab_repository}",
                message="Sync ran with offline=True; remote GitLab evidence collection remains pending.",
                retryable=True,
            )
        if stop_reason == "scope_complete":
            stop_reason = "offline"

    if store.config.github_repository:
        if request.offline:
            with store.write_transaction(bump_generation=True) as conn:
                Store.upsert_gap(
                    conn,
                    repository_id=repo_id,
                    code="GITHUB_BACKFILL_PENDING",
                    scope=f"github:{store.config.github_repository}",
                    message="Sync ran with offline=True; remote GitHub evidence collection remains pending.",
                    retryable=True,
                )
            if stop_reason == "scope_complete":
                stop_reason = "offline"
        elif stop_reason == "scope_complete":
            rem_reqs = request.max_requests - requests_attempted
            if rem_reqs > 0 and time.monotonic() < deadline_mono:
                with GitHubClient(token=github_token, transport=github_transport) as gh_client:
                    progress["http_client"] = gh_client

                    def _apply(done: bool) -> None:
                        nonlocal stop_reason
                        if not done:
                            if requests_attempted >= request.max_requests:
                                stop_reason = "request_budget"
                            elif time.monotonic() >= deadline_mono:
                                stop_reason = "time_budget"

                    if focus_path_bytes is not None and not request.reconcile:
                        used, s_add, v_add, gh_done = collect_focused_github_evidence(
                            store,
                            gh_client,
                            commit_oids=focus_commits,
                            remaining_requests=rem_reqs,
                            deadline_monotonic=deadline_mono,
                        )
                        requests_attempted += used
                        sources_added += s_add
                        versions_added += v_add
                        _apply(gh_done)
                    elif request.reconcile:
                        used2, s_add2, v_add2, gh_done2 = collect_full_and_incremental_github(
                            store,
                            gh_client,
                            reconcile=True,
                            remaining_requests=rem_reqs,
                            deadline_monotonic=deadline_mono,
                            use_graphql=request.use_graphql,
                        )
                        requests_attempted += used2
                        sources_added += s_add2
                        versions_added += v_add2
                        _apply(gh_done2)
                    else:
                        # Run focused commit PRs first if any, then full /issues + incremental streams
                        target_commits_for_gh = focus_commits
                        if not target_commits_for_gh:
                            with store.connect_read() as r_conn:
                                c_rows = r_conn.execute(
                                    "SELECT commit_oid FROM snapshot_commits LIMIT 25"
                                ).fetchall()
                                target_commits_for_gh = [str(r["commit_oid"]) for r in c_rows]

                        used1, s_add1, v_add1, gh_done1 = collect_focused_github_evidence(
                            store,
                            gh_client,
                            commit_oids=target_commits_for_gh,
                            remaining_requests=rem_reqs,
                            deadline_monotonic=deadline_mono,
                        )
                        requests_attempted += used1
                        sources_added += s_add1
                        versions_added += v_add1

                        rem_reqs2 = request.max_requests - requests_attempted
                        if gh_done1 and rem_reqs2 > 0 and time.monotonic() < deadline_mono:
                            used2, s_add2, v_add2, gh_done2 = collect_full_and_incremental_github(
                                store,
                                gh_client,
                                reconcile=False,
                                remaining_requests=rem_reqs2,
                                deadline_monotonic=deadline_mono,
                                use_graphql=request.use_graphql,
                            )
                            requests_attempted += used2
                            sources_added += s_add2
                            versions_added += v_add2
                            _apply(gh_done2)
                        else:
                            _apply(gh_done1)

    # History left over after remote collection uses whatever budget remains.
    if stop_reason == "scope_complete" and not _drain_commit_diff_jobs(store, mirror, deadline_mono, failed_diff_jobs):
        stop_reason = "time_budget"

    # Check if all git commit diffs are complete and resolve GIT_HISTORY_PENDING if so
    with store.write_transaction(bump_generation=True) as conn:
        rem_git = int(
            conn.execute("SELECT COUNT(*) FROM commits WHERE diff_state != 'complete'").fetchone()[0]
        )
        if rem_git == 0:
            Store.resolve_gaps_in_scope(conn, scope="git:history", code="GIT_HISTORY_PENDING")
        else:
            Store.upsert_gap(
                conn,
                repository_id=repo_id,
                code="GIT_HISTORY_PENDING",
                scope="git:history",
                message=f"{rem_git} commit(s) remaining in Git history diff queue.",
                retryable=True,
            )

        q_counts = Store.get_queue_counts(conn)
        cov, gaps, _ = Store.build_coverage_and_gaps(conn, max_detailed_gaps=20)
        meta_after = conn.execute(
            "SELECT generation, access_epoch FROM store_meta WHERE singleton = 1"
        ).fetchone()
        gen_after = int(meta_after["generation"])
        epoch_after = int(meta_after["access_epoch"])

    finished_us = now_us()
    elapsed_ms = max(0, (finished_us - started_us) // 1000)
    is_partial = (
        stop_reason != "scope_complete"
        or q_counts.pending > 0
        or cov.git.state != "complete"
        or cov.structure.state != "complete"
        or (store.config.github_repository is not None and cov.github.state != "complete")
    )
    report_state: Any = "partial" if is_partial else "complete"

    return SyncReport(
        repository_id=repo_id,
        revision=active_rev,
        generation=gen_after,
        access_epoch=epoch_after,
        coverage=cov,
        gaps=gaps,
        state=report_state,
        started_at=us_to_iso_z(started_us),
        finished_at=us_to_iso_z(finished_us),
        elapsed_ms=elapsed_ms,
        requests_attempted=requests_attempted,
        commits_added=commits_added,
        blobs_parsed=blobs_parsed,
        sources_added=sources_added,
        versions_added=versions_added,
        cache_hits=cache_hits,
        jobs=q_counts,
        stop_reason=stop_reason,  # type: ignore[arg-type]
    )
