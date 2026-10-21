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
from codecontext.lineage import index_commit_changes_and_hunks, index_focused_path_history
from codecontext.models import PathTarget, SymbolTarget, SyncReport, SyncRequest
from codecontext.store import Store, now_us
from codecontext.syntax import index_active_snapshot_files


def execute_sync(
    store: Store,
    mirror: GitMirror,
    request: SyncRequest,
    *,
    github_transport: httpx.BaseTransport | None = None,
    github_token: str | None = None,
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
            return _execute_sync(store, mirror, request, progress=progress, github_transport=github_transport, github_token=github_token)
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
                      commits_added=commits_added, blobs_parsed=blobs_parsed,
                      sources_added=sources_added, versions_added=versions_added, cache_hits=0, jobs=jobs, stop_reason="time_budget")


def _execute_sync(
    store: Store,
    mirror: GitMirror,
    request: SyncRequest,
    *,
    progress: dict[str, Any],
    github_transport: httpx.BaseTransport | None = None,
    github_token: str | None = None,
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

    with store.connect_read() as r_conn:
        meta = r_conn.execute(
            "SELECT repository_id, active_revision FROM store_meta WHERE singleton = 1"
        ).fetchone()
        assert meta is not None
        repo_id = str(meta["repository_id"])
        commits_before = int(r_conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0])

    # 1. Prepare active snapshot
    active_rev = mirror.prepare_snapshot(request.revision, include_workspace=request.revision == "HEAD")

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

    # 2. Extract active snapshot syntax (focus first)
    blobs_parsed, cache_hits = index_active_snapshot_files(
        store,
        mirror,
        focus_path=focus_path_bytes,
    )

    progress["structure_complete"] = True
    with store.write_transaction() as conn:
        Store.resolve_gaps_in_scope(conn, scope="sync:deadline")
    check_deadline()
    requests_attempted = 0
    sources_added = 0
    versions_added = 0
    stop_reason: str = "scope_complete"

    # 3. Process focus history (or all reachable commits if no focus) within time budget
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
        # Enqueue pending commits in jobs queue
        with store.write_transaction(bump_generation=True) as conn:
            pending_c = conn.execute(
                "SELECT oid FROM commits WHERE diff_state != 'complete' ORDER BY first_seen_at_us ASC, oid ASC"
            ).fetchall()
            for r in pending_c:
                Store.enqueue_job(conn, kind="git_commit_diff", owner_id=str(r["oid"]), priority=50)

    # 4. Process pending git_commit_diff jobs first when no focus was supplied so commits have changes before GitHub collection
    while stop_reason == "scope_complete":
        if time.monotonic() >= deadline_mono:
            stop_reason = "time_budget"
            break
        with store.connect_read() as r_conn:
            next_job = r_conn.execute(
                """
                SELECT id, kind, owner_id
                FROM jobs
                WHERE state = 'pending' AND kind = 'git_commit_diff'
                ORDER BY priority ASC, created_at_us ASC, id ASC
                LIMIT 1
                """
            ).fetchone()
        if not next_job:
            break

        job_id = str(next_job["id"])
        c_oid = str(next_job["owner_id"])
        with store.write_transaction(bump_generation=False) as conn:
            conn.execute(
                "UPDATE jobs SET state = 'running', updated_at_us = ? WHERE id = ?",
                (now_us(), job_id),
            )
        index_commit_changes_and_hunks(store, mirror, c_oid)
        with store.write_transaction(bump_generation=True) as conn:
            conn.execute(
                "UPDATE jobs SET state = 'complete', updated_at_us = ? WHERE id = ?",
                (now_us(), job_id),
            )

    # 5. If online and GitHub or GitLab repository is configured, collect remote evidence
    gitlab_info = (
        detect_gitlab_repository_from_git(Path(store.config.repo_path))
        if not request.offline and store.config.repo_path and not store.config.github_repository
        else None
    )

    if gitlab_info is not None and not request.offline and stop_reason == "scope_complete":
        gl_base, gl_proj = gitlab_info
        rem_reqs = request.max_requests - requests_attempted
        if rem_reqs > 0 and time.monotonic() < deadline_mono:
            with GitLabClient(base_url=gl_base, token=github_token, transport=github_transport) as gl_client:
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
                        if not gh_done:
                            if requests_attempted >= request.max_requests:
                                stop_reason = "request_budget"
                            elif time.monotonic() >= deadline_mono:
                                stop_reason = "time_budget"
                    else:
                        if request.reconcile:
                            used2, s_add2, v_add2, gh_done2 = collect_full_and_incremental_github(
                                store,
                                gh_client,
                                reconcile=True,
                                remaining_requests=rem_reqs,
                                deadline_monotonic=deadline_mono,
                            )
                            requests_attempted += used2
                            sources_added += s_add2
                            versions_added += v_add2
                            if not gh_done2:
                                if requests_attempted >= request.max_requests:
                                    stop_reason = "request_budget"
                                elif time.monotonic() >= deadline_mono:
                                    stop_reason = "time_budget"
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
                                )
                                requests_attempted += used2
                                sources_added += s_add2
                                versions_added += v_add2
                                if not gh_done2:
                                    if requests_attempted >= request.max_requests:
                                        stop_reason = "request_budget"
                                    elif time.monotonic() >= deadline_mono:
                                        stop_reason = "time_budget"
                            elif not gh_done1:
                                if requests_attempted >= request.max_requests:
                                    stop_reason = "request_budget"
                                elif time.monotonic() >= deadline_mono:
                                    stop_reason = "time_budget"

    # 6. Process any remaining pending git_commit_diff jobs while time budget allows
    while stop_reason == "scope_complete":
        if time.monotonic() >= deadline_mono:
            stop_reason = "time_budget"
            break
        with store.connect_read() as r_conn:
            next_job = r_conn.execute(
                """
                SELECT id, kind, owner_id
                FROM jobs
                WHERE state = 'pending' AND kind = 'git_commit_diff'
                ORDER BY priority ASC, created_at_us ASC, id ASC
                LIMIT 1
                """
            ).fetchone()
        if not next_job:
            break

        job_id = str(next_job["id"])
        c_oid = str(next_job["owner_id"])
        with store.write_transaction(bump_generation=False) as conn:
            conn.execute(
                "UPDATE jobs SET state = 'running', updated_at_us = ? WHERE id = ?",
                (now_us(), job_id),
            )
        index_commit_changes_and_hunks(store, mirror, c_oid)
        with store.write_transaction(bump_generation=True) as conn:
            conn.execute(
                "UPDATE jobs SET state = 'complete', updated_at_us = ? WHERE id = ?",
                (now_us(), job_id),
            )

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
