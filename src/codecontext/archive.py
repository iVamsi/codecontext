import json
import os
import shutil
import sqlite3
import tempfile
from collections import deque
from pathlib import Path

from codecontext.errors import CodeContextError
from codecontext.identity import sha256_bytes
from codecontext.models import ExportReport, ExportRequest, PurgeReport, PurgeRequest
from codecontext.store import Store, now_us


def execute_purge(store: Store, request: PurgeRequest) -> PurgeReport:
    """Execute either a source-specific purge (deleting transitive reverse-dependency closure,
    generated bodies, FTS rows, relations, evidence, and source versions, leaving only a non-content
    source tombstone) or an entire-store purge.
    """
    if request.entire_store:
        with store.acquire_writer_lock():
            with store.connect_read() as conn:
                meta = conn.execute(
                    "SELECT repository_id, generation, access_epoch FROM store_meta WHERE singleton = 1"
                ).fetchone()
                if not meta or str(meta["repository_id"]) != store.config.repository_id:
                    raise CodeContextError(
                        code="REPOSITORY_MISMATCH",
                        message="Store ownership marker validation failed before entire-store purge.",
                        retryable=False,
                    )
                repo_id = str(meta["repository_id"])
                gen = int(meta["generation"]) + 1
                epoch = int(meta["access_epoch"]) + 1
                ver_cnt = int(conn.execute("SELECT COUNT(*) FROM source_versions").fetchone()[0])
                cl_cnt = int(conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0])
                sum_cnt = int(conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0])

            # Block reads and bump access_epoch before removing owned files
            w_conn = store._open_raw_connection(for_write=True)
            try:
                w_conn.execute("BEGIN IMMEDIATE")
                w_conn.execute(
                    "UPDATE store_meta SET access_state = 'blocked', generation = ?, access_epoch = ? WHERE singleton = 1",
                    (gen, epoch),
                )
                w_conn.execute("COMMIT")
            finally:
                w_conn.close()

            store.close()

            # Remove owned store files (`memory.sqlite3*`, `repository.git`, `store.json`, `tmp`)
            for item_name in (
                "memory.sqlite3",
                "memory.sqlite3-wal",
                "memory.sqlite3-shm",
                "store.json",
                "manifest.json",
                "OVERVIEW.md",
                ".gitignore",
            ):
                p = store.store_path / item_name
                if p.exists() or p.is_symlink():
                    p.unlink()

            for dir_name in ("repository.git", "tmp"):
                dp = store.store_path / dir_name
                if dp.exists():
                    shutil.rmtree(dp, ignore_errors=True)

            # Remove the store directory itself if now empty
            try:
                if store.store_path.exists() and not any(store.store_path.iterdir()):
                    store.store_path.rmdir()
            except OSError:
                pass

            return PurgeReport(
                repository_id=repo_id,
                mode="entire_store",
                source_id=None,
                generation=gen,
                access_epoch=epoch,
                removed_source_versions=ver_cnt,
                removed_claims=cl_cnt,
                removed_summaries=sum_cnt,
                cleanup_pending=False,
                closed=True,
            )

    # Source-specific purge
    assert request.source_id is not None
    src_id = request.source_id

    with store.write_transaction(bump_generation=True, bump_access_epoch=True) as conn:
        meta = conn.execute(
            "SELECT repository_id, generation, access_epoch FROM store_meta WHERE singleton = 1"
        ).fetchone()
        assert meta is not None
        repo_id = str(meta["repository_id"])

        src_row = conn.execute(
            "SELECT id, availability FROM sources WHERE id = ?", (src_id,)
        ).fetchone()
        if not src_row:
            raise CodeContextError(
                code="RECORD_NOT_FOUND",
                message=f"Source {src_id} not found.",
                retryable=False,
                details={"id": src_id},
            )

        # Collect source and any child sources owned by it
        affected_sources: list[str] = [src_id]
        child_rows = conn.execute(
            "SELECT id FROM sources WHERE parent_source_id = ?", (src_id,)
        ).fetchall()
        for cr in child_rows:
            affected_sources.append(str(cr["id"]))

        # Collect all source_versions and evidence rows for affected sources
        ver_ids: list[str] = []
        ev_ids: list[str] = []
        for s_id in affected_sources:
            v_rows = conn.execute(
                "SELECT id FROM source_versions WHERE source_id = ?", (s_id,)
            ).fetchall()
            for vr in v_rows:
                vid = str(vr["id"])
                ver_ids.append(vid)
                e_rows = conn.execute(
                    "SELECT id FROM evidence WHERE source_version_id = ?", (vid,)
                ).fetchall()
                for er in e_rows:
                    ev_ids.append(str(er["id"]))

        # Compute full reverse dependency closure (claims and summaries influenced by any affected version/evidence)
        queue: deque[tuple[str, str]] = deque()
        for vid in ver_ids:
            queue.append(("source_version", vid))
        for eid in ev_ids:
            queue.append(("evidence", eid))

        visited_dep: set[tuple[str, str]] = set(queue)
        claims_to_remove: set[str] = set()
        summaries_to_remove: set[str] = set()

        while queue:
            in_kind, in_id = queue.popleft()
            dep_rows = conn.execute(
                "SELECT dependent_kind, dependent_id FROM dependencies WHERE input_kind = ? AND input_id = ?",
                (in_kind, in_id),
            ).fetchall()
            for dr in dep_rows:
                dk = str(dr["dependent_kind"])
                did = str(dr["dependent_id"])
                if dk == "claim":
                    claims_to_remove.add(did)
                elif dk == "summary":
                    summaries_to_remove.add(did)
                pair = (dk, did)
                if pair not in visited_dep:
                    visited_dep.add(pair)
                    queue.append(pair)

        # Also include any claims directly linked via claim_evidence to affected evidence
        for eid in ev_ids:
            ce_rows = conn.execute(
                "SELECT claim_id FROM claim_evidence WHERE evidence_id = ?", (eid,)
            ).fetchall()
            for cer in ce_rows:
                cid = str(cer["claim_id"])
                if cid not in claims_to_remove:
                    claims_to_remove.add(cid)
                    # Check summaries depending on this claim
                    s_deps = conn.execute(
                        "SELECT dependent_id FROM dependencies WHERE input_kind = 'claim' AND input_id = ? AND dependent_kind = 'summary'",
                        (cid,),
                    ).fetchall()
                    for sd in s_deps:
                        summaries_to_remove.add(str(sd["dependent_id"]))

        # Delete FTS search documents for purged versions, claims, and summaries
        for vid in ver_ids:
            conn.execute(
                "DELETE FROM search_documents WHERE record_kind = 'history' AND record_id = ?",
                (vid,),
            )
        for cid in claims_to_remove:
            conn.execute(
                "DELETE FROM search_documents WHERE record_kind = 'decision' AND record_id = ?",
                (cid,),
            )
            conn.execute("DELETE FROM relations WHERE from_kind = 'claim' AND from_id = ?", (cid,))
            conn.execute("DELETE FROM relations WHERE to_kind = 'claim' AND to_id = ?", (cid,))
            conn.execute("DELETE FROM dependencies WHERE dependent_kind = 'claim' AND dependent_id = ?", (cid,))
            conn.execute("DELETE FROM dependencies WHERE input_kind = 'claim' AND input_id = ?", (cid,))
            conn.execute("DELETE FROM claims WHERE id = ?", (cid,))

        for sum_id in summaries_to_remove:
            conn.execute(
                "DELETE FROM search_documents WHERE record_kind = 'summary' AND record_id = ?",
                (sum_id,),
            )
            conn.execute("DELETE FROM dependencies WHERE dependent_kind = 'summary' AND dependent_id = ?", (sum_id,))
            conn.execute("DELETE FROM dependencies WHERE input_kind = 'summary' AND input_id = ?", (sum_id,))
            conn.execute("DELETE FROM summaries WHERE id = ?", (sum_id,))

        # Delete relations referencing affected evidence or sources BEFORE deleting evidence (foreign key does not cascade)
        for eid in ev_ids:
            conn.execute("DELETE FROM relations WHERE evidence_id = ?", (eid,))
            conn.execute("DELETE FROM relations WHERE from_kind = 'evidence' AND from_id = ?", (eid,))
            conn.execute("DELETE FROM relations WHERE to_kind = 'evidence' AND to_id = ?", (eid,))
            conn.execute("DELETE FROM dependencies WHERE input_kind = 'evidence' AND input_id = ?", (eid,))
            conn.execute("DELETE FROM evidence WHERE id = ?", (eid,))

        for vid in ver_ids:
            conn.execute(
                "DELETE FROM dependencies WHERE input_kind = 'source_version' AND input_id = ?",
                (vid,),
            )

        # Reset or remove episodes referencing affected sources
        for s_id in affected_sources:
            conn.execute("DELETE FROM relations WHERE from_kind = 'source' AND from_id = ?", (s_id,))
            conn.execute("DELETE FROM relations WHERE to_kind = 'source' AND to_id = ?", (s_id,))
            conn.execute("DELETE FROM coverage_gaps WHERE record_id = ?", (s_id,))
            ep_rows = conn.execute(
                "SELECT id FROM episodes WHERE primary_source_id = ?", (s_id,)
            ).fetchall()
            for epr in ep_rows:
                epid = str(epr["id"])
                conn.execute("DELETE FROM episode_segments WHERE episode_id = ?", (epid,))
                conn.execute("DELETE FROM claims WHERE episode_id = ?", (epid,))
                conn.execute("DELETE FROM episodes WHERE id = ?", (epid,))

            # Leave only a non-content source tombstone (`availability='redacted'`, `current_version_id=NULL`)
            conn.execute(
                "UPDATE sources SET current_version_id = NULL, availability = 'redacted', last_seen_at_us = ? WHERE id = ?",
                (now_us(), s_id),
            )
            conn.execute("DELETE FROM source_versions WHERE source_id = ?", (s_id,))

    # Best-effort WAL checkpoint and VACUUM when no reader is active
    cleanup_pending = False
    try:
        with store.acquire_writer_lock(timeout_seconds=2.0):
            v_conn = store._open_raw_connection(for_write=True)
            try:
                v_conn.execute("PRAGMA secure_delete=ON")
                v_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                v_conn.execute("VACUUM")
            finally:
                v_conn.close()
    except Exception:
        cleanup_pending = True
        with store.write_transaction(bump_generation=False) as conn:
            Store.upsert_gap(
                conn,
                repository_id=repo_id,
                code="CLEANUP_PENDING",
                scope=f"purge:{src_id}",
                message="Logical purge completed; physical SQLite compaction is pending reader completion.",
                retryable=True,
            )

    with store.connect_read() as r_conn:
        meta_after = r_conn.execute(
            "SELECT generation, access_epoch FROM store_meta WHERE singleton = 1"
        ).fetchone()
        gen_after = int(meta_after["generation"])
        epoch_after = int(meta_after["access_epoch"])

    return PurgeReport(
        repository_id=repo_id,
        mode="source",
        source_id=src_id,
        generation=gen_after,
        access_epoch=epoch_after,
        removed_source_versions=len(ver_ids),
        removed_claims=len(claims_to_remove),
        removed_summaries=len(summaries_to_remove),
        cleanup_pending=cleanup_pending,
        closed=False,
    )


def sanitized_bare_git_config(obj_fmt: str) -> str:
    fmt_version = "1" if obj_fmt == "sha256" else "0"
    ext_section = "\n[extensions]\n\tobjectFormat = sha256\n" if obj_fmt == "sha256" else ""
    return (
        f"[core]\n\trepositoryformatversion = {fmt_version}\n\tfilemode = true\n\tbare = true\n"
        f"[gc]\n\tauto = 0\n{ext_section}"
    )


def execute_export(store: Store, request: ExportRequest) -> ExportReport:
    """Create a portable, read-only offline archive at `request.destination` (which must not exist).
    Scrubs machine-specific paths and credentials from SQLite (`secure_delete=ON` + `VACUUM`),
    `store.json`, and `repository.git/config`, writes `manifest.json` and `OVERVIEW.md`, and publishes atomically.
    """
    dest = request.destination.expanduser()
    if dest.exists() or dest.is_symlink():
        raise CodeContextError(
            code="INVALID_REQUEST",
            message=f"Export destination already exists: {dest}",
            retryable=False,
        )
    dest_parent = dest.parent.expanduser().resolve()
    dest_parent.mkdir(parents=True, exist_ok=True)
    resolved_dest = dest_parent / dest.name

    # Reject destinations inside the owned store or indexed source checkout
    if resolved_dest == store.store_path or store.store_path in resolved_dest.parents:
        raise CodeContextError(
            code="INVALID_REQUEST",
            message="Export destination must not be inside the live store directory.",
            retryable=False,
        )
    if store.config.repo_path:
        src_repo_real = Path(store.config.repo_path).resolve()
        if resolved_dest == src_repo_real or src_repo_real in resolved_dest.parents:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message="Export destination must not be inside the indexed source repository.",
                retryable=False,
            )

    with store.acquire_writer_lock(timeout_seconds=5.0, allow_read_only=True):
        # Create temporary sibling directory on the same filesystem
        tmp_export_dir = Path(
            tempfile.mkdtemp(prefix=f".{dest.name}.tmp_", dir=str(dest_parent))
        )
        try:
            (tmp_export_dir / "tmp").mkdir(parents=True, exist_ok=True)
            dest_db = tmp_export_dir / "memory.sqlite3"

            # 1. Consistent SQLite backup using sqlite3 backup API
            with store.connect_read() as src_conn:
                meta = src_conn.execute(
                    "SELECT repository_id, active_revision, generation, access_epoch, object_format "
                    "FROM store_meta WHERE singleton = 1"
                ).fetchone()
                if not meta or str(meta["repository_id"]) != store.config.repository_id:
                    raise CodeContextError(
                        code="REPOSITORY_MISMATCH",
                        message="Store identity mismatch before export.",
                        retryable=False,
                    )
                repo_id = str(meta["repository_id"])
                active_rev = meta["active_revision"]
                gen = int(meta["generation"])
                epoch = int(meta["access_epoch"])
                obj_fmt = str(meta["object_format"])

                b_conn = sqlite3.connect(str(dest_db))
                try:
                    src_conn.backup(b_conn)
                finally:
                    b_conn.close()

            # 2. Scrub machine bindings in destination SQLite with secure_delete=ON, then checkpoint and VACUUM
            scrub_conn = sqlite3.connect(str(dest_db), isolation_level=None)
            try:
                scrub_conn.execute("PRAGMA journal_mode=DELETE")
                scrub_conn.execute("PRAGMA secure_delete=ON")
                scrub_conn.execute("BEGIN IMMEDIATE")
                scrub_conn.execute(
                    """
                    UPDATE store_meta
                    SET local_repository_path = NULL,
                        common_dir_realpath = NULL,
                        common_dir_device = NULL,
                        common_dir_inode = NULL,
                        principal_id = NULL
                    WHERE singleton = 1
                    """
                )
                if not request.include_git_objects:
                    Store.upsert_gap(
                        scrub_conn,
                        repository_id=repo_id,
                        code="EXPORT_GIT_OBJECTS_OMITTED",
                        scope="export:git_objects",
                        message="Archive was exported with metadata-only (include_git_objects=False); offline code expansion is unavailable.",
                        retryable=False,
                    )
                scrub_conn.execute("COMMIT")
                scrub_conn.execute("VACUUM")
            finally:
                scrub_conn.close()

            # Remove any leftover -wal / -shm files
            for suffix in ("-wal", "-shm"):
                wf = Path(str(dest_db) + suffix)
                if wf.exists():
                    wf.unlink()

            # 3. Copy owned Git object store (if include_git_objects=True) with minimal sanitized config
            warnings: list[str] = [
                "Exported copies moved outside this store cannot be recalled by subsequent purge operations.",
                "Private repository content and secrets committed to Git history remain source evidence in full exports.",
            ]
            if request.include_git_objects and store.git_dir.exists():
                dest_git = tmp_export_dir / "repository.git"
                dest_git.mkdir(parents=True, exist_ok=True)
                for sub_name in ("objects", "refs", "HEAD", "packed-refs"):
                    src_sub = store.git_dir / sub_name
                    dst_sub = dest_git / sub_name
                    if src_sub.is_dir():
                        shutil.copytree(src_sub, dst_sub, symlinks=False)
                        # Remove any info/alternates if present
                        alt_file = dst_sub / "info" / "alternates"
                        if alt_file.exists():
                            alt_file.unlink()
                    elif src_sub.is_file():
                        shutil.copy2(src_sub, dst_sub)

                (dest_git / "config").write_text(sanitized_bare_git_config(obj_fmt), encoding="utf-8")
            elif not request.include_git_objects:
                warnings.append(
                    "Git objects were omitted (--metadata-only); offline source/code expansion is not guaranteed."
                )

            # 4. Write sanitized read-only `store.json` (`read_only_archive=True`, `repo_path=None`)
            export_store_cfg = {
                "allowed_issue_repositories": store.config.allowed_issue_repositories,
                "format_version": 1,
                "github_repository": store.config.github_repository,
                "read_only_archive": True,
                "repo_path": None,
                "repository_id": repo_id,
            }
            (tmp_export_dir / "store.json").write_text(
                json.dumps(export_store_cfg, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            # 5. Write `OVERVIEW.md` from published summary or deterministic structural overview
            overview_md = _render_export_overview_md(dest_db, repo_id, active_rev, gen)
            (tmp_export_dir / "OVERVIEW.md").write_text(overview_md, encoding="utf-8")

            # 6. Compute SHA-256 hashes of all exported files and write `manifest.json`
            file_hashes: dict[str, str] = {}
            total_bytes = 0
            file_count = 0
            for root_p, _dirs, files in os.walk(tmp_export_dir):
                for fname in sorted(files):
                    fpath = Path(root_p) / fname
                    rel = fpath.relative_to(tmp_export_dir).as_posix()
                    if rel == "manifest.json":
                        continue
                    fbytes = fpath.read_bytes()
                    file_hashes[rel] = sha256_bytes(fbytes)
                    total_bytes += len(fbytes)
                    file_count += 1

            manifest_payload = {
                "access_epoch": epoch,
                "active_revision": active_rev,
                "file_count": file_count,
                "files": file_hashes,
                "generation": gen,
                "include_git_objects": request.include_git_objects,
                "limitations": warnings,
                "object_format": obj_fmt,
                "repository_id": repo_id,
                "schema_version": 1,
            }
            manifest_bytes = (json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
            manifest_sha = sha256_bytes(manifest_bytes)
            (tmp_export_dir / "manifest.json").write_bytes(manifest_bytes)
            total_bytes += len(manifest_bytes)
            file_count += 1

            # 7. Atomically publish without replacing if destination was created concurrently
            if resolved_dest.exists() or resolved_dest.is_symlink():
                raise CodeContextError(
                    code="INVALID_REQUEST",
                    message=f"Export destination {resolved_dest} was created concurrently; refusing to overwrite.",
                    retryable=False,
                )
            os.rename(tmp_export_dir, resolved_dest)

        except BaseException:
            if tmp_export_dir.exists():
                shutil.rmtree(tmp_export_dir, ignore_errors=True)
            raise

    return ExportReport(
        repository_id=repo_id,
        revision=active_rev,
        generation=gen,
        access_epoch=epoch,
        destination=str(resolved_dest),
        include_git_objects=request.include_git_objects,
        file_count=file_count,
        total_bytes=total_bytes,
        manifest_sha256=manifest_sha,
        warnings=warnings[:20],
    )


def _render_export_overview_md(
    db_path: Path,
    repo_id: str,
    active_rev: str | None,
    generation: int,
) -> str:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        snap_cnt = int(conn.execute("SELECT COUNT(*) FROM snapshot_files").fetchone()[0])
        commit_cnt = int(conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0])
        source_cnt = int(conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0])
        claim_cnt = int(conn.execute("SELECT COUNT(*) FROM claims WHERE state = 'ready'").fetchone()[0])
        gap_rows = conn.execute(
            "SELECT code, scope, message FROM coverage_gaps WHERE active = 1 ORDER BY scope ASC, code ASC LIMIT 20"
        ).fetchall()

        sum_row = (
            conn.execute(
                "SELECT text FROM summaries WHERE revision = ? AND scope_kind = 'repository' AND state = 'ready' LIMIT 1",
                (active_rev,),
            ).fetchone()
            if active_rev
            else None
        )
    finally:
        conn.close()

    lines = [
        "# CodeContext Project Memory Overview",
        "",
        f"- **Repository ID:** `{repo_id}`",
        f"- **Prepared Revision:** `{active_rev or 'none'}`",
        f"- **Generation:** `{generation}`",
        f"- **Snapshot Files:** {snap_cnt}",
        f"- **Indexed Commits:** {commit_cnt}",
        f"- **Discussion Sources:** {source_cnt}",
        f"- **Ready Decision Claims:** {claim_cnt}",
        "",
    ]
    if sum_row and sum_row["text"]:
        lines.extend(["## Summary", "", str(sum_row["text"]), ""])
    else:
        lines.extend(
            [
                "## Structural Overview",
                "",
                f"Deterministic snapshot of {snap_cnt} tracked files and {commit_cnt} commits.",
                "",
            ]
        )

    if gap_rows:
        lines.extend(["## Active Coverage Gaps", ""])
        for gr in gap_rows:
            lines.append(f"- `{gr['code']}` ({gr['scope']}): {gr['message']}")
        lines.append("")

    return "\n".join(lines)
