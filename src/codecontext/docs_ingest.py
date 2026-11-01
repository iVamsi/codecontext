import json
import re
from typing import Any

from codecontext.deadline import check_deadline
from codecontext.git import GitMirror
from codecontext.identity import (
    LineByteIndex,
    canonical_json,
    hash_source_version_payload,
    make_episode_id,
    make_ev_id,
    make_relation_id,
    make_src_id,
    make_ver_id,
    path_bytes_to_display,
    sha256_bytes,
)
from codecontext.store import Store, now_us
from codecontext.syntax import MAX_PARSE_BYTES, chunk_utf8_bytes, normalize_search_text

H1_TITLE_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
BACKTICK_TOKEN_RE = re.compile(r"`([A-Za-z0-9_./:-]{2,120})`")

DESIGN_DOC_EXACT_NAMES = {
    "architecture.md",
    "design.md",
    "decisions.md",
    "docs/architecture.md",
    "docs/design.md",
    "docs/decisions.md",
}


def classify_doc_path(display_path: str) -> str | None:
    """Return `'adr'`, `'design_doc'`, or `None` for a repository-relative POSIX path."""
    norm = display_path.strip("/").lower()
    if not norm.endswith(".md"):
        return None
    if (
        norm.startswith("docs/adr/")
        or norm.startswith("adr/")
        or norm.startswith("docs/adrs/")
        or norm.startswith("adrs/")
        or norm.startswith("docs/decisions/")
        or norm.startswith("decisions/")
        or "/adr/" in norm
        or "/adrs/" in norm
    ):
        return "adr"
    if norm in DESIGN_DOC_EXACT_NAMES or norm.startswith(("doc/architecture/", "docs/architecture/")):
        return "design_doc"
    return None


def _extract_markdown_title(body: str, fallback_path: str) -> str:
    m = H1_TITLE_RE.search(body)
    if m:
        title = m.group(1).strip()
        if title:
            return title[:300]
    stem = fallback_path.rsplit("/", 1)[-1]
    if stem.lower().endswith(".md"):
        stem = stem[:-3]
    return stem or fallback_path


def ingest_repository_docs(
    store: Store,
    mirror: GitMirror,
    *,
    active_revision: str,
) -> tuple[int, int]:
    """Ingest tracked ADRs and architecture/design Markdown documents from the active snapshot
    as `provider='git'` sources, immutable `source_versions`, discussion `evidence`, FTS documents,
    target symbol `relations`, and enrichable `episodes`.
    Returns `(sources_added, versions_added)`.
    """
    with store.connect_read() as r_conn:
        meta = r_conn.execute(
            "SELECT repository_id FROM store_meta WHERE singleton = 1"
        ).fetchone()
        if not meta:
            return (0, 0)
        repo_id = str(meta["repository_id"])
        snap_rows = r_conn.execute(
            "SELECT path, blob_oid, mode FROM snapshot_files ORDER BY path ASC"
        ).fetchall()

    candidates: list[tuple[bytes, str, str | None, str, str]] = []
    active_doc_provider_ids: set[str] = set()
    for sr in snap_rows:
        mode = int(sr["mode"])
        if mode in (0o160000, 0o120000):
            continue
        p_bytes = bytes(sr["path"])
        disp_p, p_b64 = path_bytes_to_display(p_bytes)
        doc_kind = classify_doc_path(disp_p)
        if doc_kind is not None:
            candidates.append((p_bytes, disp_p, p_b64, str(sr["blob_oid"]), doc_kind))
            active_doc_provider_ids.add(disp_p)

    # Check if we already had any git doc sources that might need deletion reconciliation
    with store.connect_read() as r_conn:
        existing_git_docs = r_conn.execute(
            "SELECT id, provider_id, current_version_id, availability FROM sources WHERE provider = 'git' AND repository_provider_id = ?",
            (repo_id,),
        ).fetchall()

    if not candidates and not existing_git_docs:
        return (0, 0)

    loaded_docs: list[tuple[str, str | None, str, str, bytes, str]] = []
    if candidates:
        with mirror.open_batch_session() as batch:
            for _p_bytes, disp_p, p_b64, blob_oid, doc_kind in candidates:
                check_deadline()
                obj = batch.read_object_bounded(blob_oid, max_bytes=MAX_PARSE_BYTES)
                if obj is None or obj[0] != "blob" or obj[2] is None:
                    continue
                raw_bytes = obj[2]
                if b"\x00" in raw_bytes:
                    continue
                try:
                    text = raw_bytes.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    continue
                if not text.strip():
                    continue
                loaded_docs.append((disp_p, p_b64, blob_oid, doc_kind, raw_bytes, text))

    sources_added = 0
    versions_added = 0
    ts = now_us()

    with store.write_transaction(bump_generation=True) as conn:
        # Mark removed git docs as deleted
        for ed in existing_git_docs:
            if str(ed["provider_id"]) not in active_doc_provider_ids and str(ed["availability"]) == "available":
                ed_id = str(ed["id"])
                ed_vid = ed["current_version_id"]
                conn.execute(
                    "UPDATE sources SET availability = 'deleted', last_reconciled_at_us = ? WHERE id = ?",
                    (ts, ed_id),
                )
                if ed_vid:
                    Store.invalidate_dependents_recursive(conn, [("source_version", str(ed_vid))])
                    conn.execute(
                        "DELETE FROM search_documents WHERE record_kind = 'history' AND record_id = ?",
                        (str(ed_vid),),
                    )

        if not loaded_docs:
            return (0, 0)

        # Build lookup of active symbol names/qualified_names for linking doc references
        sym_rows = conn.execute(
            """
            SELECT s.id, s.name, s.qualified_name, sf.path
            FROM snapshot_files sf
            JOIN symbols s ON s.extraction_id = sf.extraction_id
            """
        ).fetchall()
        by_qname: dict[str, list[str]] = {}
        by_name: dict[str, list[str]] = {}
        by_path_syms: dict[str, list[str]] = {}
        for sr in sym_rows:
            sid = str(sr["id"])
            qn = str(sr["qualified_name"])
            nm = str(sr["name"])
            sp_disp, _ = path_bytes_to_display(bytes(sr["path"]))
            by_qname.setdefault(qn, []).append(sid)
            by_name.setdefault(nm, []).append(sid)
            by_path_syms.setdefault(sp_disp, []).append(sid)

        # Look up commit timestamp if active_revision is in `commits`
        rev_c = conn.execute(
            "SELECT authored_at_us, committed_at_us, author_name FROM commits WHERE oid = ?",
            (active_revision,),
        ).fetchone()
        authored_us = int(rev_c["authored_at_us"]) if rev_c and rev_c["authored_at_us"] is not None else None
        updated_us = int(rev_c["committed_at_us"]) if rev_c and rev_c["committed_at_us"] is not None else authored_us
        author_name = str(rev_c["author_name"]) if rev_c and rev_c["author_name"] else None
        author_json_str = (
            canonical_json({"actor_id": None, "display_name": author_name, "provider": "git"})
            if author_name
            else None
        )

        for disp_p, p_b64, blob_oid, doc_kind, raw_bytes, body_text in loaded_docs:
            check_deadline()
            title = _extract_markdown_title(body_text, disp_p)
            src_id = make_src_id(
                provider="git",
                repository_provider_id=repo_id,
                kind=doc_kind,
                provider_id=disp_p,
            )
            anchors: dict[str, Any] = {
                "blob_oid": blob_oid,
                "path": disp_p,
            }
            content_hash = hash_source_version_payload(
                title=title,
                body=body_text,
                author_id=None,
                author_name=author_name,
                anchors=anchors,
                authored_at_us=None,
                updated_at_us=None,
                lifecycle_fields={"kind": doc_kind},
            )
            ver_id = make_ver_id(source_id=src_id, content_sha256=content_hash)

            existing_src = conn.execute(
                "SELECT id, current_version_id, availability FROM sources WHERE id = ?",
                (src_id,),
            ).fetchone()
            src_is_new = existing_src is None
            prev_ver_id = (
                str(existing_src["current_version_id"])
                if existing_src and existing_src["current_version_id"]
                else None
            )

            if src_is_new:
                conn.execute(
                    """
                    INSERT INTO sources (
                        id, provider, repository_provider_id, provider_id, kind,
                        parent_source_id, current_version_id, availability, last_seen_at_us, last_reconciled_at_us
                    ) VALUES (?, 'git', ?, ?, ?, NULL, NULL, 'available', ?, ?)
                    """,
                    (src_id, repo_id, disp_p, doc_kind, ts, ts),
                )
                sources_added += 1
            else:
                conn.execute(
                    "UPDATE sources SET availability = 'available', last_seen_at_us = ?, last_reconciled_at_us = ? WHERE id = ?",
                    (ts, ts, src_id),
                )

            # Deleting a doc drops its search rows, so a doc that returns unchanged must rebuild them.
            was_deleted = existing_src is not None and str(existing_src["availability"]) == "deleted"
            if prev_ver_id == ver_id and not was_deleted:
                conn.execute(
                    "UPDATE source_versions SET last_observed_at_us = ? WHERE id = ?",
                    (ts, ver_id),
                )
                continue

            existing_ver = conn.execute(
                "SELECT id FROM source_versions WHERE id = ?",
                (ver_id,),
            ).fetchone()
            if existing_ver is None:
                conn.execute(
                    """
                    INSERT INTO source_versions (
                        id, source_id, content_sha256, title, body, raw_payload_json,
                        author_json, source_url, authored_at_us, updated_at_us,
                        first_observed_at_us, last_observed_at_us
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
                    """,
                    (
                        ver_id,
                        src_id,
                        content_hash,
                        title,
                        body_text,
                        canonical_json({"blob_oid": blob_oid, "kind": doc_kind, "path": disp_p}),
                        author_json_str,
                        authored_us,
                        updated_us,
                        ts,
                        ts,
                    ),
                )
                versions_added += 1
            else:
                conn.execute(
                    "UPDATE source_versions SET last_observed_at_us = ? WHERE id = ?",
                    (ts, ver_id),
                )

            conn.execute(
                "UPDATE sources SET current_version_id = ? WHERE id = ?",
                (ver_id, src_id),
            )

            line_idx = LineByteIndex(raw_bytes)
            l_s, l_e = line_idx.byte_span_to_lines(0, len(raw_bytes))
            disc_anchor = {
                "git_oid": blob_oid,
                "kind": "discussion",
                "original_review_commit": None,
                "path": {"path": disp_p, "path_bytes_b64": p_b64},
                "revision": active_revision,
                "source_version_id": ver_id,
                "span": {
                    "byte_end": len(raw_bytes),
                    "byte_start": 0,
                    "line_end": l_e,
                    "line_start": l_s,
                },
                "underlying_record_id": src_id,
            }
            body_sha = sha256_bytes(raw_bytes)
            ev_id = make_ev_id(
                repository_id=repo_id,
                kind="discussion",
                source_version_id=ver_id,
                git_oid=blob_oid,
                anchor=disc_anchor,
                content_sha256=body_sha,
            )
            conn.execute(
                """
                INSERT INTO evidence (id, kind, source_version_id, git_oid, anchor_json, content_sha256, state)
                VALUES (?, 'discussion', ?, ?, ?, ?, 'available')
                ON CONFLICT(id) DO NOTHING
                """,
                (ev_id, ver_id, blob_oid, canonical_json(disc_anchor), body_sha),
            )

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

            chunks = chunk_utf8_bytes(raw_bytes)
            for ch in chunks:
                meta_json = canonical_json(
                    {
                        "authored_at_us": authored_us,
                        "current": True,
                        "display_title": f"{title} ({disp_p})"[:200],
                        "evidence_id": ev_id,
                        "kind": doc_kind,
                        "path": disp_p,
                        "path_bytes_b64": p_b64,
                        "snippet": ch.text[:320],
                        "source_id": src_id,
                        "updated_at_us": updated_us,
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
                        normalize_search_text(f"{title} {disp_p}"),
                        normalize_search_text(f"{doc_kind} {disp_p}"),
                        normalize_search_text(f"{title} {ch.text}"),
                        meta_json,
                    ),
                )

            # Link backticked code references in ADR/design doc to active snapshot symbols
            linked_syms: list[str] = []
            seen_sym_ids: set[str] = set()
            for tm in BACKTICK_TOKEN_RE.finditer(body_text):
                tok = tm.group(1).strip("()")
                matched_ids: list[str] = []
                if tok in by_qname and len(by_qname[tok]) <= 3:
                    matched_ids.extend(by_qname[tok])
                elif tok in by_name and len(by_name[tok]) == 1:
                    matched_ids.extend(by_name[tok])
                elif tok in by_path_syms:
                    matched_ids.extend(by_path_syms[tok][:3])
                for m_sid in matched_ids:
                    if m_sid not in seen_sym_ids:
                        seen_sym_ids.add(m_sid)
                        linked_syms.append(m_sid)
                if len(linked_syms) >= 25:
                    break

            for t_sym_id in linked_syms:
                rel_id = make_relation_id(
                    from_kind="source",
                    from_id=src_id,
                    to_kind="symbol",
                    to_id=t_sym_id,
                    relation_kind="targets",
                    method="markdown_code_reference",
                )
                conn.execute(
                    """
                    INSERT INTO relations (
                        id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
                    ) VALUES (?, 'source', ?, 'symbol', ?, 'targets', 'markdown_code_reference', 'candidate', ?, ?, 'active')
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (rel_id, src_id, t_sym_id, ev_id, canonical_json({"doc_path": disp_p})),
                )

            ep_id = make_episode_id(repository_id=repo_id, pr_source_id=src_id)
            has_rev_commit = (
                conn.execute("SELECT 1 FROM commits WHERE oid = ?", (active_revision,)).fetchone()
                is not None
            )
            conn.execute(
                """
                INSERT INTO episodes (id, kind, primary_source_id, commit_oid, input_fingerprint, state)
                VALUES (?, 'doc', ?, ?, ?, 'pending')
                ON CONFLICT(id) DO UPDATE SET
                    commit_oid = COALESCE(excluded.commit_oid, episodes.commit_oid),
                    input_fingerprint = excluded.input_fingerprint,
                    state = CASE WHEN episodes.input_fingerprint != excluded.input_fingerprint THEN 'pending' ELSE episodes.state END
                """,
                (ep_id, src_id, active_revision if has_rev_commit else None, ver_id),
            )

    return (sources_added, versions_added)
