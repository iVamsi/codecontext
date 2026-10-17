import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codecontext.errors import CodeContextError
from codecontext.git import GitMirror
from codecontext.identity import (
    byte_span_to_lines,
    canonical_sha256,
    datetime_to_us,
    default_token_estimator,
    is_valid_record_id,
    make_ev_id,
    parse_public_id,
    path_bytes_to_display,
    sha256_bytes,
    us_to_iso_z,
)
from codecontext.models import (
    Anchor,
    Attribution,
    Budget,
    ByteSpan,
    ChangeRecord,
    ClaimRecord,
    CodeView,
    ContentSlice,
    ContextPack,
    ContextRequest,
    DecisionView,
    EvidenceRecord,
    EvidenceRef,
    EvidenceView,
    Expand,
    Gap,
    GapRecord,
    GetRequest,
    HunkRecord,
    OmittedCounts,
    Overview,
    PathRef,
    PathTarget,
    Record,
    RecordKind,
    RecordPage,
    RelationView,
    ResolvedTarget,
    SearchHit,
    SearchRequest,
    SearchResult,
    SourceRecord,
    SourceTimes,
    SourceVersionRecord,
    SummaryRecord,
    SummaryStatement,
    SymbolRecord,
    SymbolTarget,
)
from codecontext.store import (
    Store,
    decode_and_validate_cursor,
    encode_cursor,
)
from codecontext.syntax import detect_language_for_path, normalize_search_text
from codecontext.workspace import inspect_workspace

SEARCH_TOKEN_CEILING = 12000
MAX_QUERY_TERMS = 64
MAX_TERM_CHARS = 128

PREFIX_TO_RECORD_KIND: dict[str, RecordKind] = {
    "sym_": "symbol",
    "chg_": "change",
    "hunk_": "hunk",
    "src_": "source",
    "ver_": "source_version",
    "ev_": "evidence",
    "claim_": "claim",
    "sum_": "summary",
    "gap_": "gap",
}


def normalize_query_terms(query: str) -> tuple[list[str], bool]:
    """Normalize query through `normalize_search_text`, deduplicate terms in order,
    bound each term to 128 chars, and take at most 64 terms.
    Returns `(terms, query_truncated)`.
    """
    norm = normalize_search_text(query)
    if not norm:
        return ([], False)
    raw_terms = norm.split(" ")
    deduped: list[str] = []
    seen: set[str] = set()
    truncated = False
    for t in raw_terms:
        if not t:
            continue
        if len(t) > MAX_TERM_CHARS:
            t = t[:MAX_TERM_CHARS]
            truncated = True
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    if len(deduped) > MAX_QUERY_TERMS:
        deduped = deduped[:MAX_QUERY_TERMS]
        truncated = True
    return (deduped, truncated)


def build_fts_match_expression(terms: list[str]) -> str:
    """Build literal-safe FTS5 OR expression quoting each normalized term."""
    quoted: list[str] = []
    for t in terms:
        safe = t.replace('"', '""')
        quoted.append(f'"{safe}"')
    return " OR ".join(quoted)


def _check_source_version_cutoff_eligible(
    conn: sqlite3.Connection,
    ver_id: str,
    cutoff_us: int | None,
) -> tuple[bool, str | None]:
    """Check if a source_version is available and satisfies `cutoff_us`."""
    row = conn.execute(
        """
        SELECT sv.authored_at_us, sv.updated_at_us, s.availability
        FROM source_versions sv
        JOIN sources s ON s.id = sv.source_id
        WHERE sv.id = ?
        """,
        (ver_id,),
    ).fetchone()
    if not row:
        return (False, "record_not_found")
    if row["availability"] in ("inaccessible", "redacted", "deleted"):
        return (False, f"source_{row['availability']}")
    if cutoff_us is None:
        return (True, None)

    authored = row["authored_at_us"]
    updated = row["updated_at_us"]
    if authored is None and updated is None:
        return (False, "source_time_unknown")
    if authored is not None and int(authored) > cutoff_us:
        return (False, "source_authored_after_cutoff")
    if updated is not None and int(updated) > cutoff_us:
        return (False, "source_edited_after_cutoff")
    return (True, None)


def _select_eligible_version_for_source(
    conn: sqlite3.Connection,
    source_id: str,
    cutoff_us: int | None,
) -> tuple[str | None, bool]:
    """Select the latest eligible version of `source_id` at `cutoff_us`.
    Returns `(selected_ver_id, had_unknown_timestamp)`.
    """
    src_row = conn.execute(
        "SELECT current_version_id, availability FROM sources WHERE id = ?",
        (source_id,),
    ).fetchone()
    if not src_row or src_row["availability"] in ("inaccessible", "redacted", "deleted"):
        return (None, False)
    if cutoff_us is None:
        return (src_row["current_version_id"], False)

    ver_rows = conn.execute(
        """
        SELECT id, authored_at_us, updated_at_us
        FROM source_versions
        WHERE source_id = ?
        ORDER BY COALESCE(updated_at_us, authored_at_us, 0) DESC, first_observed_at_us DESC, id ASC
        """,
        (source_id,),
    ).fetchall()
    had_unknown = False
    for vr in ver_rows:
        authored = vr["authored_at_us"]
        updated = vr["updated_at_us"]
        if authored is None and updated is None:
            had_unknown = True
            continue
        if authored is not None and int(authored) > cutoff_us:
            continue
        if updated is not None and int(updated) > cutoff_us:
            continue
        return (str(vr["id"]), had_unknown)
    return (None, had_unknown)


def check_derivation_cutoff_eligible(
    conn: sqlite3.Connection,
    dependent_kind: str,
    dependent_id: str,
    cutoff_us: int | None,
    ancestor_commits: set[str] | None = None,
) -> tuple[bool, str | None]:
    """Check whether all dependencies of a claim or summary satisfy `cutoff_us` and ancestry."""
    if cutoff_us is None and ancestor_commits is None:
        return (True, None)

    dep_rows = conn.execute(
        "SELECT input_kind, input_id FROM dependencies WHERE dependent_kind = ? AND dependent_id = ?",
        (dependent_kind, dependent_id),
    ).fetchall()

    for dr in dep_rows:
        in_kind = str(dr["input_kind"])
        in_id = str(dr["input_id"])
        if in_kind == "source_version":
            ok, reason = _check_source_version_cutoff_eligible(conn, in_id, cutoff_us)
            if not ok:
                return (False, reason)
        elif in_kind == "evidence":
            ev_row = conn.execute(
                "SELECT kind, source_version_id, git_oid, state FROM evidence WHERE id = ?",
                (in_id,),
            ).fetchone()
            if not ev_row or ev_row["state"] != "available":
                return (False, "evidence_unavailable")
            if ev_row["source_version_id"]:
                ok, reason = _check_source_version_cutoff_eligible(
                    conn, str(ev_row["source_version_id"]), cutoff_us
                )
                if not ok:
                    return (False, reason)
        elif in_kind == "claim":
            cl_row = conn.execute("SELECT state FROM claims WHERE id = ?", (in_id,)).fetchone()
            if not cl_row or cl_row["state"] != "ready":
                return (False, "dependent_claim_not_ready")
            ok, reason = check_derivation_cutoff_eligible(
                conn, "claim", in_id, cutoff_us, ancestor_commits
            )
            if not ok:
                return (False, reason)

    return (True, None)


def _format_snippet(text: str, max_chars: int = 320) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1] + "…"


@dataclass
class _CandidateHit:
    id: str
    kind: RecordKind
    title: str
    snippet: str
    match: str  # 'exact_id', 'exact_path', 'exact_symbol', 'lexical'
    path: PathRef | None
    evidence_id: str | None
    state: str | None
    score: float = 0.0


def execute_search(
    store: Store,
    mirror: GitMirror,
    request: SearchRequest,
    *,
    token_counter: Callable[[str], int] = default_token_estimator,
    estimator_id: str = "utf8_bytes_div3_v1",
    count_kind: str = "estimated",
    exclude_enriched: bool = False,
    pre_authorize_hook: Callable[[], None] | None = None,
) -> SearchResult:
    """Execute deterministic lexical + exact search over the active prepared snapshot."""
    cutoff_us = datetime_to_us(request.knowledge_cutoff) if request.knowledge_cutoff else None
    q_fp = canonical_sha256(
        {
            "cutoff_us": cutoff_us,
            "exclude_enriched": exclude_enriched,
            "kind": request.kind,
            "query": request.query,
            "revision": request.revision,
        }
    )

    with store.read_snapshot() as snap:
        conn = snap.conn
        active_rev = mirror.resolve_prepared_revision(conn, request.revision)

        offset = 0
        if request.cursor is not None:
            offset, _ = decode_and_validate_cursor(
                request.cursor,
                expected_query_fingerprint=q_fp,
                current_generation=snap.generation,
                current_access_epoch=snap.access_epoch,
            )

        ancestor_commits = {
            str(r["commit_oid"])
            for r in conn.execute("SELECT commit_oid FROM snapshot_commits").fetchall()
        }

        request_gaps: list[Gap] = []
        terms, query_truncated = normalize_query_terms(request.query)
        if query_truncated:
            request_gaps.append(
                Gap(
                    id=None,
                    code="QUERY_TRUNCATED",
                    scope="search:query",
                    message="Search query exceeded 64 normalized terms or 128-character term limit and was truncated.",
                    retryable=False,
                )
            )

        pinned_hits: dict[str, _CandidateHit] = {}
        raw_q = request.query.strip()

        # 1. Exact matches (record ID, active path, qualified symbol name)
        if is_valid_record_id(raw_q):
            prefix, _ = parse_public_id(raw_q)
            rkind = PREFIX_TO_RECORD_KIND[prefix]
            kind_allowed = (
                request.kind == "all"
                or (request.kind == "code" and rkind in ("symbol", "evidence"))
                or (request.kind == "history" and rkind in ("change", "hunk", "source", "source_version", "evidence"))
                or (request.kind == "decision" and rkind == "claim" and not exclude_enriched)
                or (request.kind == "summary" and rkind == "summary" and not exclude_enriched)
            )
            if kind_allowed:
                exact_hit = _try_resolve_exact_id_hit(conn, raw_q, rkind, cutoff_us, ancestor_commits)
                if exact_hit is not None:
                    pinned_hits[exact_hit.id] = exact_hit

        if request.kind in ("all", "code"):
            # Exact active path match
            path_bytes = raw_q.encode("utf-8", errors="surrogateescape")
            sf_row = conn.execute(
                "SELECT path, blob_oid, extraction_id FROM snapshot_files WHERE path = ?",
                (path_bytes,),
            ).fetchone()
            if sf_row is not None:
                disp_p, b64_p = path_bytes_to_display(bytes(sf_row["path"]))
                pref = PathRef(path=disp_p, path_bytes_b64=b64_p)
                if sf_row["extraction_id"]:
                    first_sym = conn.execute(
                        "SELECT id, qualified_name, signature FROM symbols WHERE extraction_id = ? ORDER BY byte_start ASC, id ASC LIMIT 1",
                        (sf_row["extraction_id"],),
                    ).fetchone()
                    if first_sym:
                        sid = str(first_sym["id"])
                        if sid not in pinned_hits:
                            pinned_hits[sid] = _CandidateHit(
                                id=sid,
                                kind="symbol",
                                title=f"{first_sym['qualified_name']} ({disp_p})",
                                snippet=_format_snippet(str(first_sym["signature"])),
                                match="exact_path",
                                path=pref,
                                evidence_id=None,
                                state=None,
                            )
                else:
                    doc_row = conn.execute(
                        "SELECT record_id, metadata_json FROM search_documents WHERE record_kind = 'code' AND json_extract(metadata_json, '$.path') = ? ORDER BY chunk_ordinal ASC LIMIT 1",
                        (disp_p,),
                    ).fetchone()
                    if doc_row:
                        rid = str(doc_row["record_id"])
                        meta_d = json.loads(str(doc_row["metadata_json"]))
                        if rid not in pinned_hits:
                            pinned_hits[rid] = _CandidateHit(
                                id=rid,
                                kind="evidence",
                                title=disp_p,
                                snippet=_format_snippet(str(meta_d.get("snippet") or disp_p)),
                                match="exact_path",
                                path=pref,
                                evidence_id=rid,
                                state=None,
                            )

            # Exact qualified symbol name match in active snapshot
            sym_matches = conn.execute(
                """
                SELECT s.id, s.name, s.qualified_name, s.signature, sf.path
                FROM symbols s
                JOIN snapshot_files sf ON sf.extraction_id = s.extraction_id
                WHERE s.qualified_name = ? OR s.name = ?
                ORDER BY CASE WHEN s.qualified_name = ? THEN 0 ELSE 1 END, sf.path ASC, s.byte_start ASC, s.id ASC
                LIMIT 10
                """,
                (raw_q, raw_q, raw_q),
            ).fetchall()
            for sm in sym_matches:
                sid = str(sm["id"])
                if sid not in pinned_hits:
                    disp_p, b64_p = path_bytes_to_display(bytes(sm["path"]))
                    pinned_hits[sid] = _CandidateHit(
                        id=sid,
                        kind="symbol",
                        title=f"{sm['qualified_name']} ({disp_p})",
                        snippet=_format_snippet(str(sm["signature"])),
                        match="exact_symbol",
                        path=PathRef(path=disp_p, path_bytes_b64=b64_p),
                        evidence_id=None,
                        state=None,
                    )

        # 2. Lexical search per requested kind using BM25 + Reciprocal Rank Fusion
        requested_kinds: list[str]
        if request.kind == "all":
            requested_kinds = (
                ["code", "history"]
                if exclude_enriched
                else ["code", "history", "decision", "summary"]
            )
        elif exclude_enriched and request.kind in ("decision", "summary"):
            requested_kinds = []
        else:
            requested_kinds = [request.kind]

        fused_candidates: dict[str, _CandidateHit] = {}
        had_unknown_time_gap = False

        if terms and requested_kinds:
            match_expr = build_fts_match_expression(terms)
            for lkind in requested_kinds:
                fts_rows = conn.execute(
                    """
                    SELECT sd.record_id, sd.record_kind, sd.metadata_json,
                           bm25(search_fts) AS bm25_score
                    FROM search_fts
                    JOIN search_documents sd ON sd.rowid = search_fts.rowid
                    WHERE search_fts MATCH ? AND sd.record_kind = ?
                    ORDER BY bm25_score ASC, sd.record_id ASC, sd.chunk_ordinal ASC
                    LIMIT 250
                    """,
                    (match_expr, lkind),
                ).fetchall()

                rank_in_kind = 0
                seen_in_kind: set[str] = set()
                seen_sources_in_kind: set[str] = set()

                for frow in fts_rows:
                    rec_id = str(frow["record_id"])
                    meta = json.loads(str(frow["metadata_json"]))

                    # Eligibility filtering before ranking!
                    cand = _validate_and_project_lexical_candidate(
                        conn,
                        lkind=lkind,
                        rec_id=rec_id,
                        meta=meta,
                        cutoff_us=cutoff_us,
                        ancestor_commits=ancestor_commits,
                    )
                    if cand is None:
                        if cutoff_us is not None and meta.get("authored_at_us") is None and meta.get("committed_at_us") is None and lkind == "history":
                            had_unknown_time_gap = True
                        continue

                    # Collapse duplicate versions of the same source
                    src_collapse_key = meta.get("source_id")
                    if src_collapse_key:
                        if src_collapse_key in seen_sources_in_kind:
                            continue
                        seen_sources_in_kind.add(src_collapse_key)

                    if cand.id in seen_in_kind:
                        continue
                    seen_in_kind.add(cand.id)
                    rank_in_kind += 1

                    rrf_contrib = 1.0 / (60.0 + rank_in_kind)
                    if cand.id in fused_candidates:
                        fused_candidates[cand.id].score += rrf_contrib
                    else:
                        cand.score = rrf_contrib
                        fused_candidates[cand.id] = cand

                    if rank_in_kind >= 100:
                        break

        if had_unknown_time_gap:
            request_gaps.append(
                Gap(
                    id=None,
                    code="SOURCE_TIME_UNKNOWN",
                    scope="search:cutoff",
                    message="One or more discussion records with unknown source timestamps were excluded under knowledge_cutoff.",
                    retryable=False,
                )
            )

        # 3. Merge pinned exact matches first, then fused score descending, then stable ID
        ordered_candidates: list[_CandidateHit] = list(pinned_hits.values())
        remaining_lexical = [
            c for cid, c in fused_candidates.items() if cid not in pinned_hits
        ]
        remaining_lexical.sort(key=lambda c: (-round(c.score, 12), c.id))
        ordered_candidates.extend(remaining_lexical)

        shortlist_truncated = len(ordered_candidates) > 100
        shortlist = ordered_candidates[:100]

        page_slice = shortlist[offset : offset + request.limit]

        cov, persisted_gaps, omitted_persisted_gaps = Store.build_coverage_and_gaps(
            conn, max_detailed_gaps=max(0, 20 - len(request_gaps))
        )
        all_gaps = [*request_gaps, *persisted_gaps][:20]

        # 4. Enforce fixed 12,000-token ceiling by trimming hits from the end if needed
        admitted_count = len(page_slice)
        while admitted_count >= 0:
            cur_slice = page_slice[:admitted_count]
            hits_out = [
                SearchHit(
                    id=c.id,
                    kind=c.kind,
                    title=c.title[:1000],
                    snippet=c.snippet[:320],
                    rank=offset + idx + 1,
                    match=c.match,  # type: ignore[arg-type]
                    path=c.path,
                    evidence_id=c.evidence_id,
                    state=c.state,  # type: ignore[arg-type]
                )
                for idx, c in enumerate(cur_slice)
            ]
            omitted_hits = (len(shortlist) - (offset + admitted_count)) if (offset + admitted_count) < len(shortlist) else 0
            next_offset = offset + admitted_count
            next_cur = (
                encode_cursor(
                    query_fingerprint=q_fp,
                    generation=snap.generation,
                    access_epoch=snap.access_epoch,
                    offset=next_offset,
                )
                if next_offset < len(shortlist)
                else None
            )
            expand_list = [
                Expand(id=h.id, reason=f"Expand {h.kind} record details")
                for h in hits_out[:20]
            ]
            is_truncated = (
                query_truncated
                or shortlist_truncated
                or omitted_hits > 0
                or omitted_persisted_gaps > 0
            )
            omitted_obj = OmittedCounts(
                hits=omitted_hits,
                gaps=omitted_persisted_gaps,
            )

            used_tokens = 0
            for _ in range(3):
                result = SearchResult(
                    repository_id=snap.repository_id,
                    revision=active_rev,
                    generation=snap.generation,
                    access_epoch=snap.access_epoch,
                    coverage=cov,
                    gaps=all_gaps,
                    query=request.query,
                    hits=hits_out,
                    next_cursor=next_cur,
                    truncated=is_truncated,
                    budget=Budget(
                        requested_tokens=SEARCH_TOKEN_CEILING,
                        used_tokens=used_tokens,
                        estimator=estimator_id,
                        count_kind="tokenizer" if count_kind == "tokenizer" else "estimated",
                        truncated=is_truncated,
                        omitted_counts=omitted_obj,
                    ),
                    expand=expand_list,
                )
                new_used = int(token_counter(result.model_dump_json()))
                if new_used == used_tokens:
                    break
                used_tokens = new_used

            if used_tokens <= SEARCH_TOKEN_CEILING or admitted_count == 0:
                break
            admitted_count -= 1

        expected_epoch = snap.access_epoch

    if pre_authorize_hook is not None:
        pre_authorize_hook()
    store.verify_access_epoch_and_authorize(expected_epoch)
    return result


def _try_resolve_exact_id_hit(
    conn: sqlite3.Connection,
    rec_id: str,
    rkind: RecordKind,
    cutoff_us: int | None,
    ancestor_commits: set[str],
) -> _CandidateHit | None:
    if rkind == "symbol":
        row = conn.execute(
            """
            SELECT s.id, s.qualified_name, s.signature, sf.path
            FROM symbols s
            JOIN snapshot_files sf ON sf.extraction_id = s.extraction_id
            WHERE s.id = ?
            ORDER BY sf.path ASC LIMIT 1
            """,
            (rec_id,),
        ).fetchone()
        if not row:
            return None
        disp_p, b64_p = path_bytes_to_display(bytes(row["path"]))
        return _CandidateHit(
            id=rec_id,
            kind="symbol",
            title=f"{row['qualified_name']} ({disp_p})",
            snippet=_format_snippet(str(row["signature"])),
            match="exact_id",
            path=PathRef(path=disp_p, path_bytes_b64=b64_p),
            evidence_id=None,
            state=None,
        )
    if rkind == "claim":
        row = conn.execute(
            "SELECT id, statement, rationale, state FROM claims WHERE id = ? AND state = 'ready'",
            (rec_id,),
        ).fetchone()
        if not row:
            return None
        ok, _ = check_derivation_cutoff_eligible(conn, "claim", rec_id, cutoff_us, ancestor_commits)
        if not ok:
            return None
        return _CandidateHit(
            id=rec_id,
            kind="claim",
            title=str(row["statement"]),
            snippet=_format_snippet(str(row["rationale"] or row["statement"])),
            match="exact_id",
            path=None,
            evidence_id=None,
            state="ready",
        )
    if rkind == "source":
        sel_ver, _ = _select_eligible_version_for_source(conn, rec_id, cutoff_us)
        if not sel_ver:
            return None
        vrow = conn.execute(
            "SELECT title, body FROM source_versions WHERE id = ?", (sel_ver,)
        ).fetchone()
        if not vrow:
            return None
        return _CandidateHit(
            id=rec_id,
            kind="source",
            title=str(vrow["title"] or rec_id),
            snippet=_format_snippet(str(vrow["body"] or vrow["title"] or rec_id)),
            match="exact_id",
            path=None,
            evidence_id=None,
            state=None,
        )
    if rkind == "source_version":
        ok, _ = _check_source_version_cutoff_eligible(conn, rec_id, cutoff_us)
        if not ok:
            return None
        vrow = conn.execute(
            "SELECT title, body FROM source_versions WHERE id = ?", (rec_id,)
        ).fetchone()
        if not vrow:
            return None
        return _CandidateHit(
            id=rec_id,
            kind="source_version",
            title=str(vrow["title"] or rec_id),
            snippet=_format_snippet(str(vrow["body"] or vrow["title"] or rec_id)),
            match="exact_id",
            path=None,
            evidence_id=None,
            state=None,
        )
    return None


def _validate_and_project_lexical_candidate(
    conn: sqlite3.Connection,
    *,
    lkind: str,
    rec_id: str,
    meta: dict[str, Any],
    cutoff_us: int | None,
    ancestor_commits: set[str],
) -> _CandidateHit | None:
    prefix, _ = parse_public_id(rec_id)
    rkind = PREFIX_TO_RECORD_KIND[prefix]

    if lkind == "code":
        p_str = meta.get("path")
        pref = PathRef(path=p_str, path_bytes_b64=meta.get("path_bytes_b64")) if p_str else None
        return _CandidateHit(
            id=rec_id,
            kind=rkind,
            title=str(meta.get("display_title") or rec_id),
            snippet=_format_snippet(str(meta.get("signature") or meta.get("snippet") or "")),
            match="lexical",
            path=pref,
            evidence_id=meta.get("evidence_id"),
            state=None,
        )

    if lkind == "history":
        if rkind == "change":
            c_oid = meta.get("commit_oid")
            if c_oid and c_oid not in ancestor_commits:
                return None
            if cutoff_us is not None:
                c_time = meta.get("committed_at_us") or meta.get("authored_at_us")
                if c_time is None or int(c_time) > cutoff_us:
                    return None
            p_str = meta.get("path")
            pref = PathRef(path=p_str, path_bytes_b64=meta.get("path_bytes_b64")) if p_str else None
            return _CandidateHit(
                id=rec_id,
                kind="change",
                title=str(meta.get("display_title") or rec_id),
                snippet=_format_snippet(str(meta.get("snippet") or "")),
                match="lexical",
                path=pref,
                evidence_id=meta.get("evidence_id"),
                state=None,
            )

        if rkind == "source_version":
            src_id = meta.get("source_id")
            if not src_id:
                return None
            eligible_ver, _ = _select_eligible_version_for_source(conn, str(src_id), cutoff_us)
            if eligible_ver != rec_id:
                return None
            p_str = meta.get("path")
            pref = PathRef(path=p_str, path_bytes_b64=None) if p_str else None
            return _CandidateHit(
                id=rec_id,
                kind="source_version",
                title=str(meta.get("display_title") or rec_id),
                snippet=_format_snippet(str(meta.get("snippet") or "")),
                match="lexical",
                path=pref,
                evidence_id=meta.get("evidence_id"),
                state=None,
            )

    if lkind == "decision" and rkind == "claim":
        cl_row = conn.execute(
            "SELECT statement, rationale, state FROM claims WHERE id = ?",
            (rec_id,),
        ).fetchone()
        if not cl_row or cl_row["state"] != "ready":
            return None
        ok, _ = check_derivation_cutoff_eligible(
            conn, "claim", rec_id, cutoff_us, ancestor_commits
        )
        if not ok:
            return None
        return _CandidateHit(
            id=rec_id,
            kind="claim",
            title=str(cl_row["statement"]),
            snippet=_format_snippet(str(cl_row["rationale"] or cl_row["statement"])),
            match="lexical",
            path=None,
            evidence_id=None,
            state="ready",
        )

    if lkind == "summary" and rkind == "summary":
        s_row = conn.execute(
            "SELECT scope_kind, scope_id, text, state FROM summaries WHERE id = ?",
            (rec_id,),
        ).fetchone()
        if not s_row or s_row["state"] != "ready":
            return None
        ok, _ = check_derivation_cutoff_eligible(
            conn, "summary", rec_id, cutoff_us, ancestor_commits
        )
        if not ok:
            return None
        return _CandidateHit(
            id=rec_id,
            kind="summary",
            title=f"Summary ({s_row['scope_kind']})",
            snippet=_format_snippet(str(s_row["text"])),
            match="lexical",
            path=None,
            evidence_id=None,
            state="ready",
        )

    return None


def build_evidence_view_from_row(
    conn: sqlite3.Connection,
    mirror: GitMirror,
    ev_row: sqlite3.Row | dict[str, Any],
    *,
    ancestor_commits: set[str],
    cutoff_us: int | None = None,
    check_cutoff: bool = True,
) -> EvidenceView:
    """Project an `evidence` table row into a validated `EvidenceView`."""
    ev_id = str(ev_row["id"])
    ev_kind = str(ev_row["kind"])
    sv_id = ev_row["source_version_id"]
    git_oid = ev_row["git_oid"]
    anchor_data = json.loads(str(ev_row["anchor_json"]))
    content_sha = str(ev_row["content_sha256"])
    ev_state = str(ev_row["state"])

    anchor = Anchor.model_validate(anchor_data)
    ev_ref = EvidenceRef(id=ev_id, anchor=anchor, content_sha256=content_sha)

    if ev_kind == "discussion" and sv_id is not None:
        if check_cutoff and cutoff_us is not None:
            ok, reason = _check_source_version_cutoff_eligible(conn, str(sv_id), cutoff_us)
            if not ok:
                raise CodeContextError(
                    code="RECORD_NOT_ELIGIBLE",
                    message="Evidence source version is ineligible under knowledge_cutoff.",
                    retryable=False,
                    details={"id": ev_id, "reason": reason or "cutoff_violation"},
                )

        sv_row = conn.execute(
            """
            SELECT sv.body, sv.author_json, sv.source_url,
                   sv.authored_at_us, sv.updated_at_us,
                   sv.first_observed_at_us, sv.last_observed_at_us,
                   s.availability
            FROM source_versions sv
            JOIN sources s ON s.id = sv.source_id
            WHERE sv.id = ?
            """,
            (str(sv_id),),
        ).fetchone()
        if not sv_row or sv_row["availability"] in ("redacted", "deleted"):
            raise CodeContextError(
                code="RECORD_NOT_FOUND",
                message=f"Evidence {ev_id} is not available.",
                retryable=False,
                details={"id": ev_id},
            )
        if sv_row["availability"] == "inaccessible":
            raise CodeContextError(
                code="ACCESS_BLOCKED",
                message=f"Evidence {ev_id} source is inaccessible.",
                retryable=False,
                details={"id": ev_id},
            )

        author_info = json.loads(str(sv_row["author_json"])) if sv_row["author_json"] else {}
        attr = Attribution(
            provider="github",
            actor_id=author_info.get("actor_id"),
            display_name=author_info.get("display_name"),
            source_url=sv_row["source_url"],
        )
        times = SourceTimes(
            authored_at=us_to_iso_z(int(sv_row["authored_at_us"])) if sv_row["authored_at_us"] is not None else None,
            updated_at=us_to_iso_z(int(sv_row["updated_at_us"])) if sv_row["updated_at_us"] is not None else None,
            first_observed_at=us_to_iso_z(int(sv_row["first_observed_at_us"])),
            last_observed_at=us_to_iso_z(int(sv_row["last_observed_at_us"])),
        )
        excerpt: str | None = None
        avail: Any = "oversized" if ev_state == "oversized" else sv_row["availability"]
        if sv_row["body"] is not None and ev_state != "oversized":
            raw_b = str(sv_row["body"]).encode("utf-8")
            slice_b = raw_b[anchor.span.byte_start : anchor.span.byte_end]
            excerpt = slice_b.decode("utf-8", errors="replace")[:8192]

        return EvidenceView(
            ref=ev_ref,
            attribution=attr,
            times=times,
            excerpt=excerpt,
            availability=avail if avail in ("available", "unverified", "inaccessible", "missing_object", "oversized", "binary") else "available",
            historical_scope="discussion",
        )

    # Git-backed evidence ('code', 'commit', 'hunk')
    rev_oid = anchor.revision or git_oid
    scope_val: Any = (
        "target_ancestor"
        if (rev_oid and rev_oid in ancestor_commits)
        else ("nonancestor" if rev_oid else "unknown")
    )

    c_row = (
        conn.execute(
            "SELECT author_name, author_email, authored_at_us, committed_at_us, first_seen_at_us FROM commits WHERE oid = ?",
            (rev_oid,),
        ).fetchone()
        if rev_oid
        else None
    )
    if check_cutoff and cutoff_us is not None and c_row is not None:
        c_time = c_row["committed_at_us"] or c_row["authored_at_us"]
        if c_time is None or int(c_time) > cutoff_us:
            raise CodeContextError(
                code="RECORD_NOT_ELIGIBLE",
                message="Git evidence commit is outside the requested knowledge cutoff.",
                retryable=False,
                details={"id": ev_id, "reason": "commit_after_cutoff"},
            )

    attr = Attribution(
        provider="git",
        actor_id=c_row["author_email"] if c_row else None,
        display_name=c_row["author_name"] if c_row else None,
        source_url=None,
    )
    times = (
        SourceTimes(
            authored_at=us_to_iso_z(int(c_row["authored_at_us"])) if c_row["authored_at_us"] is not None else None,
            updated_at=us_to_iso_z(int(c_row["committed_at_us"])) if c_row["committed_at_us"] is not None else None,
            first_observed_at=us_to_iso_z(int(c_row["first_seen_at_us"])),
            last_observed_at=us_to_iso_z(int(c_row["first_seen_at_us"])),
        )
        if c_row
        else None
    )

    if ev_kind == "hunk" and anchor.underlying_record_id:
        h_row = conn.execute(
            "SELECT patch_text FROM hunks WHERE id = ?",
            (anchor.underlying_record_id,),
        ).fetchone()
        excerpt = str(h_row["patch_text"])[:8192] if h_row and h_row["patch_text"] else None
        return EvidenceView(
            ref=ev_ref,
            attribution=attr,
            times=times,
            excerpt=excerpt,
            availability="available" if h_row else "missing_object",
            historical_scope=scope_val,
        )

    if git_oid:
        with mirror.open_batch_session() as batch:
            obj = batch.read_object(str(git_oid))
        if obj is None:
            return EvidenceView(
                ref=ev_ref,
                attribution=attr,
                times=times,
                excerpt=None,
                availability="missing_object",
                historical_scope=scope_val,
            )
        raw_obj = obj[2]
        slice_b = raw_obj[anchor.span.byte_start : anchor.span.byte_end]
        if b"\x00" in slice_b:
            return EvidenceView(
                ref=ev_ref,
                attribution=attr,
                times=times,
                excerpt=None,
                availability="binary",
                historical_scope=scope_val,
            )
        try:
            excerpt_str = slice_b.decode("utf-8", errors="strict")[:8192]
            return EvidenceView(
                ref=ev_ref,
                attribution=attr,
                times=times,
                excerpt=excerpt_str,
                availability="available",
                historical_scope=scope_val,
            )
        except UnicodeDecodeError:
            return EvidenceView(
                ref=ev_ref,
                attribution=attr,
                times=times,
                excerpt=None,
                availability="available",
                historical_scope=scope_val,
            )

    return EvidenceView(
        ref=ev_ref,
        attribution=attr,
        times=times,
        excerpt=None,
        availability="missing_object",
        historical_scope=scope_val,
    )


def _load_relations_for_record(
    conn: sqlite3.Connection,
    record_kind: str,
    record_id: str,
) -> list[RelationView]:
    rows = conn.execute(
        """
        SELECT id, from_kind, from_id, to_kind, to_id, relation_kind,
               method, mapping, evidence_id, details_json, state
        FROM relations
        WHERE (from_kind = ? AND from_id = ?) OR (to_kind = ? AND to_id = ?)
        ORDER BY id ASC
        """,
        (record_kind, record_id, record_kind, record_id),
    ).fetchall()
    out: list[RelationView] = []
    for r in rows:
        details = json.loads(str(r["details_json"])) if r["details_json"] else {}
        out.append(
            RelationView(
                id=str(r["id"]),
                from_kind=str(r["from_kind"]),
                from_id=str(r["from_id"]),
                to_kind=str(r["to_kind"]),
                to_id=str(r["to_id"]),
                relation_kind=r["relation_kind"],
                method=str(r["method"]),
                mapping=r["mapping"],
                evidence_id=r["evidence_id"],
                state=r["state"],
                explanation=details.get("explanation"),
            )
        )
    return out


def build_decision_view_for_claim(
    conn: sqlite3.Connection,
    mirror: GitMirror,
    claim_row: sqlite3.Row | dict[str, Any],
    *,
    ancestor_commits: set[str],
    cutoff_us: int | None,
    target_mapping: str = "exact",
    target_applicability: str = "historical_only",
) -> tuple[DecisionView, list[EvidenceView]]:
    """Build a `DecisionView` and its required supporting/counter/applicability `EvidenceView` list."""
    cid = str(claim_row["id"])
    ce_rows = conn.execute(
        """
        SELECT ce.evidence_id, ce.role, e.id, e.kind, e.source_version_id,
               e.git_oid, e.anchor_json, e.content_sha256, e.state
        FROM claim_evidence ce
        JOIN evidence e ON e.id = ce.evidence_id
        WHERE ce.claim_id = ?
        ORDER BY ce.evidence_id ASC, ce.role ASC
        """,
        (cid,),
    ).fetchall()

    support_refs: list[EvidenceRef] = []
    counter_refs: list[EvidenceRef] = []
    app_refs: list[EvidenceRef] = []
    ev_views_by_id: dict[str, EvidenceView] = {}
    attributions: list[Attribution] = []
    seen_attr: set[tuple[str, str | None, str | None]] = set()

    for cer in ce_rows:
        ev_view = build_evidence_view_from_row(
            conn,
            mirror,
            cer,
            ancestor_commits=ancestor_commits,
            cutoff_us=cutoff_us,
        )
        ev_views_by_id[ev_view.ref.id] = ev_view
        role = str(cer["role"])
        if role in ("support", "lifecycle"):
            if all(r.id != ev_view.ref.id for r in support_refs):
                support_refs.append(ev_view.ref)
        elif role == "counterevidence":
            if all(r.id != ev_view.ref.id for r in counter_refs):
                counter_refs.append(ev_view.ref)
        elif role == "applicability":
            if all(r.id != ev_view.ref.id for r in app_refs):
                app_refs.append(ev_view.ref)

        attr_key = (
            ev_view.attribution.provider,
            ev_view.attribution.actor_id,
            ev_view.attribution.display_name,
        )
        if attr_key not in seen_attr:
            seen_attr.add(attr_key)
            attributions.append(ev_view.attribution)

    effective_app = target_applicability if (target_applicability == "corroborated" and app_refs) else (
        "historical_only" if target_applicability == "corroborated" else target_applicability
    )

    d_view = DecisionView(
        id=cid,
        statement=str(claim_row["statement"]),
        rationale=claim_row["rationale"],
        alternatives=json.loads(str(claim_row["alternatives_json"])),
        constraints=json.loads(str(claim_row["constraints_json"])),
        support=claim_row["support"],
        lifecycle=claim_row["lifecycle"],
        state=claim_row["state"],
        mapping=target_mapping,  # type: ignore[arg-type]
        applicability=effective_app,  # type: ignore[arg-type]
        attributions=attributions[:20],
        evidence_refs=support_refs[:40],
        counterevidence_refs=counter_refs[:20],
        applicability_evidence_refs=app_refs[:10],
    )
    return d_view, list(ev_views_by_id.values())


def _slice_utf8_bytes_for_page(
    raw_bytes: bytes,
    start_offset: int,
    max_slice_bytes: int,
) -> tuple[str, ByteSpan, int]:
    """Return a UTF-8-safe slice starting at `start_offset` of at most `max_slice_bytes` bytes.
    Returns `(decoded_text, byte_span, next_byte_offset)`.
    """
    n = len(raw_bytes)
    if start_offset >= n or max_slice_bytes <= 0:
        return ("", ByteSpan(byte_start=min(start_offset, n), byte_end=min(start_offset, n)), n)

    end = min(start_offset + max_slice_bytes, n)
    if end < n:
        while end > start_offset and (raw_bytes[end] & 0xC0) == 0x80:
            end -= 1
    chunk_b = raw_bytes[start_offset:end]
    text = chunk_b.decode("utf-8", errors="replace")
    l_s, l_e = byte_span_to_lines(raw_bytes, start_offset, end)
    span = ByteSpan(byte_start=start_offset, byte_end=end, line_start=l_s, line_end=l_e)
    return (text, span, end)


def execute_get(
    store: Store,
    mirror: GitMirror,
    request: GetRequest,
    *,
    token_counter: Callable[[str], int] = default_token_estimator,
    estimator_id: str = "utf8_bytes_div3_v1",
    count_kind: str = "estimated",
    pre_authorize_hook: Callable[[], None] | None = None,
) -> RecordPage:
    """Resolve and paginate a public record ID (`sym_`, `chg_`, `hunk_`, `src_`, `ver_`, `ev_`, `claim_`, `sum_`, `gap_`)."""
    prefix, _ = parse_public_id(request.id)
    rkind = PREFIX_TO_RECORD_KIND[prefix]
    cutoff_us = datetime_to_us(request.knowledge_cutoff) if request.knowledge_cutoff else None
    q_fp = canonical_sha256({"cutoff_us": cutoff_us, "id": request.id})

    with store.read_snapshot() as snap:
        conn = snap.conn
        ancestor_commits = {
            str(r["commit_oid"])
            for r in conn.execute("SELECT commit_oid FROM snapshot_commits").fetchall()
        }

        byte_offset = 0
        rel_offset = 0
        if request.cursor is not None:
            byte_offset, extra = decode_and_validate_cursor(
                request.cursor,
                expected_query_fingerprint=q_fp,
                current_generation=snap.generation,
                current_access_epoch=snap.access_epoch,
            )
            rel_offset = int(extra.get("rel_offset", 0))

        request_gaps: list[Gap] = []
        raw_content_field: str | None = None
        raw_content_bytes: bytes = b""
        ev_list: list[EvidenceView] = []
        rel_list: list[RelationView] = []
        expand_list: list[Expand] = []
        record_obj: Record

        if rkind == "symbol":
            srow = conn.execute(
                """
                SELECT s.id, s.extraction_id, s.parent_symbol_id, s.kind, s.name, s.qualified_name,
                       s.byte_start, s.byte_end, s.body_byte_start, s.body_byte_end,
                       s.line_start, s.line_end, s.signature,
                       e.blob_oid, e.language, e.extractor_fingerprint
                FROM symbols s
                JOIN extractions e ON e.id = s.extraction_id
                WHERE s.id = ?
                """,
                (request.id,),
            ).fetchone()
            if not srow:
                raise CodeContextError(
                    code="RECORD_NOT_FOUND",
                    message=f"Symbol {request.id} not found.",
                    retryable=False,
                    details={"id": request.id},
                )
            b_oid = str(srow["blob_oid"])
            decl_span = ByteSpan(
                byte_start=int(srow["byte_start"]),
                byte_end=int(srow["byte_end"]),
                line_start=int(srow["line_start"]),
                line_end=int(srow["line_end"]),
            )
            body_span: ByteSpan | None = None
            with mirror.open_batch_session() as batch:
                b_obj = batch.read_object(b_oid)
            if b_obj is not None:
                full_blob = b_obj[2]
                raw_content_field = "code"
                raw_content_bytes = full_blob[decl_span.byte_start : decl_span.byte_end]
                if srow["body_byte_start"] is not None and srow["body_byte_end"] is not None:
                    bl_s, bl_e = byte_span_to_lines(
                        full_blob, int(srow["body_byte_start"]), int(srow["body_byte_end"])
                    )
                    body_span = ByteSpan(
                        byte_start=int(srow["body_byte_start"]),
                        byte_end=int(srow["body_byte_end"]),
                        line_start=bl_s,
                        line_end=bl_e,
                    )
            else:
                request_gaps.append(
                    Gap(
                        id=None,
                        code="MISSING_GIT_OBJECT",
                        scope=f"blob:{b_oid}",
                        message=f"Git blob {b_oid[:12]} is missing from object store.",
                        retryable=True,
                    )
                )

            record_obj = SymbolRecord(
                id=request.id,
                blob_oid=b_oid,
                language=str(srow["language"]),
                extractor_fingerprint=str(srow["extractor_fingerprint"]),
                name=str(srow["name"]),
                qualified_name=str(srow["qualified_name"]),
                symbol_kind=str(srow["kind"]),
                declaration_span=decl_span,
                body_span=body_span,
                parent_symbol_id=srow["parent_symbol_id"],
                signature=str(srow["signature"]),
            )
            rel_list = _load_relations_for_record(conn, "symbol", request.id)

        elif rkind == "change":
            crow = conn.execute(
                "SELECT * FROM changes WHERE id = ?",
                (request.id,),
            ).fetchone()
            if not crow:
                raise CodeContextError(
                    code="RECORD_NOT_FOUND",
                    message=f"Change {request.id} not found.",
                    retryable=False,
                    details={"id": request.id},
                )
            c_oid = str(crow["commit_oid"])
            if cutoff_us is not None:
                c_meta = conn.execute(
                    "SELECT authored_at_us, committed_at_us FROM commits WHERE oid = ?",
                    (c_oid,),
                ).fetchone()
                if c_meta:
                    ct = c_meta["committed_at_us"] or c_meta["authored_at_us"]
                    if ct is None or int(ct) > cutoff_us:
                        raise CodeContextError(
                            code="RECORD_NOT_ELIGIBLE",
                            message="Change commit is outside the requested knowledge cutoff.",
                            retryable=False,
                            details={"id": request.id, "reason": "commit_after_cutoff"},
                        )

            old_p = (
                PathRef(
                    path=path_bytes_to_display(bytes(crow["old_path"]))[0],
                    path_bytes_b64=path_bytes_to_display(bytes(crow["old_path"]))[1],
                )
                if crow["old_path"] is not None
                else None
            )
            new_p = (
                PathRef(
                    path=path_bytes_to_display(bytes(crow["new_path"]))[0],
                    path_bytes_b64=path_bytes_to_display(bytes(crow["new_path"]))[1],
                )
                if crow["new_path"] is not None
                else None
            )
            record_obj = ChangeRecord(
                id=request.id,
                commit_oid=c_oid,
                parent_index=int(crow["parent_index"]),
                parent_oid=crow["parent_oid"],
                old_path=old_p,
                new_path=new_p,
                old_blob_oid=crow["old_blob_oid"],
                new_blob_oid=crow["new_blob_oid"],
                old_mode=crow["old_mode"],
                new_mode=crow["new_mode"],
                change_kind=crow["change_kind"],
                rename_score=crow["rename_score"],
                hunks_state=crow["hunks_state"],
            )
            rel_list = _load_relations_for_record(conn, "change", request.id)
            h_rows = conn.execute(
                "SELECT id FROM hunks WHERE change_id = ? ORDER BY ordinal ASC LIMIT 20",
                (request.id,),
            ).fetchall()
            for hr in h_rows:
                expand_list.append(Expand(id=str(hr["id"]), reason="Expand hunk patch"))

        elif rkind == "hunk":
            hrow = conn.execute("SELECT * FROM hunks WHERE id = ?", (request.id,)).fetchone()
            if not hrow:
                raise CodeContextError(
                    code="RECORD_NOT_FOUND",
                    message=f"Hunk {request.id} not found.",
                    retryable=False,
                    details={"id": request.id},
                )
            if cutoff_us is not None:
                c_meta = conn.execute(
                    "SELECT cm.authored_at_us, cm.committed_at_us FROM changes ch JOIN commits cm ON cm.oid = ch.commit_oid WHERE ch.id = ?",
                    (str(hrow["change_id"]),),
                ).fetchone()
                if c_meta:
                    ct = c_meta["committed_at_us"] or c_meta["authored_at_us"]
                    if ct is None or int(ct) > cutoff_us:
                        raise CodeContextError(
                            code="RECORD_NOT_ELIGIBLE",
                            message="Hunk commit is outside the requested knowledge cutoff.",
                            retryable=False,
                            details={"id": request.id, "reason": "commit_after_cutoff"},
                        )
            record_obj = HunkRecord(
                id=request.id,
                change_id=str(hrow["change_id"]),
                ordinal=int(hrow["ordinal"]),
                old_start=int(hrow["old_start"]),
                old_count=int(hrow["old_count"]),
                new_start=int(hrow["new_start"]),
                new_count=int(hrow["new_count"]),
            )
            if hrow["patch_text"]:
                raw_content_field = "patch"
                raw_content_bytes = str(hrow["patch_text"]).encode("utf-8")
            expand_list.append(Expand(id=str(hrow["change_id"]), reason="Expand parent change record"))

        elif rkind == "source":
            src_row = conn.execute("SELECT * FROM sources WHERE id = ?", (request.id,)).fetchone()
            if not src_row or src_row["availability"] in ("redacted", "deleted"):
                raise CodeContextError(
                    code="RECORD_NOT_FOUND",
                    message=f"Source {request.id} not found.",
                    retryable=False,
                    details={"id": request.id},
                )
            if src_row["availability"] == "inaccessible":
                raise CodeContextError(
                    code="ACCESS_BLOCKED",
                    message=f"Source {request.id} is inaccessible.",
                    retryable=False,
                    details={"id": request.id},
                )
            sel_ver_id, had_unk = _select_eligible_version_for_source(conn, request.id, cutoff_us)
            if cutoff_us is not None and sel_ver_id is None:
                raise CodeContextError(
                    code="RECORD_NOT_ELIGIBLE",
                    message="No stored version of this source satisfies knowledge_cutoff.",
                    retryable=False,
                    details={
                        "id": request.id,
                        "reason": "source_time_unknown" if had_unk else "source_after_cutoff",
                    },
                )
            record_obj = SourceRecord(
                id=request.id,
                source_kind=src_row["kind"],
                provider="github",
                repository_provider_id=str(src_row["repository_provider_id"]),
                provider_id=str(src_row["provider_id"]),
                parent_source_id=src_row["parent_source_id"],
                selected_version_id=sel_ver_id,
                availability=src_row["availability"],
                last_seen_at=us_to_iso_z(int(src_row["last_seen_at_us"])),
                last_reconciled_at=(
                    us_to_iso_z(int(src_row["last_reconciled_at_us"]))
                    if src_row["last_reconciled_at_us"] is not None
                    else None
                ),
            )
            rel_list = _load_relations_for_record(conn, "source", request.id)
            if sel_ver_id is not None:
                expand_list.append(
                    Expand(id=sel_ver_id, reason="Expand selected source version body")
                )

        elif rkind == "source_version":
            vrow = conn.execute(
                """
                SELECT sv.*, s.current_version_id, s.availability
                FROM source_versions sv
                JOIN sources s ON s.id = sv.source_id
                WHERE sv.id = ?
                """,
                (request.id,),
            ).fetchone()
            if not vrow or vrow["availability"] in ("redacted", "deleted"):
                raise CodeContextError(
                    code="RECORD_NOT_FOUND",
                    message=f"Source version {request.id} not found.",
                    retryable=False,
                    details={"id": request.id},
                )
            if vrow["availability"] == "inaccessible":
                raise CodeContextError(
                    code="ACCESS_BLOCKED",
                    message=f"Source version {request.id} is inaccessible.",
                    retryable=False,
                    details={"id": request.id},
                )
            ok, reason = _check_source_version_cutoff_eligible(conn, request.id, cutoff_us)
            if not ok:
                raise CodeContextError(
                    code="RECORD_NOT_ELIGIBLE",
                    message="This record is outside the requested knowledge cutoff.",
                    retryable=False,
                    details={"id": request.id, "reason": reason or "cutoff_violation"},
                )

            author_info = json.loads(str(vrow["author_json"])) if vrow["author_json"] else {}
            body_b = str(vrow["body"]).encode("utf-8") if vrow["body"] is not None else b""
            raw_content_field = "body"
            raw_content_bytes = body_b

            record_obj = SourceVersionRecord(
                id=request.id,
                source_id=str(vrow["source_id"]),
                title=vrow["title"][:4000] if vrow["title"] else None,
                attribution=Attribution(
                    provider="github",
                    actor_id=author_info.get("actor_id"),
                    display_name=author_info.get("display_name"),
                    source_url=vrow["source_url"],
                ),
                times=SourceTimes(
                    authored_at=us_to_iso_z(int(vrow["authored_at_us"])) if vrow["authored_at_us"] is not None else None,
                    updated_at=us_to_iso_z(int(vrow["updated_at_us"])) if vrow["updated_at_us"] is not None else None,
                    first_observed_at=us_to_iso_z(int(vrow["first_observed_at_us"])),
                    last_observed_at=us_to_iso_z(int(vrow["last_observed_at_us"])),
                ),
                content_sha256=str(vrow["content_sha256"]),
                body_bytes=len(body_b),
                current=(str(vrow["current_version_id"]) == request.id),
            )

        elif rkind == "evidence":
            erow = conn.execute("SELECT * FROM evidence WHERE id = ?", (request.id,)).fetchone()
            if not erow:
                raise CodeContextError(
                    code="RECORD_NOT_FOUND",
                    message=f"Evidence {request.id} not found.",
                    retryable=False,
                    details={"id": request.id},
                )
            ev_view = build_evidence_view_from_row(
                conn,
                mirror,
                erow,
                ancestor_commits=ancestor_commits,
                cutoff_us=cutoff_us,
                check_cutoff=True,
            )
            record_obj = EvidenceRecord(id=request.id, evidence=ev_view)
            # Also serve full underlying slice for pagination when needed
            anchor = ev_view.ref.anchor
            if anchor.kind == "discussion" and anchor.source_version_id:
                sv_b = conn.execute(
                    "SELECT body FROM source_versions WHERE id = ?",
                    (anchor.source_version_id,),
                ).fetchone()
                if sv_b and sv_b["body"] is not None:
                    full_b = str(sv_b["body"]).encode("utf-8")
                    raw_content_field = "body"
                    raw_content_bytes = full_b[anchor.span.byte_start : anchor.span.byte_end]
            elif anchor.kind == "hunk" and anchor.underlying_record_id:
                hk_b = conn.execute(
                    "SELECT patch_text FROM hunks WHERE id = ?",
                    (anchor.underlying_record_id,),
                ).fetchone()
                if hk_b and hk_b["patch_text"] is not None:
                    raw_content_field = "patch"
                    raw_content_bytes = str(hk_b["patch_text"]).encode("utf-8")
            elif anchor.git_oid:
                with mirror.open_batch_session() as batch:
                    gobj = batch.read_object(anchor.git_oid)
                if gobj is not None and b"\x00" not in gobj[2]:
                    raw_content_field = "commit_message" if anchor.kind == "commit" else "code"
                    raw_content_bytes = gobj[2][anchor.span.byte_start : anchor.span.byte_end]

        elif rkind == "claim":
            cl_row = conn.execute("SELECT * FROM claims WHERE id = ?", (request.id,)).fetchone()
            if not cl_row or cl_row["state"] == "retracted":
                raise CodeContextError(
                    code="RECORD_NOT_FOUND",
                    message=f"Claim {request.id} not found.",
                    retryable=False,
                    details={"id": request.id},
                )
            ok, reason = check_derivation_cutoff_eligible(
                conn, "claim", request.id, cutoff_us, ancestor_commits=None
            )
            if not ok:
                raise CodeContextError(
                    code="RECORD_NOT_ELIGIBLE",
                    message="Claim dependencies violate the requested knowledge cutoff.",
                    retryable=False,
                    details={"id": request.id, "reason": reason or "cutoff_violation"},
                )
            if cl_row["state"] == "stale":
                request_gaps.append(
                    Gap(
                        id=None,
                        code="DERIVED_ENRICHMENT_PENDING",
                        scope=f"claim:{request.id}",
                        message="This claim is stale due to an underlying source or mapping update; re-run enrich before treating as current.",
                        retryable=True,
                        record_id=request.id,
                    )
                )
            dec_view, dec_evs = build_decision_view_for_claim(
                conn,
                mirror,
                cl_row,
                ancestor_commits=ancestor_commits,
                cutoff_us=cutoff_us,
                target_mapping="exact",
                target_applicability="historical_only",
            )
            ev_list = dec_evs
            record_obj = ClaimRecord(
                id=request.id,
                episode_id=str(cl_row["episode_id"]),
                state=cl_row["state"],
                model_run_id=str(cl_row["model_run_id"]),
                created_at=us_to_iso_z(int(cl_row["created_at_us"])),
                decision=dec_view,
            )
            rel_list = _load_relations_for_record(conn, "claim", request.id)

        elif rkind == "summary":
            sum_row = conn.execute("SELECT * FROM summaries WHERE id = ?", (request.id,)).fetchone()
            if not sum_row or sum_row["state"] == "retracted":
                raise CodeContextError(
                    code="RECORD_NOT_FOUND",
                    message=f"Summary {request.id} not found.",
                    retryable=False,
                    details={"id": request.id},
                )
            ok, reason = check_derivation_cutoff_eligible(
                conn, "summary", request.id, cutoff_us, ancestor_commits=None
            )
            if not ok:
                raise CodeContextError(
                    code="RECORD_NOT_ELIGIBLE",
                    message="Summary dependencies violate the requested knowledge cutoff.",
                    retryable=False,
                    details={"id": request.id, "reason": reason or "cutoff_violation"},
                )
            if sum_row["state"] == "stale":
                request_gaps.append(
                    Gap(
                        id=None,
                        code="DERIVED_ENRICHMENT_PENDING",
                        scope=f"summary:{request.id}",
                        message="This summary is stale due to an underlying dependency update.",
                        retryable=True,
                        record_id=request.id,
                    )
                )
            raw_stmts = json.loads(str(sum_row["statements_json"]))
            cov_data = json.loads(str(sum_row["coverage_json"]))
            stmt_objs: list[SummaryStatement] = []
            seen_ev: dict[str, EvidenceView] = {}
            for st_item in raw_stmts[byte_offset:]:
                st_refs: list[EvidenceRef] = []
                for ev_id_str in st_item.get("evidence_ids", []):
                    erow = conn.execute("SELECT * FROM evidence WHERE id = ?", (ev_id_str,)).fetchone()
                    if erow:
                        ev_v = build_evidence_view_from_row(
                            conn,
                            mirror,
                            erow,
                            ancestor_commits=ancestor_commits,
                            cutoff_us=cutoff_us,
                        )
                        seen_ev[ev_v.ref.id] = ev_v
                        st_refs.append(ev_v.ref)
                if st_refs:
                    stmt_objs.append(SummaryStatement(text=st_item["text"], evidence_refs=st_refs))

            ev_list = list(seen_ev.values())
            record_obj = SummaryRecord(
                id=request.id,
                scope_kind=sum_row["scope_kind"],
                scope_id=str(sum_row["scope_id"]),
                revision=str(sum_row["revision"]),
                state=sum_row["state"],
                input_fingerprint=str(sum_row["input_fingerprint"]),
                partial=bool(cov_data.get("partial", False)),
                statements=stmt_objs[:40],
            )

        elif rkind == "gap":
            grow = conn.execute("SELECT * FROM coverage_gaps WHERE id = ?", (request.id,)).fetchone()
            if not grow:
                raise CodeContextError(
                    code="RECORD_NOT_FOUND",
                    message=f"Gap {request.id} not found.",
                    retryable=False,
                    details={"id": request.id},
                )
            # Compute next_gap_id in (scope, code, id) order among active gaps
            next_row = conn.execute(
                """
                SELECT id FROM coverage_gaps
                WHERE active = 1 AND (
                    scope > ?
                    OR (scope = ? AND code > ?)
                    OR (scope = ? AND code = ? AND id > ?)
                )
                ORDER BY scope ASC, code ASC, id ASC
                LIMIT 1
                """,
                (
                    grow["scope"],
                    grow["scope"],
                    grow["code"],
                    grow["scope"],
                    grow["code"],
                    grow["id"],
                ),
            ).fetchone()
            next_gap_id = str(next_row["id"]) if next_row else None
            if next_gap_id:
                expand_list.append(Expand(id=next_gap_id, reason="Inspect next active coverage gap"))

            record_obj = GapRecord(
                id=request.id,
                gap=Gap(
                    id=request.id,
                    code=str(grow["code"]),
                    scope=str(grow["scope"]),
                    message=str(grow["message"]),
                    retryable=bool(grow["retryable"]),
                    record_id=grow["record_id"],
                ),
                active=bool(grow["active"]),
                first_seen_at=us_to_iso_z(int(grow["first_seen_at_us"])),
                last_seen_at=us_to_iso_z(int(grow["last_seen_at_us"])),
                next_gap_id=next_gap_id,
            )

        else:
            raise CodeContextError(
                code="RECORD_NOT_FOUND",
                message=f"Unknown record kind for {request.id}",
                retryable=False,
            )

        cov, persisted_gaps, omitted_persisted_gaps = Store.build_coverage_and_gaps(
            conn, max_detailed_gaps=max(0, 20 - len(request_gaps))
        )
        all_gaps = [*request_gaps, *persisted_gaps][:20]

        # Preflight mandatory minimal envelope (record without optional content/relations)
        minimal_record = (
            record_obj.model_copy(update={"decision": None})
            if isinstance(record_obj, ClaimRecord)
            else (
                record_obj.model_copy(update={"statements": []})
                if isinstance(record_obj, SummaryRecord)
                else record_obj
            )
        )
        min_page = RecordPage(
            repository_id=snap.repository_id,
            revision=snap.revision,
            generation=snap.generation,
            access_epoch=snap.access_epoch,
            coverage=cov,
            gaps=all_gaps[:3],
            record=minimal_record,
            content=[],
            evidence=[],
            relations=[],
            next_cursor=None,
            budget=Budget(
                requested_tokens=request.max_tokens,
                used_tokens=request.max_tokens,
                estimator=estimator_id,
                count_kind="tokenizer" if count_kind == "tokenizer" else "estimated",
                truncated=True,
                omitted_counts=OmittedCounts(),
            ),
            expand=[],
        )
        min_tokens = int(token_counter(min_page.model_dump_json())) + 25
        if min_tokens > request.max_tokens:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message=f"max_tokens ({request.max_tokens}) is smaller than the minimum required envelope ({min_tokens} tokens).",
                retryable=False,
                details={"minimum_required_tokens": min_tokens, "requested_tokens": request.max_tokens},
            )

        # For ClaimRecord, verify the complete decision + evidence bundle fits on this page
        if isinstance(record_obj, ClaimRecord) and record_obj.decision is not None:
            claim_check_page = min_page.model_copy(
                update={"record": record_obj, "evidence": ev_list}
            )
            claim_req_tokens = int(token_counter(claim_check_page.model_dump_json())) + 25
            if claim_req_tokens > request.max_tokens:
                raise CodeContextError(
                    code="INVALID_REQUEST",
                    message=f"max_tokens ({request.max_tokens}) is too small to fit the claim's required evidence bundle ({claim_req_tokens} tokens).",
                    retryable=False,
                    details={"minimum_required_tokens": claim_req_tokens, "requested_tokens": request.max_tokens},
                )

        # Fit content slice and relations within request.max_tokens
        total_raw_bytes = len(raw_content_bytes)
        remaining_rels = rel_list[rel_offset:]

        # Estimate available bytes for content slice
        avail_tokens = max(0, request.max_tokens - min_tokens)
        candidate_slice_bytes = min(32768, max(256, avail_tokens * 2)) if raw_content_field and byte_offset < total_raw_bytes else 0
        admitted_rels_count = min(100, len(remaining_rels))

        while True:
            content_slices: list[ContentSlice] = []
            next_byte_off = total_raw_bytes
            if raw_content_field is not None and byte_offset < total_raw_bytes and candidate_slice_bytes > 0:
                slice_text, slice_span, next_byte_off = _slice_utf8_bytes_for_page(
                    raw_content_bytes, byte_offset, candidate_slice_bytes
                )
                content_slices.append(
                    ContentSlice(
                        field=raw_content_field,  # type: ignore[arg-type]
                        span=slice_span,
                        text=slice_text,
                        total_bytes=total_raw_bytes,
                    )
                )

            cur_rels = remaining_rels[:admitted_rels_count]
            next_rel_off = rel_offset + len(cur_rels)

            omitted_bytes = max(0, total_raw_bytes - next_byte_off) if raw_content_field else 0
            omitted_rels = max(0, len(rel_list) - next_rel_off)

            has_more = (omitted_bytes > 0) or (omitted_rels > 0)
            next_cur = (
                encode_cursor(
                    query_fingerprint=q_fp,
                    generation=snap.generation,
                    access_epoch=snap.access_epoch,
                    offset=next_byte_off,
                    extra={"rel_offset": next_rel_off},
                )
                if has_more
                else None
            )
            is_truncated = has_more or (omitted_persisted_gaps > 0)
            omitted_obj = OmittedCounts(
                content_bytes=omitted_bytes,
                relations=omitted_rels,
                gaps=omitted_persisted_gaps,
            )

            used_tokens = 0
            for _ in range(3):
                page = RecordPage(
                    repository_id=snap.repository_id,
                    revision=snap.revision,
                    generation=snap.generation,
                    access_epoch=snap.access_epoch,
                    coverage=cov,
                    gaps=all_gaps,
                    record=record_obj,
                    content=content_slices,
                    evidence=ev_list[:100],
                    relations=cur_rels,
                    next_cursor=next_cur,
                    budget=Budget(
                        requested_tokens=request.max_tokens,
                        used_tokens=used_tokens,
                        estimator=estimator_id,
                        count_kind="tokenizer" if count_kind == "tokenizer" else "estimated",
                        truncated=is_truncated,
                        omitted_counts=omitted_obj,
                    ),
                    expand=expand_list[:20],
                )
                new_used = int(token_counter(page.model_dump_json()))
                if new_used == used_tokens:
                    break
                used_tokens = new_used

            if used_tokens <= request.max_tokens:
                break

            if admitted_rels_count > 0:
                admitted_rels_count = max(0, admitted_rels_count // 2)
                continue
            if candidate_slice_bytes > 256:
                candidate_slice_bytes = max(256, candidate_slice_bytes // 2)
                continue
            if len(all_gaps) > 3:
                all_gaps = all_gaps[:3]
                continue
            break

        expected_epoch = snap.access_epoch

    if pre_authorize_hook is not None:
        pre_authorize_hook()
    store.verify_access_epoch_and_authorize(expected_epoch)
    return page


@dataclass(frozen=True)
class _ResolvedTargetInfo:
    resolved_target: ResolvedTarget | None
    path_bytes: bytes | None
    blob_oid: str | None
    blob_bytes: bytes | None
    extraction_id: str | None
    symbol_row: sqlite3.Row | None
    is_untracked_workspace_target: bool = False


def _resolve_context_target(
    conn: sqlite3.Connection,
    mirror: GitMirror,
    request: ContextRequest,
    *,
    active_rev: str,
    workspace_path: Path | None,
) -> _ResolvedTargetInfo:
    if request.target is None:
        return _ResolvedTargetInfo(
            resolved_target=None,
            path_bytes=None,
            blob_oid=None,
            blob_bytes=None,
            extraction_id=None,
            symbol_row=None,
        )

    if isinstance(request.target, SymbolTarget):
        sym_id = request.target.id
        s_rows = conn.execute(
            """
            SELECT s.id, s.extraction_id, s.kind, s.name, s.qualified_name,
                   s.byte_start, s.byte_end, s.line_start, s.line_end, s.signature,
                   sf.path, sf.blob_oid
            FROM symbols s
            JOIN snapshot_files sf ON sf.extraction_id = s.extraction_id
            WHERE s.id = ?
            ORDER BY sf.path ASC
            """,
            (sym_id,),
        ).fetchall()
        if not s_rows:
            raise CodeContextError(
                code="RECORD_NOT_FOUND",
                message=f"Symbol target {sym_id} not found in the active prepared snapshot.",
                retryable=False,
                details={"id": sym_id},
            )
        if len(s_rows) > 1:
            candidates = [path_bytes_to_display(bytes(r["path"]))[0] for r in s_rows[:10]]
            raise CodeContextError(
                code="AMBIGUOUS_TARGET",
                message=f"Symbol {sym_id} occurs in {len(s_rows)} active paths; supply an explicit path target.",
                retryable=False,
                details={"candidates": candidates, "symbol_id": sym_id},
            )
        srow = s_rows[0]
        p_bytes = bytes(srow["path"])
        b_oid = str(srow["blob_oid"])
        disp_p, b64_p = path_bytes_to_display(p_bytes)
        with mirror.open_batch_session() as batch:
            b_obj = batch.read_object(b_oid)
        b_bytes = b_obj[2] if b_obj is not None else None
        return _ResolvedTargetInfo(
            resolved_target=ResolvedTarget(
                kind="symbol",
                symbol_id=sym_id,
                path=PathRef(path=disp_p, path_bytes_b64=b64_p),
                revision=active_rev,
                blob_oid=b_oid,
                line=int(srow["line_start"]),
            ),
            path_bytes=p_bytes,
            blob_oid=b_oid,
            blob_bytes=b_bytes,
            extraction_id=str(srow["extraction_id"]),
            symbol_row=srow,
        )

    assert isinstance(request.target, PathTarget)
    p_bytes = request.target.raw_bytes()
    disp_p, b64_p = path_bytes_to_display(p_bytes)
    sf_row = conn.execute(
        "SELECT path, blob_oid, extraction_id, content_state FROM snapshot_files WHERE path = ?",
        (p_bytes,),
    ).fetchone()

    if sf_row is None:
        # Check if workspace is configured and this path is a verified untracked file
        if workspace_path is not None:
            ws_state, _ = inspect_workspace(
                mirror.store,
                workspace_path,
                active_revision=active_rev,
                target_path_bytes=p_bytes,
                index_blob_oid=None,
                index_blob_bytes=None,
            )
            if ws_state.target_state == "untracked":
                return _ResolvedTargetInfo(
                    resolved_target=None,
                    path_bytes=p_bytes,
                    blob_oid=None,
                    blob_bytes=None,
                    extraction_id=None,
                    symbol_row=None,
                    is_untracked_workspace_target=True,
                )
        raise CodeContextError(
            code="RECORD_NOT_FOUND",
            message=f"Target path {disp_p!r} not found in the active prepared snapshot.",
            retryable=False,
            details={"path": disp_p},
        )

    b_oid = str(sf_row["blob_oid"])
    ext_id = str(sf_row["extraction_id"]) if sf_row["extraction_id"] else None
    with mirror.open_batch_session() as batch:
        b_obj = batch.read_object(b_oid)
    b_bytes = b_obj[2] if b_obj is not None else None

    sym_row: sqlite3.Row | None = None
    sym_id_out: str | None = None
    if request.target.line is not None and ext_id is not None:
        req_line = request.target.line
        enclosing = conn.execute(
            """
            SELECT id, extraction_id, kind, name, qualified_name,
                   byte_start, byte_end, line_start, line_end, signature
            FROM symbols
            WHERE extraction_id = ? AND line_start <= ? AND line_end >= ?
            ORDER BY (byte_end - byte_start) ASC, id ASC
            """,
            (ext_id, req_line, req_line),
        ).fetchall()
        if enclosing:
            min_len = int(enclosing[0]["byte_end"]) - int(enclosing[0]["byte_start"])
            equal_smallest = [
                r
                for r in enclosing
                if (int(r["byte_end"]) - int(r["byte_start"])) == min_len
            ]
            if len(equal_smallest) > 1:
                raise CodeContextError(
                    code="AMBIGUOUS_TARGET",
                    message=f"Line {req_line} in {disp_p} is enclosed by multiple equal-sized symbols.",
                    retryable=False,
                    details={
                        "candidates": [str(r["id"]) for r in equal_smallest[:10]],
                        "line": req_line,
                        "path": disp_p,
                    },
                )
            sym_row = equal_smallest[0]
            sym_id_out = str(sym_row["id"])

    return _ResolvedTargetInfo(
        resolved_target=ResolvedTarget(
            kind="path",
            symbol_id=sym_id_out,
            path=PathRef(path=disp_p, path_bytes_b64=b64_p),
            revision=active_rev,
            blob_oid=b_oid,
            line=request.target.line,
        ),
        path_bytes=p_bytes,
        blob_oid=b_oid,
        blob_bytes=b_bytes,
        extraction_id=ext_id,
        symbol_row=sym_row,
    )


def execute_context(
    store: Store,
    mirror: GitMirror,
    request: ContextRequest,
    *,
    workspace_path: Path | None = None,
    token_counter: Callable[[str], int] = default_token_estimator,
    estimator_id: str = "utf8_bytes_div3_v1",
    count_kind: str = "estimated",
    exclude_enriched: bool = False,
    soft_deadline_seconds: float = 2.0,
    pre_authorize_hook: Callable[[], None] | None = None,
) -> ContextPack:
    """Assemble a bounded, cited `ContextPack` for `request` without invoking a model, parser, or network."""
    start_mono = time.monotonic()
    deadline_mono = start_mono + soft_deadline_seconds
    cutoff_us = datetime_to_us(request.knowledge_cutoff) if request.knowledge_cutoff else None

    with store.read_snapshot() as snap:
        conn = snap.conn
        repo_id = snap.repository_id
        active_rev = mirror.resolve_prepared_revision(conn, request.revision)

        ancestor_commits = {
            str(r["commit_oid"])
            for r in conn.execute("SELECT commit_oid FROM snapshot_commits").fetchall()
        }

        target_info = _resolve_context_target(
            conn,
            mirror,
            request,
            active_rev=active_rev,
            workspace_path=workspace_path,
        )
        expected_epoch = snap.access_epoch

    # Inspect workspace outside SQLite snapshot transaction per Storage §4 Read snapshot recipe
    ws_state, ws_gaps = inspect_workspace(
        store,
        workspace_path,
        active_revision=active_rev,
        target_path_bytes=target_info.path_bytes,
        index_blob_oid=target_info.blob_oid,
        index_blob_bytes=target_info.blob_bytes,
    )

    is_target_dirty = ws_state.target_state in (
        "staged_change",
        "unstaged_change",
        "staged_and_unstaged",
        "deleted",
        "untracked",
    ) or any(g.code == "WORKSPACE_TARGET_DIFFERS" for g in ws_gaps)

    # Re-enter read snapshot at the same generation/epoch for bounded expansion & packing
    with store.read_snapshot() as snap2:
        conn = snap2.conn
        if snap2.access_epoch != expected_epoch:
            raise CodeContextError(
                code="CURSOR_STALE",
                message="Access epoch changed during context read.",
                retryable=True,
                details={"reason": "access_epoch_changed"},
            )

        request_gaps: list[Gap] = list(ws_gaps)
        code_views: list[CodeView] = []
        decision_bundles: list[tuple[DecisionView, list[EvidenceView]]] = []
        raw_evidence_views: list[EvidenceView] = []
        expand_handles: list[Expand] = []
        seen_expand_ids: set[str] = set()

        def add_expand(rec_id: str, reason: str) -> None:
            if rec_id not in seen_expand_ids and is_valid_record_id(rec_id):
                seen_expand_ids.add(rec_id)
                expand_handles.append(Expand(id=rec_id, reason=reason))

        visited_records = 0
        relevant_changes_count = 0
        relevant_episodes_count = 0
        deadline_exceeded = False

        def check_bounds() -> bool:
            nonlocal deadline_exceeded
            if time.monotonic() >= deadline_mono:
                deadline_exceeded = True
                return False
            if visited_records >= 500 or relevant_changes_count >= 100 or relevant_episodes_count >= 25:
                return False
            return True

        # If target is untracked in workspace, return UNTRACKED_TARGET + search suggestions without invented lineage
        if target_info.is_untracked_workspace_target:
            search_res = execute_search(
                store,
                mirror,
                SearchRequest(
                    query=request.query,
                    revision=active_rev,
                    kind="all",
                    limit=5,
                    knowledge_cutoff=request.knowledge_cutoff,
                ),
                token_counter=token_counter,
                estimator_id=estimator_id,
                count_kind=count_kind,
                exclude_enriched=exclude_enriched,
            )
            for h in search_res.hits[:5]:
                add_expand(h.id, f"Search suggestion: {h.title[:100]}")

        # 1. Build target CodeView and structural 1-hop neighbors when target is in snapshot
        target_sym_ids: list[str] = []
        if target_info.resolved_target is not None and target_info.path_bytes is not None and target_info.blob_oid is not None:
            t_pref = target_info.resolved_target.path
            b_oid = target_info.blob_oid
            b_bytes = target_info.blob_bytes or b""
            lang = detect_language_for_path(target_info.path_bytes)

            if target_info.symbol_row is not None:
                srow = target_info.symbol_row
                sid = str(srow["id"])
                target_sym_ids.append(sid)
                visited_records += 1
                b_s = int(srow["byte_start"])
                b_e = int(srow["byte_end"])
                l_s = int(srow["line_start"])
                l_e = int(srow["line_end"])
                sym_slice = b_bytes[b_s:b_e]
                sym_text = sym_slice.decode("utf-8", errors="replace")[:16384]
                c_anchor_dict = {
                    "git_oid": b_oid,
                    "kind": "code",
                    "original_review_commit": None,
                    "path": t_pref.model_dump(),
                    "revision": active_rev,
                    "source_version_id": None,
                    "span": {
                        "byte_end": b_e,
                        "byte_start": b_s,
                        "line_end": l_e,
                        "line_start": l_s,
                    },
                    "underlying_record_id": sid,
                }
                c_ev_id = make_ev_id(
                    repository_id=repo_id,
                    kind="code",
                    source_version_id=None,
                    git_oid=b_oid,
                    anchor=c_anchor_dict,
                    content_sha256=sha256_bytes(sym_slice),
                )
                code_views.append(
                    CodeView(
                        evidence_id=c_ev_id,
                        symbol_id=sid,
                        anchor=Anchor.model_validate(c_anchor_dict),
                        language=lang,
                        signature=str(srow["signature"]),
                        text=sym_text,
                        mapping="exact",
                    )
                )
                add_expand(sid, f"Expand symbol {srow['qualified_name']}")

                # One structural step: containment/calls_candidate neighbors in same blob
                if check_bounds():
                    neighbor_rows = conn.execute(
                        """
                        SELECT s.id, s.qualified_name, s.signature, s.byte_start, s.byte_end, s.line_start, s.line_end, r.mapping
                        FROM relations r
                        JOIN symbols s ON (s.id = r.to_id OR s.id = r.from_id)
                        WHERE (r.from_id = ? OR r.to_id = ?)
                          AND r.relation_kind IN ('contains', 'calls_candidate', 'imports_candidate')
                          AND s.id != ?
                        ORDER BY s.byte_start ASC, s.id ASC
                        LIMIT 5
                        """,
                        (sid, sid, sid),
                    ).fetchall()
                    for nrow in neighbor_rows:
                        if not check_bounds():
                            break
                        visited_records += 1
                        nsid = str(nrow["id"])
                        nb_s = int(nrow["byte_start"])
                        nb_e = int(nrow["byte_end"])
                        nslice = b_bytes[nb_s:nb_e]
                        n_anchor_dict = {
                            "git_oid": b_oid,
                            "kind": "code",
                            "original_review_commit": None,
                            "path": t_pref.model_dump(),
                            "revision": active_rev,
                            "source_version_id": None,
                            "span": {
                                "byte_end": nb_e,
                                "byte_start": nb_s,
                                "line_end": int(nrow["line_end"]),
                                "line_start": int(nrow["line_start"]),
                            },
                            "underlying_record_id": nsid,
                        }
                        n_ev_id = make_ev_id(
                            repository_id=repo_id,
                            kind="code",
                            source_version_id=None,
                            git_oid=b_oid,
                            anchor=n_anchor_dict,
                            content_sha256=sha256_bytes(nslice),
                        )
                        code_views.append(
                            CodeView(
                                evidence_id=n_ev_id,
                                symbol_id=nsid,
                                anchor=Anchor.model_validate(n_anchor_dict),
                                language=lang,
                                signature=str(nrow["signature"]),
                                text=nslice.decode("utf-8", errors="replace")[:16384],
                                mapping=nrow["mapping"],
                            )
                        )
            else:
                # File-level target
                visited_records += 1
                if target_info.extraction_id:
                    file_syms = conn.execute(
                        "SELECT id FROM symbols WHERE extraction_id = ? ORDER BY byte_start ASC LIMIT 20",
                        (target_info.extraction_id,),
                    ).fetchall()
                    target_sym_ids.extend(str(r["id"]) for r in file_syms)

                f_slice = b_bytes[:16384]
                fl_s, fl_e = byte_span_to_lines(b_bytes, 0, len(f_slice))
                f_anchor_dict = {
                    "git_oid": b_oid,
                    "kind": "code",
                    "original_review_commit": None,
                    "path": t_pref.model_dump(),
                    "revision": active_rev,
                    "source_version_id": None,
                    "span": {
                        "byte_end": len(f_slice),
                        "byte_start": 0,
                        "line_end": fl_e,
                        "line_start": fl_s,
                    },
                    "underlying_record_id": None,
                }
                f_ev_id = make_ev_id(
                    repository_id=repo_id,
                    kind="code",
                    source_version_id=None,
                    git_oid=b_oid,
                    anchor=f_anchor_dict,
                    content_sha256=sha256_bytes(f_slice),
                )
                code_views.append(
                    CodeView(
                        evidence_id=f_ev_id,
                        symbol_id=None,
                        anchor=Anchor.model_validate(f_anchor_dict),
                        language=lang,
                        signature=None,
                        text=f_slice.decode("utf-8", errors="replace"),
                        mapping="exact",
                    )
                )

        # 2. Traverse relevant changes -> commits/PR episodes -> original reviews/issues -> claims
        relevant_commit_oids: list[str] = []
        seen_commits: set[str] = set()
        candidate_mapping_commits: set[str] = set()
        had_ambiguous_mapping = False

        if not target_info.is_untracked_workspace_target and check_bounds():
            if target_info.symbol_row is not None:
                # Find changes touching this symbol qualified_name on this path or predecessor paths
                sym_qname = str(target_info.symbol_row["qualified_name"])
                chg_rows = conn.execute(
                    """
                    SELECT DISTINCT c.id AS chg_id, c.commit_oid, c.new_path, c.old_path,
                                    cm.committed_at_us, cm.authored_at_us
                    FROM changes c
                    JOIN commits cm ON cm.oid = c.commit_oid
                    JOIN snapshot_commits sc ON sc.commit_oid = c.commit_oid
                    LEFT JOIN relations r ON r.from_kind = 'change' AND r.from_id = c.id AND r.relation_kind = 'changes'
                    LEFT JOIN symbols s ON s.id = r.to_id
                    WHERE (s.qualified_name = ? OR c.new_path = ? OR c.old_path = ?)
                    ORDER BY COALESCE(cm.committed_at_us, cm.authored_at_us, 0) DESC, c.commit_oid ASC
                    LIMIT 100
                    """,
                    (sym_qname, target_info.path_bytes, target_info.path_bytes),
                ).fetchall()
            elif target_info.path_bytes is not None:
                chg_rows = conn.execute(
                    """
                    SELECT DISTINCT c.id AS chg_id, c.commit_oid, c.new_path, c.old_path,
                                    cm.committed_at_us, cm.authored_at_us
                    FROM changes c
                    JOIN commits cm ON cm.oid = c.commit_oid
                    JOIN snapshot_commits sc ON sc.commit_oid = c.commit_oid
                    WHERE c.new_path = ? OR c.old_path = ?
                    ORDER BY COALESCE(cm.committed_at_us, cm.authored_at_us, 0) DESC, c.commit_oid ASC
                    LIMIT 100
                    """,
                    (target_info.path_bytes, target_info.path_bytes),
                ).fetchall()
            else:
                # Untargeted query: start from top lexical search candidates
                s_res = execute_search(
                    store,
                    mirror,
                    SearchRequest(
                        query=request.query,
                        revision=active_rev,
                        kind="all",
                        limit=20,
                        knowledge_cutoff=request.knowledge_cutoff,
                    ),
                    token_counter=token_counter,
                    estimator_id=estimator_id,
                    count_kind=count_kind,
                    exclude_enriched=exclude_enriched,
                )
                for h in s_res.hits[:10]:
                    add_expand(h.id, f"Expand search hit: {h.title[:80]}")
                chg_rows = conn.execute(
                    """
                    SELECT c.id AS chg_id, c.commit_oid, c.new_path, c.old_path,
                           cm.committed_at_us, cm.authored_at_us
                    FROM changes c
                    JOIN commits cm ON cm.oid = c.commit_oid
                    JOIN snapshot_commits sc ON sc.commit_oid = c.commit_oid
                    ORDER BY COALESCE(cm.committed_at_us, cm.authored_at_us, 0) DESC, c.commit_oid ASC
                    LIMIT 25
                    """
                ).fetchall()

            for cr in chg_rows:
                if not check_bounds():
                    break
                c_oid = str(cr["commit_oid"])
                if cutoff_us is not None:
                    ct = cr["committed_at_us"] or cr["authored_at_us"]
                    if ct is None or int(ct) > cutoff_us:
                        continue
                relevant_changes_count += 1
                visited_records += 1
                add_expand(str(cr["chg_id"]), f"Expand change in commit {c_oid[:12]}")
                if c_oid not in seen_commits:
                    seen_commits.add(c_oid)
                    relevant_commit_oids.append(c_oid)

                # Attach hunk evidence for this change
                ev_rows = conn.execute(
                    """
                    SELECT e.* FROM evidence e
                    WHERE e.git_oid = ? AND e.kind IN ('hunk', 'commit')
                    ORDER BY e.kind DESC, e.id ASC LIMIT 3
                    """,
                    (c_oid,),
                ).fetchall()
                for er in ev_rows:
                    if not check_bounds():
                        break
                    try:
                        ev_v = build_evidence_view_from_row(
                            conn,
                            mirror,
                            er,
                            ancestor_commits=ancestor_commits,
                            cutoff_us=cutoff_us,
                        )
                        raw_evidence_views.append(ev_v)
                        visited_records += 1
                    except CodeContextError:
                        pass

        # 3. Discover PR/commit episodes for relevant commits (up to 25 episodes)
        episode_rows: list[sqlite3.Row] = []
        seen_episode_ids: set[str] = set()
        pr_source_ids: list[str] = []

        for c_oid in relevant_commit_oids:
            if not check_bounds() or relevant_episodes_count >= 25:
                break
            # Find associated PR sources via relations
            pr_rels = conn.execute(
                """
                SELECT to_id, mapping FROM relations
                WHERE from_kind = 'commit' AND from_id = ? AND to_kind = 'source' AND relation_kind IN ('associated_pr', 'landed_mapping')
                ORDER BY id ASC
                """,
                (c_oid,),
            ).fetchall()
            for pr_r in pr_rels:
                pr_sid = str(pr_r["to_id"])
                if pr_r["mapping"] == "candidate":
                    candidate_mapping_commits.add(c_oid)
                if pr_sid not in pr_source_ids:
                    pr_source_ids.append(pr_sid)
                ep_rows = conn.execute(
                    "SELECT * FROM episodes WHERE primary_source_id = ?",
                    (pr_sid,),
                ).fetchall()
                for epr in ep_rows:
                    epid = str(epr["id"])
                    if epid not in seen_episode_ids and relevant_episodes_count < 25:
                        seen_episode_ids.add(epid)
                        episode_rows.append(epr)
                        relevant_episodes_count += 1

            # Also check commit-scoped episodes
            c_eps = conn.execute(
                "SELECT * FROM episodes WHERE commit_oid = ?",
                (c_oid,),
            ).fetchall()
            for epr in c_eps:
                epid = str(epr["id"])
                if epid not in seen_episode_ids and relevant_episodes_count < 25:
                    seen_episode_ids.add(epid)
                    episode_rows.append(epr)
                    relevant_episodes_count += 1

        # If untargeted, also include any episodes matched by claims/sources
        if request.target is None and relevant_episodes_count < 25:
            extra_eps = conn.execute(
                "SELECT * FROM episodes ORDER BY id ASC LIMIT ?",
                (25 - relevant_episodes_count,),
            ).fetchall()
            for epr in extra_eps:
                epid = str(epr["id"])
                if epid not in seen_episode_ids:
                    seen_episode_ids.add(epid)
                    episode_rows.append(epr)
                    relevant_episodes_count += 1
                    if epr["primary_source_id"] and str(epr["primary_source_id"]) not in pr_source_ids:
                        pr_source_ids.append(str(epr["primary_source_id"]))

        # 4. Collect raw discussion evidence from associated PRs, reviews, comments, and linked issues
        had_unknown_source_time = False
        had_unavailable_source = False

        for pr_sid in pr_source_ids:
            if not check_bounds():
                break
            family_sources = conn.execute(
                """
                SELECT id, kind, availability FROM sources
                WHERE id = ? OR parent_source_id = ?
                   OR id IN (SELECT to_id FROM relations WHERE from_kind = 'source' AND from_id = ? AND relation_kind = 'references_issue')
                ORDER BY last_seen_at_us ASC, id ASC
                """,
                (pr_sid, pr_sid, pr_sid),
            ).fetchall()
            for fsrc in family_sources:
                if not check_bounds():
                    break
                fs_id = str(fsrc["id"])
                if fsrc["availability"] != "available":
                    had_unavailable_source = True
                    continue
                sel_ver, unk_t = _select_eligible_version_for_source(conn, fs_id, cutoff_us)
                if unk_t:
                    had_unknown_source_time = True
                if sel_ver is None:
                    continue
                add_expand(sel_ver, f"Expand {fsrc['kind']} discussion version")
                ev_rows = conn.execute(
                    "SELECT * FROM evidence WHERE source_version_id = ? ORDER BY id ASC LIMIT 5",
                    (sel_ver,),
                ).fetchall()
                for er in ev_rows:
                    try:
                        ev_v = build_evidence_view_from_row(
                            conn,
                            mirror,
                            er,
                            ancestor_commits=ancestor_commits,
                            cutoff_us=cutoff_us,
                        )
                        raw_evidence_views.append(ev_v)
                        visited_records += 1
                    except CodeContextError:
                        pass

        if had_unknown_source_time:
            request_gaps.append(
                Gap(
                    id=None,
                    code="SOURCE_TIME_UNKNOWN",
                    scope="context:cutoff",
                    message="One or more discussion versions with missing source timestamps were excluded under knowledge_cutoff.",
                    retryable=False,
                )
            )

        # 5. Collect extracted claims for relevant episodes (when not exclude_enriched)
        had_stale_claim = False
        if not exclude_enriched:
            for epr in episode_rows:
                if not check_bounds():
                    break
                epid = str(epr["id"])
                cl_rows = conn.execute(
                    "SELECT * FROM claims WHERE episode_id = ? ORDER BY created_at_us ASC, id ASC",
                    (epid,),
                ).fetchall()
                for cl_r in cl_rows:
                    if cl_r["state"] == "stale":
                        had_stale_claim = True
                        continue
                    if cl_r["state"] != "ready":
                        continue
                    ok, _ = check_derivation_cutoff_eligible(
                        conn, "claim", str(cl_r["id"]), cutoff_us, ancestor_commits
                    )
                    if not ok:
                        continue

                    # Determine target-specific mapping and applicability per Contracts §5 and Retrieval §4
                    claim_mapping = "exact"
                    claim_id_str = str(cl_r["id"])
                    target_rels = conn.execute(
                        """
                        SELECT r.to_id, r.mapping, r.details_json, e.anchor_json
                        FROM relations r
                        JOIN evidence e ON e.id = r.to_id
                        WHERE r.from_kind = 'claim' AND r.from_id = ? AND r.relation_kind = 'targets' AND r.state = 'active'
                        """,
                        (claim_id_str,),
                    ).fetchall()
                    if target_info.resolved_target is not None:
                        if not target_rels:
                            claim_mapping = "unmapped"
                        else:
                            # If caller targeted a specific symbol, check whether any target anchor of this claim matches that symbol or qualified_name
                            req_sym_id = target_info.resolved_target.symbol_id
                            req_path_str = target_info.resolved_target.path.path
                            req_qname = (
                                str(target_info.symbol_row["qualified_name"])
                                if target_info.symbol_row is not None
                                else None
                            )
                            matched_mappings: list[str] = []
                            for tr in target_rels:
                                det = json.loads(str(tr["details_json"])) if tr["details_json"] else {}
                                anc = json.loads(str(tr["anchor_json"])) if tr["anchor_json"] else {}
                                t_sym_id = det.get("symbol_id") or anc.get("underlying_record_id")
                                t_path_obj = anc.get("path") or {}
                                t_path_str = t_path_obj.get("path")
                                if req_sym_id is not None:
                                    if t_sym_id == req_sym_id:
                                        matched_mappings.append(str(tr["mapping"]))
                                    elif t_sym_id and req_qname:
                                        s_match = conn.execute(
                                            "SELECT qualified_name FROM symbols WHERE id = ?",
                                            (str(t_sym_id),),
                                        ).fetchone()
                                        if s_match and str(s_match["qualified_name"]) == req_qname:
                                            matched_mappings.append(
                                                "exact"
                                                if str(tr["mapping"]) == "exact"
                                                and epr["commit_oid"] not in candidate_mapping_commits
                                                else "candidate"
                                            )
                                else:
                                    # File-level target
                                    if t_path_str == req_path_str:
                                        matched_mappings.append(str(tr["mapping"]))

                            if not matched_mappings:
                                # Claim targets a different symbol/file in the same PR! Skip for this symbol target.
                                if req_sym_id is not None:
                                    continue
                                claim_mapping = "unmapped"
                            elif "exact" in matched_mappings:
                                claim_mapping = "exact"
                            elif "candidate" in matched_mappings:
                                claim_mapping = "candidate"
                                had_ambiguous_mapping = True
                            else:
                                claim_mapping = "unmapped"

                    # Compute conservative applicability: default `historical_only`;
                    # upgrade to `corroborated` ONLY if not dirty, mapping is exact, and an eligible applicability evidence record matches active_rev
                    claim_app = "historical_only"
                    if not is_target_dirty and claim_mapping == "exact":
                        corr_rows = conn.execute(
                            """
                            SELECT ce.evidence_id, ce.details_json, e.source_version_id, e.git_oid
                            FROM claim_evidence ce
                            JOIN evidence e ON e.id = ce.evidence_id
                            WHERE ce.claim_id = ? AND ce.role = 'applicability'
                            """,
                            (claim_id_str,),
                        ).fetchall()
                        for corr in corr_rows:
                            c_det = json.loads(str(corr["details_json"])) if corr["details_json"] else {}
                            if c_det.get("target_revision") == active_rev:
                                claim_app = "corroborated"
                                break

                    try:
                        d_view, d_evs = build_decision_view_for_claim(
                            conn,
                            mirror,
                            cl_r,
                            ancestor_commits=ancestor_commits,
                            cutoff_us=cutoff_us,
                            target_mapping=claim_mapping,
                            target_applicability=claim_app,
                        )
                        decision_bundles.append((d_view, d_evs))
                        add_expand(d_view.id, "Expand decision claim details")
                    except CodeContextError:
                        pass

        if had_stale_claim:
            request_gaps.append(
                Gap(
                    id=None,
                    code="DERIVED_ENRICHMENT_PENDING",
                    scope="context:claims",
                    message="Stale derived claims were excluded because underlying sources changed; re-run enrich.",
                    retryable=True,
                )
            )
        if had_ambiguous_mapping:
            request_gaps.append(
                Gap(
                    id=None,
                    code="MAPPING_AMBIGUOUS",
                    scope="context:lineage",
                    message="One or more historical anchors mapped with candidate correspondence.",
                    retryable=False,
                )
            )
        if deadline_exceeded:
            request_gaps.append(
                Gap(
                    id=None,
                    code="READ_BUDGET_EXHAUSTED",
                    scope="context:deadline",
                    message="Context expansion reached the 2-second soft read deadline; returning bounded partial results.",
                    retryable=True,
                )
            )

        # 6. Build deterministic structural Overview (or stored summary in T19)
        snap_file_cnt = int(conn.execute("SELECT COUNT(*) FROM snapshot_files").fetchone()[0])
        overview_obj, overview_evs = _build_context_overview(
            conn,
            mirror,
            active_rev=active_rev,
            target_info=target_info,
            snap_file_cnt=snap_file_cnt,
            included_claims_cnt=len(decision_bundles),
            omitted_claims_cnt=0,
            cutoff_us=cutoff_us,
            ancestor_commits=ancestor_commits,
            exclude_enriched=exclude_enriched,
        )

        cov, persisted_gaps, omitted_persisted_gaps = Store.build_coverage_and_gaps(
            conn, max_detailed_gaps=max(0, 20 - len(request_gaps))
        )
        all_gaps = [*request_gaps, *persisted_gaps][:20]

        # 7. Preflight mandatory minimal envelope
        min_pack = ContextPack(
            repository_id=repo_id,
            revision=active_rev,
            generation=snap2.generation,
            access_epoch=snap2.access_epoch,
            coverage=cov,
            gaps=all_gaps[:3],
            query=request.query,
            target=target_info.resolved_target,
            workspace=ws_state,
            overview=None,
            code=[],
            decisions=[],
            evidence=[],
            rationale_status="evidence_incomplete",
            budget=Budget(
                requested_tokens=request.max_tokens,
                used_tokens=request.max_tokens,
                estimator=estimator_id,
                count_kind="tokenizer" if count_kind == "tokenizer" else "estimated",
                truncated=True,
                omitted_counts=OmittedCounts(),
            ),
            expand=[],
        )
        min_req_tokens = int(token_counter(min_pack.model_dump_json())) + 30
        if min_req_tokens > request.max_tokens:
            raise CodeContextError(
                code="INVALID_REQUEST",
                message=f"max_tokens ({request.max_tokens}) is smaller than the minimum required context envelope ({min_req_tokens} tokens).",
                retryable=False,
                details={"minimum_required_tokens": min_req_tokens, "requested_tokens": request.max_tokens},
            )

        # 8. Packing loop: admit target code, then claim+support+counterevidence bundles atomically, then raw evidence, then overview
        # Never emit a claim without its supporting and known counterevidence in the same pack!
        admitted_code = list(code_views[:20])
        admitted_decisions: list[DecisionView] = []
        admitted_ev_map: dict[str, EvidenceView] = {}
        omitted_decisions_count = 0
        conflicting_omitted = False

        for d_view, d_evs in decision_bundles[:20]:
            trial_decisions = [*admitted_decisions, d_view]
            trial_ev_map = dict(admitted_ev_map)
            for ev in d_evs:
                trial_ev_map[ev.ref.id] = ev
            if len(trial_ev_map) > 100:
                omitted_decisions_count += 1
                if d_view.counterevidence_refs:
                    conflicting_omitted = True
                add_expand(d_view.id, "Omitted decision bundle due to evidence limit")
                continue

            trial_pack = min_pack.model_copy(
                update={
                    "code": admitted_code[:1],
                    "decisions": trial_decisions,
                    "evidence": list(trial_ev_map.values()),
                    "gaps": all_gaps,
                }
            )
            if int(token_counter(trial_pack.model_dump_json())) + 40 <= request.max_tokens:
                admitted_decisions = trial_decisions
                admitted_ev_map = trial_ev_map
            else:
                omitted_decisions_count += 1
                if d_view.counterevidence_refs:
                    conflicting_omitted = True
                add_expand(d_view.id, "Expand omitted decision bundle")

        if conflicting_omitted:
            conflict_gap = Gap(
                id=None,
                code="CONFLICTING_EVIDENCE_OMITTED",
                scope="context:budget",
                message="A decision bundle with counterevidence could not fit within max_tokens and was omitted as a whole; use get on the expand handle.",
                retryable=False,
            )
            all_gaps = [conflict_gap, *all_gaps][:20]

        # Add overview evidence if stored_summary
        for o_ev in overview_evs:
            if o_ev.ref.id not in admitted_ev_map and len(admitted_ev_map) < 100:
                admitted_ev_map[o_ev.ref.id] = o_ev

        # Add raw episode evidence that fits while reserving room for overview
        omitted_ev_count = 0
        for rev_ev in raw_evidence_views:
            if rev_ev.ref.id in admitted_ev_map:
                continue
            if len(admitted_ev_map) >= 100:
                omitted_ev_count += 1
                add_expand(rev_ev.ref.id, "Expand omitted raw evidence")
                continue
            trial_ev_map = {**admitted_ev_map, rev_ev.ref.id: rev_ev}
            trial_pack = min_pack.model_copy(
                update={
                    "overview": overview_obj,
                    "code": admitted_code,
                    "decisions": admitted_decisions,
                    "evidence": list(trial_ev_map.values()),
                    "gaps": all_gaps,
                    "expand": expand_handles[:20],
                }
            )
            if int(token_counter(trial_pack.model_dump_json())) + 250 <= request.max_tokens:
                admitted_ev_map = trial_ev_map
            else:
                omitted_ev_count += 1
                add_expand(rev_ev.ref.id, "Expand omitted raw evidence")

        # Determine `rationale_status` per Contracts §5 and Retrieval §4
        rationale_status = _compute_rationale_status(
            conn,
            target_info=target_info,
            admitted_decisions=admitted_decisions,
            episode_rows=episode_rows,
            all_gaps=all_gaps,
            had_unavailable_source=had_unavailable_source,
            had_stale_claim=had_stale_claim,
            exclude_enriched=exclude_enriched,
        )

        # Trim optional neighbors / overview until canonical full JSON fits within `request.max_tokens`
        admitted_overview: Overview | None = overview_obj
        omitted_code_count = max(0, len(code_views) - len(admitted_code))

        while True:
            is_truncated = (
                omitted_code_count > 0
                or omitted_decisions_count > 0
                or omitted_ev_count > 0
                or omitted_persisted_gaps > 0
                or deadline_exceeded
            )
            omitted_obj = OmittedCounts(
                code=omitted_code_count,
                decisions=omitted_decisions_count,
                evidence=omitted_ev_count,
                gaps=omitted_persisted_gaps,
            )
            used_tokens = 0
            for _ in range(3):
                pack = ContextPack(
                    repository_id=repo_id,
                    revision=active_rev,
                    generation=snap2.generation,
                    access_epoch=snap2.access_epoch,
                    coverage=cov,
                    gaps=all_gaps,
                    query=request.query,
                    target=target_info.resolved_target,
                    workspace=ws_state,
                    overview=admitted_overview,
                    code=admitted_code,
                    decisions=admitted_decisions,
                    evidence=list(admitted_ev_map.values()),
                    rationale_status=rationale_status,  # type: ignore[arg-type]
                    budget=Budget(
                        requested_tokens=request.max_tokens,
                        used_tokens=used_tokens,
                        estimator=estimator_id,
                        count_kind="tokenizer" if count_kind == "tokenizer" else "estimated",
                        truncated=is_truncated,
                        omitted_counts=omitted_obj,
                    ),
                    expand=expand_handles[:20],
                )
                new_used = int(token_counter(pack.model_dump_json()))
                if new_used == used_tokens:
                    break
                used_tokens = new_used

            if used_tokens <= request.max_tokens:
                break

            if admitted_overview is not None:
                admitted_overview = None
                continue
            if len(admitted_code) > 1:
                admitted_code.pop()
                omitted_code_count += 1
                continue
            if len(admitted_code) == 1 and len(admitted_code[0].text) > 512:
                c0 = admitted_code[0]
                admitted_code[0] = c0.model_copy(update={"text": c0.text[: len(c0.text) // 2]})
                continue
            if len(admitted_code) == 1:
                admitted_code.pop()
                omitted_code_count += 1
                continue
            if len(all_gaps) > 3:
                omitted_persisted_gaps += len(all_gaps) - 3
                all_gaps = all_gaps[:3]
                continue
            break

    if pre_authorize_hook is not None:
        pre_authorize_hook()
    store.verify_access_epoch_and_authorize(expected_epoch)
    return pack


def _compute_rationale_status(
    conn: sqlite3.Connection,
    *,
    target_info: _ResolvedTargetInfo,
    admitted_decisions: list[DecisionView],
    episode_rows: list[sqlite3.Row],
    all_gaps: list[Gap],
    had_unavailable_source: bool,
    had_stale_claim: bool,
    exclude_enriched: bool,
) -> str:
    """Compute `rationale_status` per Contracts §5 and Retrieval §4:
    1. `supported` if at least one returned ready decision has `support='explicit'`, non-null `rationale`,
       and (when target is supplied) `mapping='exact'`.
    2. Otherwise, if collection/mapping gaps or untracked target exist -> `evidence_incomplete`.
    3. Otherwise, if episodes are uninspected or stale or exclude_enriched -> `enrichment_pending`.
    4. Only when every required segment of relevant collected episodes is completely inspected -> `undocumented_in_available_evidence`.
    """
    qualifying = [
        d
        for d in admitted_decisions
        if d.support == "explicit"
        and d.rationale is not None
        and d.state == "ready"
        and (target_info.resolved_target is None or d.mapping == "exact")
    ]
    if qualifying:
        return "supported"

    if target_info.is_untracked_workspace_target or had_unavailable_source:
        return "evidence_incomplete"

    incomplete_codes = {
        "SOURCE_UNAVAILABLE",
        "SOURCE_UNVERIFIED",
        "MISSING_GIT_OBJECT",
        "MISSING_PARENT",
        "SHALLOW_HISTORY",
        "MAPPING_UNAVAILABLE",
        "MAPPING_AMBIGUOUS",
        "REVIEW_COMPARISON_MISSING",
        "AUTH_REQUIRED",
        "ACCESS_BLOCKED",
    }
    if any(g.code in incomplete_codes for g in all_gaps):
        return "evidence_incomplete"

    if not episode_rows:
        # Check if git/github collection hasn't finished
        if any(g.code in ("GIT_HISTORY_PENDING", "GITHUB_BACKFILL_PENDING") for g in all_gaps):
            return "evidence_incomplete"
        unprocessed_commits = int(
            conn.execute(
                "SELECT COUNT(*) FROM commits c JOIN snapshot_commits sc ON sc.commit_oid = c.oid WHERE c.diff_state != 'complete'"
            ).fetchone()[0]
        )
        if unprocessed_commits > 0:
            return "evidence_incomplete"
        return "undocumented_in_available_evidence"

    if exclude_enriched or had_stale_claim:
        return "enrichment_pending"

    for epr in episode_rows:
        if epr["state"] != "inspected":
            return "enrichment_pending"
        ep_id = str(epr["id"])
        ep_fp = str(epr["input_fingerprint"])
        seg_rows = conn.execute(
            "SELECT required_slots_json, inspected_slots_json, unresolved_slots_json, state FROM episode_segments WHERE episode_id = ? AND input_fingerprint = ?",
            (ep_id, ep_fp),
        ).fetchall()
        if not seg_rows:
            return "enrichment_pending"
        for sr in seg_rows:
            if sr["state"] != "complete":
                return "enrichment_pending"
            req_slots = set(json.loads(str(sr["required_slots_json"])))
            insp_slots = set(json.loads(str(sr["inspected_slots_json"])))
            if not req_slots.issubset(insp_slots):
                return "enrichment_pending"

    return "undocumented_in_available_evidence"


def _build_context_overview(
    conn: sqlite3.Connection,
    mirror: GitMirror,
    *,
    active_rev: str,
    target_info: _ResolvedTargetInfo,
    snap_file_cnt: int,
    included_claims_cnt: int,
    omitted_claims_cnt: int,
    cutoff_us: int | None,
    ancestor_commits: set[str],
    exclude_enriched: bool,
) -> tuple[Overview, list[EvidenceView]]:
    if not exclude_enriched:
        sum_row = conn.execute(
            "SELECT * FROM summaries WHERE revision = ? AND scope_kind = 'repository' AND state = 'ready' ORDER BY id ASC LIMIT 1",
            (active_rev,),
        ).fetchone()
        if sum_row:
            ok, _ = check_derivation_cutoff_eligible(
                conn, "summary", str(sum_row["id"]), cutoff_us, ancestor_commits
            )
            if ok:
                raw_stmts = json.loads(str(sum_row["statements_json"]))
                cov_d = json.loads(str(sum_row["coverage_json"]))
                stmts_out: list[SummaryStatement] = []
                sum_evs: dict[str, EvidenceView] = {}
                for item in raw_stmts[:40]:
                    refs: list[EvidenceRef] = []
                    for ev_id_str in item.get("evidence_ids", []):
                        er = conn.execute("SELECT * FROM evidence WHERE id = ?", (ev_id_str,)).fetchone()
                        if er:
                            ev_v = build_evidence_view_from_row(
                                conn, mirror, er, ancestor_commits=ancestor_commits, cutoff_us=cutoff_us
                            )
                            sum_evs[ev_v.ref.id] = ev_v
                            refs.append(ev_v.ref)
                    if refs:
                        stmts_out.append(SummaryStatement(text=item["text"], evidence_refs=refs))
                if stmts_out:
                    return (
                        Overview(
                            kind="stored_summary",
                            summary_id=str(sum_row["id"]),
                            text="\n".join(s.text for s in stmts_out)[:16000],
                            statements=stmts_out,
                            file_count=int(cov_d.get("file_count", snap_file_cnt)),
                            group_count=int(cov_d.get("group_count", 1)),
                            included_file_count=int(cov_d.get("included_file_count", snap_file_cnt)),
                            included_group_count=int(cov_d.get("included_group_count", 1)),
                            included_claim_count=included_claims_cnt,
                            omitted_claim_count=omitted_claims_cnt,
                            partial=bool(cov_d.get("partial", False)),
                        ),
                        list(sum_evs.values()),
                    )

    if target_info.resolved_target is not None:
        t_desc = target_info.resolved_target.path.path
        if target_info.symbol_row is not None:
            t_desc = f"{target_info.symbol_row['qualified_name']} in {t_desc}"
        text = f"Target {t_desc} at prepared revision {active_rev[:12]} ({snap_file_cnt} tracked files)."
    else:
        text = f"Repository snapshot at prepared revision {active_rev[:12]} ({snap_file_cnt} tracked files)."

    return (
        Overview(
            kind="structural",
            summary_id=None,
            text=text,
            statements=[],
            file_count=snap_file_cnt,
            group_count=1,
            included_file_count=1 if target_info.resolved_target is not None else snap_file_cnt,
            included_group_count=1,
            included_claim_count=included_claims_cnt,
            omitted_claim_count=omitted_claims_cnt,
            partial=False,
        ),
        [],
    )

