import contextlib
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from codecontext.deadline import check_deadline
from codecontext.git import (
    CatFileBatchSession,
    GitMirror,
    ParsedHunk,
    RawChange,
    compute_unified_zero_hunks,
    parse_diff_tree_raw_z,
    parse_raw_commit_object,
    run_safe_git,
)
from codecontext.identity import (
    LineByteIndex,
    canonical_json,
    encode_path_b64,
    line_range_to_byte_span,
    make_chg_id,
    make_episode_id,
    make_ev_id,
    make_hunk_id,
    make_relation_id,
    make_src_id,
    path_bytes_to_display,
    sha256_bytes,
)
from codecontext.store import Store, now_us
from codecontext.syntax import (
    MAX_PARSE_BYTES,
    ExtractedSymbol,
    detect_language_for_path,
    index_blob_into_store,
    is_lfs_pointer,
    normalize_search_text,
)

# Commits touching more files than this are treated as tree-wide; focused history then follows only the tracked path.
WIDE_COMMIT_FILE_THRESHOLD = 32
REVERT_COMMIT_RE = re.compile(r"This reverts commit ([0-9a-f]{40}(?:[0-9a-f]{24})?)\b", re.IGNORECASE)
PR_TRAILER_RE = re.compile(
    r"(?:\(#(?P<squash_pr>[1-9]\d*)\)\s*$|Merge pull request #(?P<merge_pr>[1-9]\d*)\b|"
    r"See merge request [A-Za-z0-9_./-]*!(?P<gl_mr>[1-9]\d*)\b|"
    r"^(?:Pull-Request|PR):\s*#?(?P<hdr_pr>[1-9]\d*)\b)",
    re.IGNORECASE | re.MULTILINE,
)
ISSUE_TRAILER_RE = re.compile(
    r"\b(?:fix(?:es|ed)?|close(?:s|d)?|resolve(?:s|d)?|refs?|references?)\s+#(?P<num>[1-9]\d*)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SymbolChangeAssociation:
    symbol: ExtractedSymbol
    side: str  # 'old' or 'new'
    is_primary: bool
    hunk_id: str


@dataclass(frozen=True)
class AnchorMappingResult:
    mapping: str  # 'exact', 'candidate', or 'unmapped'
    method: str
    target_path: bytes | None = None
    target_blob_oid: str | None = None
    target_line_start: int | None = None
    target_line_end: int | None = None
    target_byte_start: int | None = None
    target_byte_end: int | None = None
    target_symbol_id: str | None = None
    transformations: list[dict[str, Any]] = field(default_factory=list)
    gap_code: str | None = None
    gap_message: str | None = None


def associate_hunk_side_with_symbols(
    blob_bytes: bytes,
    line_start: int,
    line_count: int,
    symbols: list[ExtractedSymbol],
    *,
    line_index: LineByteIndex | None = None,
) -> list[tuple[ExtractedSymbol, bool]]:
    """Map a hunk side (1-based line_start, line_count) against extracted symbols of that blob.
    Returns list of `(symbol, is_primary)`.
    """
    if not symbols or not blob_bytes:
        return []
    try:
        if line_index is not None:
            b_start, b_end, _, _ = line_index.line_range_to_byte_span(line_start, line_count)
        else:
            b_start, b_end, _, _ = line_range_to_byte_span(blob_bytes, line_start, line_count)
    except ValueError:
        return []

    matching: list[ExtractedSymbol] = []
    if line_count == 0:
        boundary = b_start
        for sym in symbols:
            if sym.body_byte_start is not None and sym.body_byte_end is not None:
                if sym.body_byte_start <= boundary <= sym.body_byte_end:
                    matching.append(sym)
    else:
        for sym in symbols:
            if max(sym.byte_start, b_start) < min(sym.byte_end, b_end):
                matching.append(sym)

    if not matching:
        return []

    smallest = min(matching, key=lambda s: (s.byte_end - s.byte_start, -s.byte_start, s.id))
    return [(s, s.id == smallest.id) for s in matching]


def transport_line_range_across_hunks(
    line_start: int,
    line_end: int,
    hunks: list[ParsedHunk],
) -> tuple[int, int] | None:
    """Transport a 1-based inclusive [line_start, line_end] from old blob to new blob across hunks.
    Returns `(new_line_start, new_line_end)` if no hunk overlaps the range, or `None` if modified/overlapped.
    """
    if line_start < 1 or line_end < line_start:
        return None
    delta = 0
    for h in hunks:
        if h.old_count == 0:
            if line_start <= h.old_start < line_end:
                return None
            if h.old_start < line_start:
                delta += h.new_count
        else:
            h_old_start = h.old_start
            h_old_end = h.old_start + h.old_count - 1
            if max(line_start, h_old_start) <= min(line_end, h_old_end):
                return None
            if h_old_end < line_start:
                delta += h.new_count - h.old_count
    return (line_start + delta, line_end + delta)


def detect_commit_renames_and_moves(
    mirror: GitMirror,
    parent_oid: str | None,
    commit_oid: str,
    raw_changes: list[RawChange],
) -> tuple[dict[bytes, tuple[bytes, str, int]], bool]:
    """Identify moves/renames between `parent_oid` and `commit_oid`:
    1. Unique deleted path and unique added path with identical blob_oid -> ('exact', 100)
    2. Bounded Git similarity pass (`-M50% -l200`) for edited renames -> ('candidate', score)
    Returns `(new_path -> (old_path, mapping_kind, score), rename_limit_exceeded)`.
    """
    deleted_by_blob: dict[str, list[bytes]] = {}
    added_by_blob: dict[str, list[bytes]] = {}
    deleted_count = 0
    added_count = 0

    for ch in raw_changes:
        if ch.change_kind == "deleted" and ch.old_blob_oid and ch.old_path is not None:
            deleted_by_blob.setdefault(ch.old_blob_oid, []).append(ch.old_path)
            deleted_count += 1
        elif ch.change_kind == "added" and ch.new_blob_oid and ch.new_path is not None:
            added_by_blob.setdefault(ch.new_blob_oid, []).append(ch.new_path)
            added_count += 1

    moves: dict[bytes, tuple[bytes, str, int]] = {}
    claimed_old_paths: set[bytes] = set()
    for blob_oid, add_paths in added_by_blob.items():
        del_paths = deleted_by_blob.get(blob_oid, [])
        if len(add_paths) == 1 and len(del_paths) == 1:
            moves[add_paths[0]] = (del_paths[0], "exact", 100)
            claimed_old_paths.add(del_paths[0])

    if parent_oid is None or deleted_count == 0 or added_count == 0:
        return moves, False

    # If all deletes or all adds were already paired by exact blob OID, no similarity diff-tree needed
    if len(moves) == min(deleted_count, added_count):
        return moves, False

    if deleted_count + added_count > 400:
        return moves, True

    proc = run_safe_git(
        [
            "diff-tree",
            "--no-commit-id",
            "--raw",
            "-r",
            "-z",
            "--no-abbrev",
            "-M50%",
            "-l200",
            parent_oid,
            commit_oid,
        ],
        git_dir=mirror.git_dir,
        empty_hooks_dir=mirror.empty_hooks_dir,
        check=False,
    )
    if proc.returncode != 0:
        return moves, False

    sim_changes = parse_diff_tree_raw_z(proc.stdout, parent_index=0, parent_oid=parent_oid)
    for sch in sim_changes:
        if sch.change_kind == "renamed" and sch.old_path is not None and sch.new_path is not None:
            if sch.new_path not in moves and sch.old_path not in claimed_old_paths:
                score = sch.rename_score if sch.rename_score is not None else 50
                if sch.old_blob_oid and sch.old_blob_oid == sch.new_blob_oid:
                    if (
                        len(added_by_blob.get(sch.new_blob_oid, [])) == 1
                        and len(deleted_by_blob.get(sch.old_blob_oid, [])) == 1
                    ):
                        moves[sch.new_path] = (sch.old_path, "exact", 100)
                    else:
                        moves[sch.new_path] = (sch.old_path, "candidate", score)
                else:
                    moves[sch.new_path] = (sch.old_path, "candidate", score)
                claimed_old_paths.add(sch.old_path)

    return moves, False


def _coalesce_renamed_raw_changes(
    raw_changes: list[RawChange],
    moves: dict[bytes, tuple[bytes, str, int]],
) -> list[RawChange]:
    """Combine 1-to-1 paired deleted+added `RawChange` items into `change_kind='renamed'` records
    with `rename_score` populated.
    """
    if not moves:
        return raw_changes

    deleted_by_path: dict[bytes, RawChange] = {}
    for rch in raw_changes:
        if rch.change_kind == "deleted" and rch.old_path is not None:
            deleted_by_path[rch.old_path] = rch

    paired_old: set[bytes] = set()
    paired_new: dict[bytes, tuple[RawChange, int]] = {}
    for new_p, (old_p, _kind, score) in moves.items():
        if old_p in deleted_by_path and old_p not in paired_old:
            paired_old.add(old_p)
            paired_new[new_p] = (deleted_by_path[old_p], score)

    if not paired_new:
        return raw_changes

    coalesced: list[RawChange] = []
    for rch in raw_changes:
        if rch.change_kind == "deleted" and rch.old_path in paired_old:
            continue
        if rch.change_kind == "added" and rch.new_path in paired_new:
            del_ch, score = paired_new[rch.new_path]
            coalesced.append(
                RawChange(
                    parent_index=rch.parent_index,
                    parent_oid=rch.parent_oid,
                    old_mode=del_ch.old_mode,
                    new_mode=rch.new_mode,
                    old_blob_oid=del_ch.old_blob_oid,
                    new_blob_oid=rch.new_blob_oid,
                    change_kind="renamed",
                    rename_score=score,
                    old_path=del_ch.old_path,
                    new_path=rch.new_path,
                )
            )
        else:
            coalesced.append(rch)
    return coalesced


def map_original_anchor_to_target(
    conn: sqlite3.Connection,
    mirror: GitMirror,
    *,
    original_commit_oid: str,
    original_path: bytes,
    original_line_start: int,
    original_line_end: int,
    target_revision: str,
) -> AnchorMappingResult:
    """Apply the ordered mapping procedure from 02-ingestion-and-lineage §6."""
    try:
        orig_entries = {e.path: e for e in mirror.list_tree_entries(original_commit_oid)}
    except Exception:
        return AnchorMappingResult(
            mapping="unmapped",
            method="missing_original_commit",
            gap_code="MISSING_GIT_OBJECT",
            gap_message=f"Original commit {original_commit_oid[:12]} is unavailable.",
        )

    orig_entry = orig_entries.get(original_path)
    if orig_entry is None:
        return AnchorMappingResult(
            mapping="unmapped",
            method="missing_original_path",
            gap_code="MAPPING_UNAVAILABLE",
            gap_message="Original path was not found in the original commit tree.",
        )

    with mirror.open_batch_session() as batch:
        orig_blob_obj = batch.read_object_bounded(orig_entry.oid, max_bytes=MAX_PARSE_BYTES)
        if orig_blob_obj is None:
            return AnchorMappingResult(
                mapping="unmapped",
                method="missing_original_blob",
                gap_code="MISSING_GIT_OBJECT",
                gap_message=f"Original blob {orig_entry.oid[:12]} is missing.",
            )
        if orig_blob_obj[2] is None:
            return AnchorMappingResult(
                mapping="unmapped",
                method="oversized_original_blob",
                gap_code="BLOB_TOO_LARGE",
                gap_message=f"Original blob {orig_entry.oid[:12]} exceeds 2 MiB limit.",
            )
        orig_bytes = orig_blob_obj[2]
        orig_idx = LineByteIndex(orig_bytes)

        line_count = original_line_end - original_line_start + 1
        try:
            orig_b_start, orig_b_end, _, _ = orig_idx.line_range_to_byte_span(
                original_line_start, line_count
            )
        except ValueError:
            return AnchorMappingResult(
                mapping="unmapped",
                method="invalid_original_range",
                gap_code="MAPPING_UNAVAILABLE",
                gap_message="Original line range exceeds original blob bounds.",
            )

        orig_slice = orig_bytes[orig_b_start:orig_b_end]

        try:
            target_entries = {e.path: e for e in mirror.list_tree_entries(target_revision)}
        except Exception:
            return AnchorMappingResult(
                mapping="unmapped",
                method="missing_target_tree",
                gap_code="MISSING_GIT_OBJECT",
                gap_message=f"Target revision {target_revision[:12]} tree is unavailable.",
            )

        # Case 2A: Same path exists in target revision with identical blob OID
        t_same_path = target_entries.get(original_path)
        if t_same_path is not None and t_same_path.oid == orig_entry.oid:
            sym_id = _find_enclosing_symbol_in_db(
                conn, t_same_path.oid, orig_b_start, orig_b_end
            )
            return AnchorMappingResult(
                mapping="exact",
                method="identical_path_and_blob",
                target_path=original_path,
                target_blob_oid=t_same_path.oid,
                target_line_start=original_line_start,
                target_line_end=original_line_end,
                target_byte_start=orig_b_start,
                target_byte_end=orig_b_end,
                target_symbol_id=sym_id,
                transformations=[{"step": "unchanged_blob", "path": path_bytes_to_display(original_path)[0]}],
            )

        # Case 2B: Same path exists in target revision; check unchanged range transport across diff hunks
        if t_same_path is not None:
            t_blob_obj = batch.read_object_bounded(t_same_path.oid, max_bytes=MAX_PARSE_BYTES)
            if t_blob_obj is not None and t_blob_obj[2] is not None:
                t_bytes = t_blob_obj[2]
                hunks = compute_unified_zero_hunks(
                    orig_bytes,
                    t_bytes,
                    mirror=mirror,
                    old_oid=orig_entry.oid,
                    new_oid=t_same_path.oid,
                )
                transported = (
                    transport_line_range_across_hunks(original_line_start, original_line_end, hunks)
                    if hunks is not None
                    else None
                )
                if transported is not None:
                    new_l_start, new_l_end = transported
                    try:
                        t_idx = LineByteIndex(t_bytes)
                        new_b_start, new_b_end, _, _ = t_idx.line_range_to_byte_span(
                            new_l_start, new_l_end - new_l_start + 1
                        )
                        if t_bytes[new_b_start:new_b_end] == orig_slice:
                            sym_id = _find_enclosing_symbol_in_db(
                                conn, t_same_path.oid, new_b_start, new_b_end
                            )
                            return AnchorMappingResult(
                                mapping="exact",
                                method="unchanged_range_transport",
                                target_path=original_path,
                                target_blob_oid=t_same_path.oid,
                                target_line_start=new_l_start,
                                target_line_end=new_l_end,
                                target_byte_start=new_b_start,
                                target_byte_end=new_b_end,
                                target_symbol_id=sym_id,
                                transformations=[
                                    {
                                        "step": "hunk_offset_transport",
                                        "old_lines": [original_line_start, original_line_end],
                                        "new_lines": [new_l_start, new_l_end],
                                    }
                                ],
                            )
                    except ValueError:
                        pass

        # Case 2C: Unique exact-byte file move
        if t_same_path is None:
            matching_blob_paths = [
                p for p, e in target_entries.items() if e.oid == orig_entry.oid
            ]
            if len(matching_blob_paths) == 1:
                moved_path = matching_blob_paths[0]
                sym_id = _find_enclosing_symbol_in_db(
                    conn, orig_entry.oid, orig_b_start, orig_b_end
                )
                return AnchorMappingResult(
                    mapping="exact",
                    method="unique_exact_blob_move",
                    target_path=moved_path,
                    target_blob_oid=orig_entry.oid,
                    target_line_start=original_line_start,
                    target_line_end=original_line_end,
                    target_byte_start=orig_b_start,
                    target_byte_end=orig_b_end,
                    target_symbol_id=sym_id,
                    transformations=[
                        {
                            "step": "unique_blob_move",
                            "from_path": path_bytes_to_display(original_path)[0],
                            "to_path": path_bytes_to_display(moved_path)[0],
                        }
                    ],
                )
            elif len(matching_blob_paths) > 1:
                return AnchorMappingResult(
                    mapping="candidate",
                    method="ambiguous_identical_blob_copies",
                    target_path=matching_blob_paths[0],
                    target_blob_oid=orig_entry.oid,
                    target_line_start=original_line_start,
                    target_line_end=original_line_end,
                    target_byte_start=orig_b_start,
                    target_byte_end=orig_b_end,
                    gap_code="MAPPING_AMBIGUOUS",
                    gap_message=f"Original blob appears at {len(matching_blob_paths)} target paths.",
                )

        # Case 3: Candidate correspondence via enclosing symbol qualified_name
        lang = detect_language_for_path(original_path)
        orig_sym_qname: str | None = None
        if lang is not None:
            from codecontext.syntax import parse_blob

            orig_parsed = parse_blob(orig_bytes, lang, blob_oid=orig_entry.oid)
            enclosing = [
                s
                for s in orig_parsed.symbols
                if max(s.byte_start, orig_b_start) < min(s.byte_end, orig_b_end)
            ]
            if enclosing:
                smallest_orig = min(enclosing, key=lambda s: (s.byte_end - s.byte_start, s.id))
                orig_sym_qname = smallest_orig.qualified_name

        if orig_sym_qname is not None:
            if t_same_path is not None:
                cand_sym = conn.execute(
                    """
                    SELECT s.id, s.byte_start, s.byte_end, s.line_start, s.line_end
                    FROM symbols s
                    JOIN extractions e ON e.id = s.extraction_id
                    WHERE e.blob_oid = ? AND s.qualified_name = ?
                    """,
                    (t_same_path.oid, orig_sym_qname),
                ).fetchall()
                if len(cand_sym) == 1:
                    cs = cand_sym[0]
                    return AnchorMappingResult(
                        mapping="candidate",
                        method="same_path_lexical_symbol",
                        target_path=original_path,
                        target_blob_oid=t_same_path.oid,
                        target_line_start=int(cs["line_start"]),
                        target_line_end=int(cs["line_end"]),
                        target_byte_start=int(cs["byte_start"]),
                        target_byte_end=int(cs["byte_end"]),
                        target_symbol_id=str(cs["id"]),
                        transformations=[
                            {"step": "symbol_qualified_name_match", "qualified_name": orig_sym_qname}
                        ],
                    )

            active_matches = conn.execute(
                """
                SELECT sf.path, sf.blob_oid, s.id, s.byte_start, s.byte_end, s.line_start, s.line_end
                FROM snapshot_files sf
                JOIN symbols s ON s.extraction_id = sf.extraction_id
                WHERE s.qualified_name = ?
                """,
                (orig_sym_qname,),
            ).fetchall()
            if len(active_matches) == 1:
                am = active_matches[0]
                return AnchorMappingResult(
                    mapping="candidate",
                    method="unique_snapshot_lexical_symbol",
                    target_path=bytes(am["path"]),
                    target_blob_oid=str(am["blob_oid"]),
                    target_line_start=int(am["line_start"]),
                    target_line_end=int(am["line_end"]),
                    target_byte_start=int(am["byte_start"]),
                    target_byte_end=int(am["byte_end"]),
                    target_symbol_id=str(am["id"]),
                    transformations=[
                        {
                            "step": "cross_file_symbol_match",
                            "qualified_name": orig_sym_qname,
                            "to_path": path_bytes_to_display(bytes(am["path"]))[0],
                        }
                    ],
                )
            elif len(active_matches) > 1:
                am0 = active_matches[0]
                return AnchorMappingResult(
                    mapping="candidate",
                    method="ambiguous_lexical_symbol_copies",
                    target_path=bytes(am0["path"]),
                    target_blob_oid=str(am0["blob_oid"]),
                    target_line_start=int(am0["line_start"]),
                    target_line_end=int(am0["line_end"]),
                    target_byte_start=int(am0["byte_start"]),
                    target_byte_end=int(am0["byte_end"]),
                    target_symbol_id=str(am0["id"]),
                    gap_code="MAPPING_AMBIGUOUS",
                    gap_message=f"Symbol {orig_sym_qname!r} matches {len(active_matches)} active paths.",
                )

    return AnchorMappingResult(
        mapping="unmapped",
        method="no_correspondence",
        gap_code="MAPPING_UNAVAILABLE",
        gap_message="Original code anchor has no verified correspondence in the target revision.",
    )


def map_review_comment_to_active_snapshot(
    conn: sqlite3.Connection,
    mirror: GitMirror | None,
    *,
    repository_id: str,
    review_comment_source_id: str,
    anchors: dict[str, Any] | None,
    evidence_id: str | None = None,
) -> None:
    """Link an ingested PR/MR inline review comment (`review_comment_source_id`) to its original
    review commit (`review_anchor`) and landed symbol (`landed_mapping`) in the active snapshot.
    """
    if not anchors:
        return
    orig_commit = anchors.get("original_commit_id") or anchors.get("commit_id")
    if isinstance(orig_commit, str) and len(orig_commit) in (40, 64):
        ra_id = make_relation_id(
            from_kind="source",
            from_id=review_comment_source_id,
            to_kind="commit",
            to_id=orig_commit,
            relation_kind="review_anchor",
            method="review_comment_commit_anchor",
        )
        conn.execute(
            """
            INSERT INTO relations (
                id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
            ) VALUES (?, 'source', ?, 'commit', ?, 'review_anchor', 'review_comment_commit_anchor', 'exact', ?, ?, 'active')
            ON CONFLICT(id) DO NOTHING
            """,
            (
                ra_id,
                review_comment_source_id,
                orig_commit,
                evidence_id,
                canonical_json({"line": anchors.get("original_line") or anchors.get("line"), "path": anchors.get("path")}),
            ),
        )

    if mirror is None or not isinstance(orig_commit, str) or len(orig_commit) not in (40, 64):
        return
    raw_path = anchors.get("path")
    line_val = anchors.get("original_line") or anchors.get("line")
    if not isinstance(raw_path, str) or not raw_path or not isinstance(line_val, int) or line_val < 1:
        return
    start_line_val = anchors.get("original_start_line") or anchors.get("start_line")
    l_start = start_line_val if isinstance(start_line_val, int) and 1 <= start_line_val <= line_val else line_val
    l_end = line_val

    meta = conn.execute("SELECT active_revision FROM store_meta WHERE singleton = 1").fetchone()
    active_rev = str(meta["active_revision"]) if meta and meta["active_revision"] else None
    if not active_rev:
        return

    res = map_original_anchor_to_target(
        conn,
        mirror,
        original_commit_oid=orig_commit,
        original_path=raw_path.encode("utf-8", errors="surrogateescape"),
        original_line_start=l_start,
        original_line_end=l_end,
        target_revision=active_rev,
    )
    if res.target_symbol_id and res.mapping in ("exact", "candidate"):
        lm_id = make_relation_id(
            from_kind="source",
            from_id=review_comment_source_id,
            to_kind="symbol",
            to_id=res.target_symbol_id,
            relation_kind="landed_mapping",
            method=res.method,
            anchor_details={"target_revision": active_rev},
        )
        conn.execute(
            """
            INSERT INTO relations (
                id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
            ) VALUES (?, 'source', ?, 'symbol', ?, 'landed_mapping', ?, ?, ?, ?, 'active')
            ON CONFLICT(id) DO UPDATE SET
                method = excluded.method,
                mapping = excluded.mapping,
                details_json = excluded.details_json,
                state = 'active'
            """,
            (
                lm_id,
                review_comment_source_id,
                res.target_symbol_id,
                res.method,
                res.mapping,
                evidence_id,
                canonical_json(
                    {
                        "target_line_end": res.target_line_end,
                        "target_line_start": res.target_line_start,
                        "target_revision": active_rev,
                        "transformations": res.transformations,
                    }
                ),
            ),
        )
    if res.gap_code == "MAPPING_AMBIGUOUS" and res.gap_message:
        Store.upsert_gap(
            conn,
            repository_id=repository_id,
            code="MAPPING_AMBIGUOUS",
            scope=f"review_comment:{review_comment_source_id}",
            message=res.gap_message,
            retryable=False,
            record_id=review_comment_source_id,
        )


def _extract_and_persist_commit_message_relations(
    conn: sqlite3.Connection,
    *,
    commit_oid: str,
    message_text: str,
    commit_ev_id: str,
) -> None:
    """Offline extraction of revert links (`This reverts commit <oid>`), squash-PR references
    (`(#N)`, `Merge pull request #N`, `See merge request !N`), and issue references (`Fixes #N`).
    """
    if not message_text:
        return

    for m in REVERT_COMMIT_RE.finditer(message_text):
        rev_oid = m.group(1).lower()
        if rev_oid == commit_oid:
            continue
        rel_id = make_relation_id(
            from_kind="commit",
            from_id=commit_oid,
            to_kind="commit",
            to_id=rev_oid,
            relation_kind="supersedes",
            method="commit_revert_trailer",
        )
        conn.execute(
            """
            INSERT INTO relations (
                id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
            ) VALUES (?, 'commit', ?, 'commit', ?, 'supersedes', 'commit_revert_trailer', 'exact', ?, '{}', 'active')
            ON CONFLICT(id) DO NOTHING
            """,
            (rel_id, commit_oid, rev_oid, commit_ev_id),
        )

    meta = conn.execute("SELECT * FROM store_meta WHERE singleton = 1").fetchone()
    gh_repo_id = str(meta["github_repository_id"]) if meta and meta["github_repository_id"] is not None else None
    gl_proj_id = (
        f"gitlab:{meta['gitlab_project_id']}"
        if meta and "gitlab_project_id" in meta.keys() and meta["gitlab_project_id"] is not None
        else None
    )

    pr_nums: list[tuple[str, str]] = []
    for pm in PR_TRAILER_RE.finditer(message_text):
        gl_num = pm.group("gl_mr")
        gh_num = pm.group("squash_pr") or pm.group("merge_pr") or pm.group("hdr_pr")
        if gl_num:
            pr_nums.append(("gitlab", gl_num))
        elif gh_num:
            pr_nums.append(("any", gh_num))

    for prov_hint, pr_num in dict.fromkeys(pr_nums):
        src_rows = conn.execute(
            "SELECT id FROM sources WHERE kind = 'pull_request' AND provider_id = ?",
            (pr_num,),
        ).fetchall()
        target_sids = [str(r["id"]) for r in src_rows]
        if not target_sids:
            if prov_hint in ("any", "github") and gh_repo_id:
                target_sids.append(
                    make_src_id(
                        provider="github",
                        repository_provider_id=gh_repo_id,
                        kind="pull_request",
                        provider_id=pr_num,
                    )
                )
            elif prov_hint in ("any", "gitlab") and gl_proj_id:
                target_sids.append(
                    make_src_id(
                        provider="gitlab",
                        repository_provider_id=gl_proj_id,
                        kind="pull_request",
                        provider_id=pr_num,
                    )
                )
        for t_sid in target_sids:
            rel_id = make_relation_id(
                from_kind="commit",
                from_id=commit_oid,
                to_kind="source",
                to_id=t_sid,
                relation_kind="associated_pr",
                method="commit_message_trailer",
            )
            conn.execute(
                """
                INSERT INTO relations (
                    id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
                ) VALUES (?, 'commit', ?, 'source', ?, 'associated_pr', 'commit_message_trailer', 'candidate', ?, ?, 'active')
                ON CONFLICT(id) DO NOTHING
                """,
                (rel_id, commit_oid, t_sid, commit_ev_id, canonical_json({"number": int(pr_num)})),
            )

    iss_nums = list(dict.fromkeys(im.group("num") for im in ISSUE_TRAILER_RE.finditer(message_text)))
    for iss_num in iss_nums:
        src_rows = conn.execute(
            "SELECT id FROM sources WHERE kind = 'issue' AND provider_id = ?",
            (iss_num,),
        ).fetchall()
        target_sids = [str(r["id"]) for r in src_rows]
        if not target_sids:
            if gh_repo_id:
                target_sids.append(
                    make_src_id(
                        provider="github",
                        repository_provider_id=gh_repo_id,
                        kind="issue",
                        provider_id=iss_num,
                    )
                )
            elif gl_proj_id:
                target_sids.append(
                    make_src_id(
                        provider="gitlab",
                        repository_provider_id=gl_proj_id,
                        kind="issue",
                        provider_id=iss_num,
                    )
                )
        for t_sid in target_sids:
            rel_id = make_relation_id(
                from_kind="commit",
                from_id=commit_oid,
                to_kind="source",
                to_id=t_sid,
                relation_kind="references_issue",
                method="commit_message_trailer",
            )
            conn.execute(
                """
                INSERT INTO relations (
                    id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
                ) VALUES (?, 'commit', ?, 'source', ?, 'references_issue', 'commit_message_trailer', 'candidate', ?, ?, 'active')
                ON CONFLICT(id) DO NOTHING
                """,
                (rel_id, commit_oid, t_sid, commit_ev_id, canonical_json({"number": int(iss_num)})),
            )


def _find_enclosing_symbol_in_db(
    conn: sqlite3.Connection,
    blob_oid: str,
    byte_start: int,
    byte_end: int,
) -> str | None:
    rows = conn.execute(
        """
        SELECT s.id, s.byte_start, s.byte_end
        FROM symbols s
        JOIN extractions e ON e.id = s.extraction_id
        WHERE e.blob_oid = ? AND s.byte_start <= ? AND s.byte_end >= ?
        ORDER BY (s.byte_end - s.byte_start) ASC, s.id ASC
        LIMIT 1
        """,
        (blob_oid, byte_start, byte_end),
    ).fetchall()
    if rows:
        return str(rows[0]["id"])
    return None


def index_commit_changes_and_hunks(
    store: Store,
    mirror: GitMirror,
    commit_oid: str,
    *,
    filter_paths: set[bytes] | None = None,
    batch: CatFileBatchSession | None = None,
    conn: sqlite3.Connection | None = None,
) -> tuple[int, set[bytes]]:
    """Index parent-specific raw changes, bounded text hunks, changed-blob syntax,
    commit/hunk/code evidence handles, and symbol-to-change relations for `commit_oid`.
    Returns `(changes_indexed, discovered_predecessor_paths)`.
    """
    check_deadline()
    if conn is not None:
        meta = conn.execute("SELECT repository_id FROM store_meta WHERE singleton = 1").fetchone()
        if not meta:
            return (0, set())
        repo_id = str(meta["repository_id"])
        parent_rows = conn.execute(
            "SELECT parent_index, parent_oid FROM commit_parents WHERE commit_oid = ? ORDER BY parent_index ASC",
            (commit_oid,),
        ).fetchall()
    else:
        with store.connect_read() as r_conn:
            meta = r_conn.execute(
                "SELECT repository_id FROM store_meta WHERE singleton = 1"
            ).fetchone()
            if not meta:
                return (0, set())
            repo_id = str(meta["repository_id"])
            parent_rows = r_conn.execute(
                "SELECT parent_index, parent_oid FROM commit_parents WHERE commit_oid = ? ORDER BY parent_index ASC",
                (commit_oid,),
            ).fetchall()

    # Merges are diffed against their first parent only (`git log --first-parent -m`): that diff
    # is what the merge brought into the mainline, and side-branch commits are indexed on their own.
    parents: list[tuple[int, str | None]] = (
        [(int(parent_rows[0]["parent_index"]), str(parent_rows[0]["parent_oid"]))]
        if parent_rows
        else [(-1, None)]
    )

    predecessor_paths: set[bytes] = set()
    missing_parent = False

    with contextlib.ExitStack() as stack:
        active_batch = batch if batch is not None else stack.enter_context(mirror.open_batch_session())
        w_conn = conn if conn is not None else stack.enter_context(store.write_transaction(bump_generation=True))

        commit_obj = active_batch.read_object(commit_oid)
        parsed_commit = (
            parse_raw_commit_object(commit_oid, commit_obj[2])
            if commit_obj is not None and commit_obj[0] == "commit"
            else None
        )

        ts = now_us()
        if parsed_commit is not None:
            w_conn.execute(
                """
                INSERT INTO commits (
                    oid, tree_oid, authored_at_us, committed_at_us,
                    author_name, author_email, message, first_seen_at_us, diff_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                ON CONFLICT(oid) DO NOTHING
                """,
                (
                    parsed_commit.oid,
                    parsed_commit.tree_oid,
                    parsed_commit.authored_at_us,
                    parsed_commit.committed_at_us,
                    parsed_commit.author_name,
                    parsed_commit.author_email,
                    parsed_commit.message_text,
                    ts,
                ),
            )
            msg_start = parsed_commit.header_byte_len
            msg_end = parsed_commit.header_byte_len + len(parsed_commit.message_bytes)
            commit_anchor = {
                "git_oid": commit_oid,
                "kind": "commit",
                "original_review_commit": None,
                "path": None,
                "revision": commit_oid,
                "source_version_id": None,
                "span": {
                    "byte_end": msg_end,
                    "byte_start": msg_start,
                    "line_end": None,
                    "line_start": None,
                },
                "underlying_record_id": None,
            }
            msg_hash = sha256_bytes(parsed_commit.message_bytes)
            commit_ev_id = make_ev_id(
                repository_id=repo_id,
                kind="commit",
                source_version_id=None,
                git_oid=commit_oid,
                anchor=commit_anchor,
                content_sha256=msg_hash,
            )
            w_conn.execute(
                """
                INSERT INTO evidence (id, kind, source_version_id, git_oid, anchor_json, content_sha256, state)
                VALUES (?, 'commit', NULL, ?, ?, ?, 'available')
                ON CONFLICT(id) DO NOTHING
                """,
                (commit_ev_id, commit_oid, canonical_json(commit_anchor), msg_hash),
            )
            has_pr_ep = w_conn.execute(
                "SELECT 1 FROM episodes WHERE kind = 'pr' AND commit_oid = ? LIMIT 1",
                (commit_oid,),
            ).fetchone()
            if not has_pr_ep:
                ep_id = make_episode_id(repository_id=repo_id, commit_oid=commit_oid)
                w_conn.execute(
                    """
                    INSERT INTO episodes (id, kind, primary_source_id, commit_oid, input_fingerprint, state)
                    VALUES (?, 'commit', NULL, ?, ?, 'pending')
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (ep_id, commit_oid, msg_hash),
                )
            _extract_and_persist_commit_message_relations(
                w_conn,
                commit_oid=commit_oid,
                message_text=parsed_commit.message_text,
                commit_ev_id=commit_ev_id,
            )

        changes_count = 0

        for p_idx, p_oid in parents:
            check_deadline()

            if p_oid is not None and active_batch.read_object_bounded(p_oid, max_bytes=0) is None:
                missing_parent = True
                Store.upsert_gap(
                    w_conn,
                    repository_id=repo_id,
                    code="MISSING_PARENT",
                    scope=f"git:commit:{commit_oid}",
                    message=f"Parent commit {p_oid[:12]} of {commit_oid[:12]} is missing.",
                    retryable=True,
                )
                w_conn.execute(
                    "UPDATE commits SET diff_state = 'partial' WHERE oid = ?",
                    (commit_oid,),
                )
                continue

            raw_changes = mirror.diff_tree_parent(p_oid, commit_oid, parent_index=p_idx)

            moves, _ = detect_commit_renames_and_moves(mirror, p_oid, commit_oid, raw_changes)
            for new_p, (old_p, _, _) in moves.items():
                if filter_paths is None or new_p in filter_paths:
                    predecessor_paths.add(old_p)

            effective_changes = _coalesce_renamed_raw_changes(raw_changes, moves)

            for rch in effective_changes:
                check_deadline()
                if filter_paths is not None:
                    touches = (rch.new_path in filter_paths) or (rch.old_path in filter_paths)
                    if not touches:
                        continue

                old_bytes: bytes | None = None
                new_bytes: bytes | None = None
                can_diff = rch.old_mode != 0o160000 and rch.new_mode != 0o160000

                if rch.old_blob_oid and can_diff:
                    ob = active_batch.read_object_bounded(rch.old_blob_oid, max_bytes=MAX_PARSE_BYTES)
                    if ob is not None and ob[0] == "blob" and ob[2] is not None:
                        if b"\x00" in ob[2] or is_lfs_pointer(ob[2]):
                            can_diff = False
                        else:
                            old_bytes = ob[2]
                    else:
                        can_diff = False

                if rch.new_blob_oid and can_diff:
                    nb = active_batch.read_object_bounded(rch.new_blob_oid, max_bytes=MAX_PARSE_BYTES)
                    if nb is not None and nb[0] == "blob" and nb[2] is not None:
                        if b"\x00" in nb[2] or is_lfs_pointer(nb[2]):
                            can_diff = False
                        else:
                            new_bytes = nb[2]
                    else:
                        can_diff = False

                if not can_diff:
                    old_bytes = None
                    new_bytes = None

                diffed = (
                    compute_unified_zero_hunks(
                        old_bytes,
                        new_bytes,
                        mirror=mirror,
                        old_oid=rch.old_blob_oid,
                        new_oid=rch.new_blob_oid,
                    )
                    if can_diff
                    else []
                )
                can_diff = can_diff and diffed is not None
                hunks: list[ParsedHunk] = diffed or []

                old_b64 = encode_path_b64(rch.old_path) if rch.old_path is not None else None
                new_b64 = encode_path_b64(rch.new_path) if rch.new_path is not None else None
                chg_id = make_chg_id(
                    repository_id=repo_id,
                    commit_oid=commit_oid,
                    parent_index=rch.parent_index,
                    old_path_b64=old_b64,
                    new_path_b64=new_b64,
                    old_blob_oid=rch.old_blob_oid,
                    new_blob_oid=rch.new_blob_oid,
                    change_kind=rch.change_kind,
                )
                h_state = "complete" if can_diff else "unavailable"
                w_conn.execute(
                    """
                    INSERT INTO changes (
                        id, commit_oid, parent_index, parent_oid, old_path, new_path,
                        old_blob_oid, new_blob_oid, old_mode, new_mode,
                        change_kind, rename_score, hunks_state
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        rename_score = COALESCE(excluded.rename_score, changes.rename_score),
                        hunks_state = excluded.hunks_state
                    """,
                    (
                        chg_id,
                        commit_oid,
                        rch.parent_index,
                        rch.parent_oid,
                        rch.old_path,
                        rch.new_path,
                        rch.old_blob_oid,
                        rch.new_blob_oid,
                        rch.old_mode,
                        rch.new_mode,
                        rch.change_kind,
                        rch.rename_score,
                        h_state,
                    ),
                )
                changes_count += 1

                old_syms: list[ExtractedSymbol] = []
                new_syms: list[ExtractedSymbol] = []
                old_line_idx: LineByteIndex | None = None
                new_line_idx: LineByteIndex | None = None

                if rch.old_path is not None and rch.old_blob_oid is not None and old_bytes is not None:
                    lang = detect_language_for_path(rch.old_path)
                    if lang is not None:
                        try:
                            old_bytes.decode("utf-8", errors="strict")
                            old_res, _ = index_blob_into_store(
                                w_conn,
                                repository_id=repo_id,
                                blob_oid=rch.old_blob_oid,
                                blob_bytes=old_bytes,
                                language=lang,
                            )
                            old_syms = old_res.symbols
                            if old_syms and hunks:
                                old_line_idx = LineByteIndex(old_bytes)
                        except UnicodeDecodeError:
                            pass

                if rch.new_path is not None and rch.new_blob_oid is not None and new_bytes is not None:
                    lang = detect_language_for_path(rch.new_path)
                    if lang is not None:
                        try:
                            new_bytes.decode("utf-8", errors="strict")
                            new_res, _ = index_blob_into_store(
                                w_conn,
                                repository_id=repo_id,
                                blob_oid=rch.new_blob_oid,
                                blob_bytes=new_bytes,
                                language=lang,
                            )
                            new_syms = new_res.symbols
                            if new_syms and hunks:
                                new_line_idx = LineByteIndex(new_bytes)
                        except UnicodeDecodeError:
                            pass

                for hk in hunks:
                    changed_symbol_names: set[str] = set()
                    p_hash = sha256_bytes(hk.patch_bytes)
                    hunk_id = make_hunk_id(
                        change_id=chg_id,
                        ordinal=hk.ordinal,
                        old_start=hk.old_start,
                        old_count=hk.old_count,
                        new_start=hk.new_start,
                        new_count=hk.new_count,
                        patch_sha256=p_hash,
                    )
                    w_conn.execute(
                        """
                        INSERT INTO hunks (
                            id, change_id, ordinal, old_start, old_count, new_start, new_count, patch_text
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO NOTHING
                        """,
                        (
                            hunk_id,
                            chg_id,
                            hk.ordinal,
                            hk.old_start,
                            hk.old_count,
                            hk.new_start,
                            hk.new_count,
                            hk.patch_text,
                        ),
                    )

                    disp_p, b64_p = (
                        path_bytes_to_display(rch.new_path)
                        if rch.new_path is not None
                        else (
                            path_bytes_to_display(rch.old_path)
                            if rch.old_path is not None
                            else ("", None)
                        )
                    )
                    hunk_anchor = {
                        "git_oid": commit_oid,
                        "kind": "hunk",
                        "original_review_commit": None,
                        "path": {"path": disp_p, "path_bytes_b64": b64_p} if disp_p else None,
                        "revision": commit_oid,
                        "source_version_id": None,
                        "span": {
                            "byte_end": len(hk.patch_bytes),
                            "byte_start": 0,
                            "line_end": None,
                            "line_start": None,
                        },
                        "underlying_record_id": hunk_id,
                    }
                    hunk_ev_id = make_ev_id(
                        repository_id=repo_id,
                        kind="hunk",
                        source_version_id=None,
                        git_oid=commit_oid,
                        anchor=hunk_anchor,
                        content_sha256=p_hash,
                    )
                    w_conn.execute(
                        """
                        INSERT INTO evidence (id, kind, source_version_id, git_oid, anchor_json, content_sha256, state)
                        VALUES (?, 'hunk', NULL, ?, ?, ?, 'available')
                        ON CONFLICT(id) DO NOTHING
                        """,
                        (hunk_ev_id, commit_oid, canonical_json(hunk_anchor), p_hash),
                    )

                    if old_bytes is not None and old_syms:
                        for sym, is_prim in associate_hunk_side_with_symbols(
                            old_bytes,
                            hk.old_start,
                            hk.old_count,
                            old_syms,
                            line_index=old_line_idx,
                        ):
                            changed_symbol_names.add(sym.qualified_name)
                            rel_id = make_relation_id(
                                from_kind="change",
                                from_id=chg_id,
                                to_kind="symbol",
                                to_id=sym.id,
                                relation_kind="changes",
                                method="hunk_old_overlap",
                                anchor_details={"hunk_id": hunk_id, "primary": is_prim, "side": "old"},
                            )
                            w_conn.execute(
                                """
                                INSERT INTO relations (
                                    id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
                                ) VALUES (?, 'change', ?, 'symbol', ?, 'changes', 'hunk_old_overlap', 'exact', ?, ?, 'active')
                                ON CONFLICT(id) DO NOTHING
                                """,
                                (
                                    rel_id,
                                    chg_id,
                                    sym.id,
                                    hunk_ev_id,
                                    canonical_json({"hunk_id": hunk_id, "primary": is_prim, "side": "old"}),
                                ),
                            )

                    if new_bytes is not None and new_syms and rch.new_path is not None and rch.new_blob_oid is not None:
                        new_disp_p, new_b64_p = path_bytes_to_display(rch.new_path)
                        for sym, is_prim in associate_hunk_side_with_symbols(
                            new_bytes,
                            hk.new_start,
                            hk.new_count,
                            new_syms,
                            line_index=new_line_idx,
                        ):
                            changed_symbol_names.add(sym.qualified_name)
                            sym_slice = new_bytes[sym.byte_start : sym.byte_end]
                            sym_hash = sha256_bytes(sym_slice)
                            code_anchor = {
                                "git_oid": rch.new_blob_oid,
                                "kind": "code",
                                "original_review_commit": None,
                                "path": {"path": new_disp_p, "path_bytes_b64": new_b64_p},
                                "revision": commit_oid,
                                "source_version_id": None,
                                "span": {
                                    "byte_end": sym.byte_end,
                                    "byte_start": sym.byte_start,
                                    "line_end": sym.line_end,
                                    "line_start": sym.line_start,
                                },
                                "underlying_record_id": sym.id,
                            }
                            code_ev_id = make_ev_id(
                                repository_id=repo_id,
                                kind="code",
                                source_version_id=None,
                                git_oid=rch.new_blob_oid,
                                anchor=code_anchor,
                                content_sha256=sym_hash,
                            )
                            w_conn.execute(
                                """
                                INSERT INTO evidence (id, kind, source_version_id, git_oid, anchor_json, content_sha256, state)
                                VALUES (?, 'code', NULL, ?, ?, ?, 'available')
                                ON CONFLICT(id) DO NOTHING
                                """,
                                (code_ev_id, rch.new_blob_oid, canonical_json(code_anchor), sym_hash),
                            )
                            rel_id = make_relation_id(
                                from_kind="change",
                                from_id=chg_id,
                                to_kind="symbol",
                                to_id=sym.id,
                                relation_kind="changes",
                                method="hunk_new_overlap",
                                anchor_details={"hunk_id": hunk_id, "primary": is_prim, "side": "new"},
                            )
                            w_conn.execute(
                                """
                                INSERT INTO relations (
                                    id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping, evidence_id, details_json, state
                                ) VALUES (?, 'change', ?, 'symbol', ?, 'changes', 'hunk_new_overlap', 'exact', ?, ?, 'active')
                                ON CONFLICT(id) DO NOTHING
                                """,
                                (
                                    rel_id,
                                    chg_id,
                                    sym.id,
                                    hunk_ev_id,
                                    canonical_json(
                                        {
                                            "code_evidence_id": code_ev_id,
                                            "hunk_id": hunk_id,
                                            "primary": is_prim,
                                            "side": "new",
                                        }
                                    ),
                                ),
                            )

                    if parsed_commit is not None:
                        title_line = (
                            parsed_commit.message_text.splitlines()[0]
                            if parsed_commit.message_text
                            else commit_oid[:12]
                        )
                        meta_json = canonical_json(
                            {
                                "authored_at_us": parsed_commit.authored_at_us,
                                "commit_oid": commit_oid,
                                "committed_at_us": parsed_commit.committed_at_us,
                                "display_title": f"{title_line} ({disp_p})",
                                "evidence_id": hunk_ev_id,
                                "path": disp_p,
                                "path_bytes_b64": b64_p,
                                "snippet": parsed_commit.message_text.strip()[:320],
                            }
                        )
                        w_conn.execute(
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
                                chg_id,
                                hk.ordinal,
                                normalize_search_text(f"{title_line} {disp_p}"),
                                normalize_search_text(" ".join([disp_p, *sorted(changed_symbol_names)])),
                                normalize_search_text(f"{parsed_commit.message_text} {hk.patch_text}"),
                                meta_json,
                            ),
                        )

                # Drop the blob references now so large files are freed before the next change is read.
                old_bytes = None
                new_bytes = None

        if filter_paths is None:
            w_conn.execute(
                "UPDATE commits SET diff_state = ? WHERE oid = ?",
                ("partial" if missing_parent else "complete", commit_oid),
            )
            if not missing_parent:
                Store.resolve_gaps_in_scope(w_conn, scope=f"git:commit:{commit_oid}", code="MISSING_PARENT")

    return (changes_count, predecessor_paths)


def index_focused_path_history(
    store: Store,
    mirror: GitMirror,
    *,
    active_revision: str,
    focus_path: bytes,
    max_changes: int = 100,
) -> list[str]:
    """Stream path-limited history for `focus_path` and discovered predecessor paths up to `max_changes`.
    Preserves remaining full-history backfill jobs and marks `GIT_HISTORY_PENDING` if unindexed commits remain.
    Returns list of relevant commit OIDs processed.
    """
    visited_commits: list[str] = []
    seen_commits: set[str] = set()
    paths_queue: list[bytes] = [focus_path]
    seen_paths: set[bytes] = {focus_path}

    with mirror.open_batch_session() as batch:
        while paths_queue and len(visited_commits) < max_changes:
            check_deadline()
            cur_path = paths_queue.pop(0)
            rem = max_changes - len(visited_commits)
            decoded_path = (
                os.fsdecode(cur_path)
                if hasattr(os, "fsdecode")
                else cur_path.decode("utf-8", errors="surrogateescape")
            )
            proc = run_safe_git(
                [
                    "rev-list",
                    "--topo-order",
                    f"--max-count={rem}",
                    active_revision,
                    "--",
                    f":(literal){decoded_path}",
                ],
                git_dir=mirror.git_dir,
                empty_hooks_dir=mirror.empty_hooks_dir,
            )
            c_oids = [
                line.decode("ascii", errors="strict").strip()
                for line in proc.stdout.splitlines()
                if line.strip()
            ]
            for c_oid in c_oids:
                if c_oid in seen_commits:
                    continue
                seen_commits.add(c_oid)
                visited_commits.append(c_oid)
                # Focused history only needs cur_path; indexing every file of a wide commit wastes the budget.
                use_filter: set[bytes] | None = None
                with store.connect_read() as r_conn:
                    p_row = r_conn.execute(
                        "SELECT parent_oid FROM commit_parents WHERE commit_oid = ? AND parent_index = 0",
                        (c_oid,),
                    ).fetchone()
                p0_oid = str(p_row["parent_oid"]) if p_row else None
                if p0_oid is None or batch.read_object_bounded(p0_oid, max_bytes=0) is not None:
                    p0_changes = mirror.diff_tree_parent(p0_oid, c_oid, parent_index=0)
                    if len(p0_changes) > WIDE_COMMIT_FILE_THRESHOLD:
                        use_filter = {cur_path}
                _, preds = index_commit_changes_and_hunks(
                    store,
                    mirror,
                    c_oid,
                    filter_paths=use_filter,
                    batch=batch,
                )
                for pred in preds:
                    if pred not in seen_paths:
                        seen_paths.add(pred)
                        paths_queue.append(pred)
                if len(visited_commits) >= max_changes:
                    break

    with store.write_transaction(bump_generation=True) as conn:
        pending_commits = conn.execute(
            "SELECT oid FROM commits WHERE diff_state != 'complete' ORDER BY first_seen_at_us ASC, oid ASC"
        ).fetchall()
        for r in pending_commits:
            c_oid = str(r["oid"])
            Store.enqueue_job(conn, kind="git_commit_diff", owner_id=c_oid, priority=50)
        if pending_commits:
            Store.upsert_gap(
                conn,
                repository_id=store.config.repository_id,
                code="GIT_HISTORY_PENDING",
                scope="git:history",
                message=f"{len(pending_commits)} commit(s) remaining in full Git history backfill.",
                retryable=True,
            )
        else:
            Store.resolve_gaps_in_scope(conn, scope="git:history", code="GIT_HISTORY_PENDING")

    return visited_commits
