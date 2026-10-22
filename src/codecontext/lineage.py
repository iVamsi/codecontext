import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from codecontext.git import (
    GitMirror,
    ParsedHunk,
    RawChange,
    parse_diff_tree_raw_z,
    parse_raw_commit_object,
    run_safe_git,
)
from codecontext.identity import (
    canonical_json,
    encode_path_b64,
    line_range_to_byte_span,
    make_chg_id,
    make_episode_id,
    make_ev_id,
    make_hunk_id,
    make_relation_id,
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
) -> list[tuple[ExtractedSymbol, bool]]:
    """Map a hunk side (1-based line_start, line_count) against extracted symbols of that blob.
    Returns list of `(symbol, is_primary)`.
    - For line_count > 0: associates every overlapping symbol (`max(s_start, h_start) < min(s_end, h_end)`),
      marking the smallest enclosing/overlapping symbol by byte span as `is_primary=True`.
    - For line_count == 0 (insertion/deletion boundary): associates only symbols whose `body_byte_start..body_byte_end`
      strictly contains or encloses the boundary byte; a boundary between top-level definitions remains file-scoped.
    """
    if not symbols or not blob_bytes:
        return []
    try:
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

    # Primary is the smallest containing/overlapping symbol (by declaration byte length, then byte_start desc)
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
            # Pure insertion in new blob after old line `h.old_start`
            # If insertion occurs strictly inside [line_start, line_end - 1], it modifies the range
            if line_start <= h.old_start < line_end:
                return None
            if h.old_start < line_start:
                delta += h.new_count
        else:
            h_old_start = h.old_start
            h_old_end = h.old_start + h.old_count - 1
            # Check overlap between [line_start, line_end] and [h_old_start, h_old_end]
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
    2. Bounded Git similarity pass (`-M50% --find-renames=50% -l200`) for edited renames -> ('candidate', score)
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
    for blob_oid, add_paths in added_by_blob.items():
        del_paths = deleted_by_blob.get(blob_oid, [])
        if len(add_paths) == 1 and len(del_paths) == 1:
            moves[add_paths[0]] = (del_paths[0], "exact", 100)

    if parent_oid is None or deleted_count == 0 or added_count == 0:
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
            if sch.new_path not in moves:
                score = sch.rename_score if sch.rename_score is not None else 50
                if sch.old_blob_oid and sch.old_blob_oid == sch.new_blob_oid:
                    # Check if unique in added/deleted
                    if (
                        len(added_by_blob.get(sch.new_blob_oid, [])) == 1
                        and len(deleted_by_blob.get(sch.old_blob_oid, [])) == 1
                    ):
                        moves[sch.new_path] = (sch.old_path, "exact", 100)
                    else:
                        moves[sch.new_path] = (sch.old_path, "candidate", score)
                else:
                    moves[sch.new_path] = (sch.old_path, "candidate", score)

    return moves, False


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
    """Apply the ordered mapping procedure from 02-ingestion-and-lineage §6:
    1. Exact original anchor validation
    2. Exact target correspondence (same blob/path/range, or uniquely verified unchanged range transported across hunks / unique exact-byte move)
    3. Candidate correspondence (edited rename, unique lexical symbol/signature association, or bounded patch similarity)
    4. Unmapped when deleted, overwritten, or ambiguous.
    """
    # Resolve original commit tree entry
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
        orig_blob_obj = batch.read_object(orig_entry.oid)
        if orig_blob_obj is None:
            return AnchorMappingResult(
                mapping="unmapped",
                method="missing_original_blob",
                gap_code="MISSING_GIT_OBJECT",
                gap_message=f"Original blob {orig_entry.oid[:12]} is missing.",
            )
        orig_bytes = orig_blob_obj[2]

        line_count = original_line_end - original_line_start + 1
        try:
            orig_b_start, orig_b_end, _, _ = line_range_to_byte_span(
                orig_bytes, original_line_start, line_count
            )
        except ValueError:
            return AnchorMappingResult(
                mapping="unmapped",
                method="invalid_original_range",
                gap_code="MAPPING_UNAVAILABLE",
                gap_message="Original line range exceeds original blob bounds.",
            )

        orig_slice = orig_bytes[orig_b_start:orig_b_end]

        # Inspect target revision tree
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
            t_blob_obj = batch.read_object(t_same_path.oid)
            if t_blob_obj is not None:
                t_bytes = t_blob_obj[2]
                hunks = mirror.diff_blobs_unified_zero(orig_entry.oid, t_same_path.oid)
                transported = transport_line_range_across_hunks(
                    original_line_start, original_line_end, hunks
                )
                if transported is not None:
                    new_l_start, new_l_end = transported
                    try:
                        new_b_start, new_b_end, _, _ = line_range_to_byte_span(
                            t_bytes, new_l_start, new_l_end - new_l_start + 1
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

        # Case 2C: Unique exact-byte file move (original path absent in target, exactly one path in target has orig_entry.oid)
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

        # Case 3: Candidate correspondence via enclosing symbol qualified_name in same path or unique renamed path
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
            # First check same path if present
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

            # Check across active snapshot symbols if original path was moved/edited
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
) -> tuple[int, set[bytes]]:
    """Index parent-specific raw changes, bounded text hunks, changed-blob syntax,
    commit/hunk/code evidence handles, and symbol-to-change relations for `commit_oid`.
    Returns `(changes_indexed, discovered_predecessor_paths)`.
    """
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

    parents: list[tuple[int, str | None]] = (
        [(int(r["parent_index"]), str(r["parent_oid"])) for r in parent_rows]
        if parent_rows
        else [(-1, None)]
    )

    all_raw_changes: list[tuple[RawChange, list[ParsedHunk], bytes | None, bytes | None]] = []
    predecessor_paths: set[bytes] = set()
    missing_parent = False

    with mirror.open_batch_session() as batch:
        commit_obj = batch.read_object(commit_oid)
        parsed_commit = (
            parse_raw_commit_object(commit_oid, commit_obj[2])
            if commit_obj is not None and commit_obj[0] == "commit"
            else None
        )

        for p_idx, p_oid in parents:
            if p_oid is not None and batch.read_object(p_oid) is None:
                missing_parent = True
                # Parent commit object is missing (e.g. shallow boundary)
                with store.write_transaction(bump_generation=True) as conn:
                    Store.upsert_gap(
                        conn,
                        repository_id=repo_id,
                        code="MISSING_PARENT",
                        scope=f"git:commit:{commit_oid}",
                        message=f"Parent commit {p_oid[:12]} of {commit_oid[:12]} is missing.",
                        retryable=True,
                    )
                    conn.execute(
                        "UPDATE commits SET diff_state = 'partial' WHERE oid = ?",
                        (commit_oid,),
                    )
                continue

            raw_changes = mirror.diff_tree_parent(p_oid, commit_oid, parent_index=p_idx)
            moves, _ = detect_commit_renames_and_moves(mirror, p_oid, commit_oid, raw_changes)
            for new_p, (old_p, _, _) in moves.items():
                if filter_paths is None or new_p in filter_paths:
                    predecessor_paths.add(old_p)

            for rch in raw_changes:
                if filter_paths is not None:
                    touches = (rch.new_path in filter_paths) or (rch.old_path in filter_paths)
                    if not touches:
                        continue

                old_bytes: bytes | None = None
                new_bytes: bytes | None = None
                can_diff = True

                if rch.old_blob_oid:
                    ob = batch.read_object(rch.old_blob_oid)
                    if ob is not None and ob[0] == "blob":
                        old_bytes = ob[2]
                        if len(old_bytes) > MAX_PARSE_BYTES or b"\x00" in old_bytes or is_lfs_pointer(old_bytes):
                            can_diff = False
                    else:
                        can_diff = False

                if rch.new_blob_oid:
                    nb = batch.read_object(rch.new_blob_oid)
                    if nb is not None and nb[0] == "blob":
                        new_bytes = nb[2]
                        if len(new_bytes) > MAX_PARSE_BYTES or b"\x00" in new_bytes or is_lfs_pointer(new_bytes):
                            can_diff = False
                    else:
                        can_diff = False

                hunks: list[ParsedHunk] = []
                if can_diff and rch.old_mode != 0o160000 and rch.new_mode != 0o160000:
                    hunks = mirror.diff_blobs_unified_zero(rch.old_blob_oid, rch.new_blob_oid)

                all_raw_changes.append((rch, hunks, old_bytes, new_bytes))

    ts = now_us()
    changes_count = 0
    with store.write_transaction(bump_generation=True) as conn:
        # Ensure commit record exists and register commit-message evidence
        if parsed_commit is not None:
            conn.execute(
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
            conn.execute(
                """
                INSERT INTO evidence (id, kind, source_version_id, git_oid, anchor_json, content_sha256, state)
                VALUES (?, 'commit', NULL, ?, ?, ?, 'available')
                ON CONFLICT(id) DO NOTHING
                """,
                (commit_ev_id, commit_oid, canonical_json(commit_anchor), msg_hash),
            )
            # Ensure commit episode exists if no PR episode claims it yet
            ep_id = make_episode_id(repository_id=repo_id, commit_oid=commit_oid)
            conn.execute(
                """
                INSERT INTO episodes (id, kind, primary_source_id, commit_oid, input_fingerprint, state)
                VALUES (?, 'commit', NULL, ?, ?, 'pending')
                ON CONFLICT(id) DO NOTHING
                """,
                (ep_id, commit_oid, msg_hash),
            )

        changed_path_terms: list[str] = []
        changed_symbol_names: list[str] = []

        for rch, hunks, old_bytes, new_bytes in all_raw_changes:
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
            h_state = "complete" if (old_bytes is not None or new_bytes is not None) else "unavailable"
            conn.execute(
                """
                INSERT INTO changes (
                    id, commit_oid, parent_index, parent_oid, old_path, new_path,
                    old_blob_oid, new_blob_oid, old_mode, new_mode,
                    change_kind, rename_score, hunks_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET hunks_state = excluded.hunks_state
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

            # Extract symbols from old_blob and new_blob if supported language
            old_syms: list[ExtractedSymbol] = []
            new_syms: list[ExtractedSymbol] = []

            if rch.old_path is not None and rch.old_blob_oid is not None and old_bytes is not None:
                old_disp, _ = path_bytes_to_display(rch.old_path)
                changed_path_terms.append(old_disp)
                lang = detect_language_for_path(rch.old_path)
                if lang is not None and len(old_bytes) <= MAX_PARSE_BYTES and b"\x00" not in old_bytes:
                    try:
                        old_bytes.decode("utf-8", errors="strict")
                        old_res, _ = index_blob_into_store(
                            conn,
                            repository_id=repo_id,
                            blob_oid=rch.old_blob_oid,
                            blob_bytes=old_bytes,
                            language=lang,
                        )
                        old_syms = old_res.symbols
                    except UnicodeDecodeError:
                        pass

            if rch.new_path is not None and rch.new_blob_oid is not None and new_bytes is not None:
                new_disp, new_p_b64 = path_bytes_to_display(rch.new_path)
                changed_path_terms.append(new_disp)
                lang = detect_language_for_path(rch.new_path)
                if lang is not None and len(new_bytes) <= MAX_PARSE_BYTES and b"\x00" not in new_bytes:
                    try:
                        new_bytes.decode("utf-8", errors="strict")
                        new_res, _ = index_blob_into_store(
                            conn,
                            repository_id=repo_id,
                            blob_oid=rch.new_blob_oid,
                            blob_bytes=new_bytes,
                            language=lang,
                        )
                        new_syms = new_res.symbols
                    except UnicodeDecodeError:
                        pass

            for hk in hunks:
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
                conn.execute(
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

                # Create hunk evidence record
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
                conn.execute(
                    """
                    INSERT INTO evidence (id, kind, source_version_id, git_oid, anchor_json, content_sha256, state)
                    VALUES (?, 'hunk', NULL, ?, ?, ?, 'available')
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (hunk_ev_id, commit_oid, canonical_json(hunk_anchor), p_hash),
                )

                # Associate hunk with old and new symbols
                if old_bytes is not None and old_syms:
                    for sym, is_prim in associate_hunk_side_with_symbols(
                        old_bytes, hk.old_start, hk.old_count, old_syms
                    ):
                        changed_symbol_names.append(sym.qualified_name)
                        rel_id = make_relation_id(
                            from_kind="change",
                            from_id=chg_id,
                            to_kind="symbol",
                            to_id=sym.id,
                            relation_kind="changes",
                            method="hunk_old_overlap",
                            anchor_details={"hunk_id": hunk_id, "primary": is_prim, "side": "old"},
                        )
                        conn.execute(
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
                        new_bytes, hk.new_start, hk.new_count, new_syms
                    ):
                        changed_symbol_names.append(sym.qualified_name)
                        # Also register exact code evidence for the modified/introduced symbol at this commit
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
                        conn.execute(
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
                        conn.execute(
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

                # Index change in FTS `history` partition
                if parsed_commit is not None:
                    disp_p = (
                        path_bytes_to_display(rch.new_path)[0]
                        if rch.new_path is not None
                        else (path_bytes_to_display(rch.old_path)[0] if rch.old_path is not None else "")
                    )
                    b64_p = (
                        path_bytes_to_display(rch.new_path)[1]
                        if rch.new_path is not None
                        else (path_bytes_to_display(rch.old_path)[1] if rch.old_path is not None else None)
                    )
                    title_line = parsed_commit.message_text.splitlines()[0] if parsed_commit.message_text else commit_oid[:12]
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
                            chg_id,
                            hk.ordinal,
                            normalize_search_text(f"{title_line} {disp_p}"),
                            normalize_search_text(" ".join([disp_p, *changed_symbol_names])),
                            normalize_search_text(f"{parsed_commit.message_text} {hk.patch_text}"),
                            meta_json,
                        ),
                    )

        if filter_paths is None:
            conn.execute(
                "UPDATE commits SET diff_state = ? WHERE oid = ?",
                ("partial" if missing_parent else "complete", commit_oid),
            )
            if not missing_parent:
                Store.resolve_gaps_in_scope(conn, scope=f"git:commit:{commit_oid}", code="MISSING_PARENT")

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

    while paths_queue and len(visited_commits) < max_changes:
        cur_path = paths_queue.pop(0)
        rem = max_changes - len(visited_commits)
        proc = run_safe_git(
            [
                "rev-list",
                "--topo-order",
                f"--max-count={rem}",
                active_revision,
                "--",
                os.fsdecode(cur_path) if hasattr(os, "fsdecode") else cur_path.decode("utf-8", errors="surrogateescape"),
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
            _, preds = index_commit_changes_and_hunks(store, mirror, c_oid)
            for pred in preds:
                if pred not in seen_paths:
                    seen_paths.add(pred)
                    paths_queue.append(pred)
            if len(visited_commits) >= max_changes:
                break

    # Enqueue remaining pending commits in `jobs` so full backfill is preserved
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
