import hashlib
import json
import sqlite3
import struct
import subprocess
import time
import zlib

import pytest

from codecontext.errors import CodeContextError
from codecontext.retrieval import check_derivation_cutoff_eligible
from codecontext.store import get_all_migrations
from codecontext.team import execute_join, execute_team_merge, project_remote_key
from tests.fixtures.builders import init_fixture_repo, run_git_fixture
from tests.unit.test_team_share import _add_decision, _join_setup, _merge_setup, _resign


def _memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    for _, sql, _ in get_all_migrations():
        conn.executescript(sql)
    conn.execute("PRAGMA foreign_keys = OFF")
    return conn


def _ready_claim(conn: sqlite3.Connection, claim_id: str) -> None:
    conn.execute(
        "INSERT INTO claims (id, episode_id, model_run_id, statement, support, lifecycle, state,"
        " created_at_us) VALUES (?, 'ep', 'mr', 's', 'direct', 'active', 'ready', 1)",
        (claim_id,),
    )


def _claim_dep(conn: sqlite3.Connection, dependent: str, dep_input: str) -> None:
    conn.execute(
        "INSERT INTO dependencies VALUES ('claim', ?, 'claim', ?)",
        (dependent, dep_input),
    )


def test_derivation_check_on_claim_cycle_fails_closed_without_recursion_error() -> None:
    conn = _memory_db()
    _ready_claim(conn, "claim_a")
    _ready_claim(conn, "claim_b")
    _claim_dep(conn, "claim_a", "claim_b")
    _claim_dep(conn, "claim_b", "claim_a")

    ok, _reason = check_derivation_cutoff_eligible(conn, "claim", "claim_a", None, frozenset())

    assert ok is False


def test_derivation_check_on_layered_diamond_is_linear() -> None:
    conn = _memory_db()
    layers = 22
    for layer in range(layers):
        for side in "lr":
            _ready_claim(conn, f"claim_{layer}_{side}")
    for layer in range(layers - 1):
        for side in "lr":
            for below in "lr":
                _claim_dep(conn, f"claim_{layer}_{side}", f"claim_{layer + 1}_{below}")

    start = time.monotonic()
    result = check_derivation_cutoff_eligible(conn, "claim", "claim_0_l", None, frozenset())
    elapsed = time.monotonic() - start

    assert result == (True, None)
    assert elapsed < 1.0


def test_derivation_check_on_chain_deeper_than_recursion_limit() -> None:
    conn = _memory_db()
    depth = 3000
    for i in range(depth):
        _ready_claim(conn, f"claim_{i}")
    for i in range(depth - 1):
        _claim_dep(conn, f"claim_{i}", f"claim_{i + 1}")

    start = time.monotonic()
    healthy = check_derivation_cutoff_eligible(conn, "claim", "claim_0", None, frozenset())
    conn.execute("INSERT INTO dependencies VALUES ('claim', ?, 'evidence', 'ev_missing')", (f"claim_{depth - 1}",))
    broken = check_derivation_cutoff_eligible(conn, "claim", "claim_0", None, frozenset())
    elapsed = time.monotonic() - start

    assert healthy == (True, None)
    assert broken == (False, "evidence_unavailable")
    assert elapsed < 1.0


def test_merge_does_not_graft_share_link_rows_onto_local_claim(tmp_path) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    local_claim = "claim_" + "a" * 64
    share_claim = "claim_" + "b" * 64
    _add_decision(store_b, local_claim, "local decision")
    _add_decision(share, share_claim, "shared decision")
    with sqlite3.connect(share / "memory.sqlite3") as conn:
        conn.execute(
            "INSERT INTO evidence (id, kind, anchor_json, content_sha256) VALUES ('ev_share', 'commit', '{}', 'h')"
        )
        conn.execute("INSERT INTO dependencies VALUES ('claim', ?, 'evidence', 'ev_missing')", (local_claim,))
        conn.execute("INSERT INTO claim_evidence (claim_id, evidence_id, role) VALUES (?, 'ev_share', 'support')", (local_claim,))
        conn.execute(
            "INSERT INTO relations (id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping)"
            " VALUES ('rel_graft', 'claim', ?, 'evidence', 'ev_share', 'targets', 'm', 'exact')",
            (local_claim,),
        )
        conn.execute("INSERT INTO dependencies VALUES ('claim', ?, 'evidence', 'ev_share')", (share_claim,))
        conn.execute("INSERT INTO claim_evidence (claim_id, evidence_id, role) VALUES (?, 'ev_share', 'support')", (share_claim,))
    _resign(share)

    execute_team_merge(store_b, share, repo_b)

    with sqlite3.connect(store_b / "memory.sqlite3") as conn:
        conn.row_factory = sqlite3.Row
        assert conn.execute("SELECT COUNT(*) FROM dependencies WHERE dependent_id = ?", (local_claim,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM claim_evidence WHERE claim_id = ?", (local_claim,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM relations WHERE id = 'rel_graft'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM dependencies WHERE dependent_id = ?", (share_claim,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM claim_evidence WHERE claim_id = ?", (share_claim,)).fetchone()[0] == 1
        assert check_derivation_cutoff_eligible(conn, "claim", local_claim, None, frozenset()) == (True, None)


def _add_to_manifest(share, rel: str) -> None:
    path = share / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"][rel] = hashlib.sha256((share / rel).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _plant_forged_loose_object(share) -> None:
    claimed_oid = hashlib.sha1(b"blob 5\0real\n").hexdigest()
    rel = f"repository.git/objects/{claimed_oid[:2]}/{claimed_oid[2:]}"
    (share / rel).parent.mkdir(parents=True, exist_ok=True)
    (share / rel).write_bytes(zlib.compress(b"blob 7\0forged\n"))
    _add_to_manifest(share, rel)


def test_join_rejects_loose_object_whose_content_does_not_match_its_name(tmp_path) -> None:
    share, store_b, repo_b = _join_setup(tmp_path)
    _plant_forged_loose_object(share)
    with pytest.raises(CodeContextError) as exc:
        execute_join(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"
    assert not store_b.exists()


def test_merge_rejects_loose_object_whose_content_does_not_match_its_name(tmp_path) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    _plant_forged_loose_object(share)
    with pytest.raises(CodeContextError) as exc:
        execute_team_merge(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"


def _lying_idx(real_idx: bytes, fake_oid: bytes) -> bytes:
    """A one-object v2 index that files the pack's only object under `fake_oid`."""
    body = real_idx[8 + 1024 :]
    crc, offset = body[20:24], body[24:28]
    pack_checksum = real_idx[-40:-20]
    fanout = b"".join(struct.pack(">I", 1 if i >= fake_oid[0] else 0) for i in range(256))
    out = b"\xfftOc" + struct.pack(">I", 2) + fanout + fake_oid + crc + offset + pack_checksum
    return out + hashlib.sha1(out).digest()


def test_join_rebuilds_pack_index_from_pack_content(tmp_path) -> None:
    share, store_b, repo_b = _join_setup(tmp_path)
    scratch = tmp_path / "scratch"
    init_fixture_repo(scratch)
    (scratch / "f").write_bytes(b"forged\n")
    true_oid = run_git_fixture(scratch, ["hash-object", "-w", "f"]).decode().strip()
    prefix = scratch / "pk"
    pack_hash = subprocess.run(
        ["git", "pack-objects", str(prefix)], cwd=scratch, input=f"{true_oid}\n".encode(),
        capture_output=True, check=True,
    ).stdout.decode().strip()
    fake_oid = hashlib.sha1(b"blob 5\0real\n").hexdigest()
    pack_dir = share / "repository.git" / "objects" / "pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    name = f"pack-{pack_hash}"
    (pack_dir / f"{name}.pack").write_bytes((scratch / f"pk-{pack_hash}.pack").read_bytes())
    real_idx = (scratch / f"pk-{pack_hash}.idx").read_bytes()
    (pack_dir / f"{name}.idx").write_bytes(_lying_idx(real_idx, bytes.fromhex(fake_oid)))
    for ext in ("pack", "idx"):
        _add_to_manifest(share, f"repository.git/objects/pack/{name}.{ext}")

    execute_join(store_b, share, repo_b)

    git_dir = store_b / "repository.git"
    forged = subprocess.run(["git", "--git-dir", str(git_dir), "cat-file", "-e", fake_oid], capture_output=True)
    assert forged.returncode != 0
    assert run_git_fixture(repo_b, ["--git-dir", str(git_dir), "cat-file", "-p", true_oid]) == b"forged\n"


def _set_share_store_json(share, **fields) -> None:
    path = share / "store.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.update(fields)
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _resign(share)


def test_join_refuses_share_bound_to_a_different_github_repository(tmp_path) -> None:
    share, store_b, repo_b = _join_setup(tmp_path)
    _set_share_store_json(share, github_repository="other/private")
    with pytest.raises(CodeContextError) as exc:
        execute_join(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"
    assert not store_b.exists()


def test_join_refuses_share_bound_to_a_different_gitlab_project(tmp_path) -> None:
    share, store_b, repo_b = _join_setup(tmp_path)
    _set_share_store_json(
        share, github_repository=None, gitlab_repository="other/private", gitlab_base_url="https://gitlab.com"
    )
    with pytest.raises(CodeContextError) as exc:
        execute_join(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"
    assert not store_b.exists()


def test_join_keeps_matching_binding_and_clears_issue_allowlist(tmp_path) -> None:
    share, store_b, repo_b = _join_setup(tmp_path)
    _set_share_store_json(share, github_repository="Acme/Widget", allowed_issue_repositories=["other/private"])

    execute_join(store_b, share, repo_b)

    raw = json.loads((store_b / "store.json").read_text(encoding="utf-8"))
    assert raw["github_repository"] == "acme/widget"
    assert raw["allowed_issue_repositories"] == []


def test_merge_keeps_share_relation_from_local_commit_to_new_share_source(tmp_path) -> None:
    share, store_b, repo_b = _merge_setup(tmp_path)
    with sqlite3.connect(store_b / "memory.sqlite3") as conn:
        oid = conn.execute("SELECT oid FROM commits LIMIT 1").fetchone()[0]
    with sqlite3.connect(share / "memory.sqlite3") as conn:
        conn.execute(
            "INSERT INTO sources (id, provider, repository_provider_id, provider_id, kind, last_seen_at_us)"
            " VALUES ('src_pr', 'github', '1', '7', 'pull_request', 1)"
        )
        conn.execute(
            "INSERT INTO relations (id, from_kind, from_id, to_kind, to_id, relation_kind, method, mapping)"
            " VALUES ('rel_pr', 'commit', ?, 'source', 'src_pr', 'associated_pr', 'm', 'exact')",
            (oid,),
        )
    _resign(share)

    execute_team_merge(store_b, share, repo_b)

    with sqlite3.connect(store_b / "memory.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM relations WHERE id = 'rel_pr'").fetchone()[0] == 1


def test_join_does_not_copy_commit_graph_or_stray_object_files(tmp_path) -> None:
    share, store_b, repo_b = _join_setup(tmp_path)
    objs = share / "repository.git" / "objects"
    (objs / "info").mkdir(parents=True, exist_ok=True)
    (objs / "info" / "commit-graph").write_bytes(b"forged graph")
    (objs / "stray").write_bytes(b"stray")
    for rel in ("info/commit-graph", "stray"):
        _add_to_manifest(share, f"repository.git/objects/{rel}")

    execute_join(store_b, share, repo_b)

    local = store_b / "repository.git" / "objects"
    assert not (local / "info" / "commit-graph").exists()
    assert not (local / "stray").exists()


def _gitlab_share(tmp_path, base_url: str, origin: str = "https://gitlab.corp/team/app.git"):
    share, store_b, repo_b = _join_setup(tmp_path)
    run_git_fixture(repo_b, ["remote", "set-url", "origin", origin])
    team_path = share / "team.json"
    team = json.loads(team_path.read_text(encoding="utf-8"))
    team["project_remote"] = project_remote_key(origin)
    team_path.write_text(json.dumps(team), encoding="utf-8")
    _set_share_store_json(share, github_repository=None, gitlab_repository="team/app", gitlab_base_url=base_url)
    return share, store_b, repo_b


def test_join_refuses_gitlab_base_url_with_a_port_the_origin_does_not_use(tmp_path) -> None:
    share, store_b, repo_b = _gitlab_share(tmp_path, "https://gitlab.corp:9999")
    with pytest.raises(CodeContextError) as exc:
        execute_join(store_b, share, repo_b)
    assert exc.value.code == "INVALID_REQUEST"


def test_join_keeps_gitlab_base_url_matching_the_origin(tmp_path) -> None:
    share, store_b, repo_b = _gitlab_share(tmp_path, "https://GitLab.corp/")
    execute_join(store_b, share, repo_b)
    raw = json.loads((store_b / "store.json").read_text(encoding="utf-8"))
    assert raw["gitlab_base_url"] == "https://gitlab.corp"
    assert raw["gitlab_repository"] == "team/app"


@pytest.mark.parametrize("origin", ["git@gitlab.corp:team/app.git", "ssh://git@gitlab.corp:2222/team/app.git"])
def test_join_accepts_any_https_port_for_ssh_origin(tmp_path, origin) -> None:
    share, store_b, repo_b = _gitlab_share(tmp_path, "https://gitlab.corp:8443", origin=origin)
    execute_join(store_b, share, repo_b)
    raw = json.loads((store_b / "store.json").read_text(encoding="utf-8"))
    assert raw["gitlab_base_url"] == "https://gitlab.corp:8443"


def test_join_treats_port_443_as_the_https_default(tmp_path) -> None:
    share, store_b, repo_b = _gitlab_share(tmp_path, "https://gitlab.corp:443")
    execute_join(store_b, share, repo_b)
    raw = json.loads((store_b / "store.json").read_text(encoding="utf-8"))
    assert raw["gitlab_base_url"] == "https://gitlab.corp"
