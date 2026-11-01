import json
import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from codecontext import workspace
from codecontext.docs_ingest import classify_doc_path
from codecontext.errors import CodeContextError
from codecontext.git import PYTHON_DIFF_MAX_LINES, GitMirror, compute_unified_zero_hunks
from codecontext.github import GitHubClient, collect_full_and_incremental_github
from codecontext.gitlab import GitLabClient, collect_gitlab_merge_requests_and_issues
from codecontext.lineage import index_commit_changes_and_hunks
from codecontext.models import InitRequest, SyncRequest
from codecontext.store import Store
from codecontext.workspace import (
    IndexStatEntry,
    index_entry_matches_stat,
    parse_git_index,
    read_git_index_stat_cache,
)
from tests.fixtures.builders import (
    GitCommitSpec,
    commit_fixture_spec,
    init_fixture_repo,
    run_git_fixture,
)


def _repo_with_files(tmp_path: Path, files: dict[str, bytes]) -> Path:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="init", files=files))
    return repo


# --- GitHub GraphQL ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "document",
    [
        "mutation { addStar(input: {}) { clientMutationId } }",
        "# leading comment\nmutation M { x }",
        "query Q { viewer { login } } mutation M { x }",
        "subscription { x }",
    ],
)
def test_graphql_query_rejects_write_operations(document: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request may be sent for a write operation")

    with GitHubClient(token="", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(CodeContextError) as exc:
            client.graphql_query(document)
    assert exc.value.code == "INVALID_REQUEST"
    assert client.requests_attempted == 0


def test_graphql_query_retries_and_raises_on_graphql_errors() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if len(calls) == 1:
            return httpx.Response(502)
        return httpx.Response(200, json={"errors": [{"message": "Field 'x' doesn't exist"}]})

    with GitHubClient(token="", transport=httpx.MockTransport(handler), sleep_fn=lambda _s: None) as client:
        with pytest.raises(CodeContextError) as exc:
            client.graphql_query("query { x }")
    assert exc.value.code == "IO_ERROR"
    assert calls == ["POST", "POST"]


def test_graphql_failure_falls_back_to_rest(tmp_path: Path) -> None:
    repo = _repo_with_files(tmp_path, {"a.py": b"x = 1\n"})
    store = Store.initialize(tmp_path / "store", InitRequest(repo_path=repo, github_repository="owner/repo"))
    paths: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode("ascii")
        paths.append((request.method, path))
        if path == "/repos/owner/repo":
            return httpx.Response(200, json={"id": 77, "full_name": "owner/repo", "private": False})
        if path == "/graphql":
            return httpx.Response(200, json={"errors": [{"message": "boom"}]})
        if path.startswith("/repos/owner/repo/"):
            return httpx.Response(200, json=[])
        return httpx.Response(404, json={})

    with GitHubClient(token="", transport=httpx.MockTransport(handler)) as client:
        _used, _sa, _va, done = collect_full_and_incremental_github(store, client, use_graphql=True)
    store.close()
    assert done is True
    gql_at = paths.index(("POST", "/graphql"))
    assert any(p.startswith("/repos/owner/repo/issues?") for _m, p in paths[gql_at + 1 :])


def test_sync_defaults_to_rest() -> None:
    assert SyncRequest().use_graphql is False


def _graphql_store_and_handler(tmp_path: Path):
    repo = _repo_with_files(tmp_path, {"a.py": b"x = 1\n"})
    store = Store.initialize(tmp_path / "store", InitRequest(repo_path=repo, github_repository="owner/repo"))
    gql_cursors: list[str | None] = []
    rest_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode("ascii")
        if path == "/repos/owner/repo":
            return httpx.Response(200, json={"id": 77, "full_name": "owner/repo", "private": False})
        if path == "/graphql":
            gql_cursors.append(json.loads(request.content)["variables"]["cursor"])
            nodes = [{"number": 1, "title": "pr", "body": "", "updatedAt": "2026-01-01T00:00:00Z"}]
            return httpx.Response(
                200,
                json={"data": {"repository": {"pullRequests": {
                    "pageInfo": {"hasNextPage": False, "endCursor": "C1"},
                    "nodes": nodes if len(gql_cursors) == 1 else [],
                }}}},
            )
        rest_paths.append(path)
        if path.startswith("/repos/owner/repo/issues?"):
            return httpx.Response(200, json=[{"number": 5, "title": "bug", "body": "", "user": {"id": 1, "login": "a"}}])
        if path.startswith("/repos/owner/repo/"):
            return httpx.Response(200, json=[])
        return httpx.Response(404, json={})

    return store, handler, gql_cursors, rest_paths


def test_graphql_success_still_collects_issues_via_rest(tmp_path: Path) -> None:
    store, handler, _cursors, rest_paths = _graphql_store_and_handler(tmp_path)
    with GitHubClient(token="", transport=httpx.MockTransport(handler)) as client:
        _used, _sa, _va, done = collect_full_and_incremental_github(store, client, use_graphql=True)
    with store.connect_read() as conn:
        kinds = {(r["kind"], r["provider_id"]) for r in conn.execute("SELECT kind, provider_id FROM sources")}
    store.close()
    assert done is True
    assert {("pull_request", "1"), ("issue", "5")} <= kinds
    assert any(p.startswith("/repos/owner/repo/pulls/comments?") for p in rest_paths)


def test_second_graphql_sync_resumes_from_saved_cursor(tmp_path: Path) -> None:
    store, handler, cursors, _rest = _graphql_store_and_handler(tmp_path)
    for _ in range(2):
        with GitHubClient(token="", transport=httpx.MockTransport(handler)) as client:
            collect_full_and_incremental_github(store, client, use_graphql=True)
    store.close()
    assert cursors == [None, "C1"]


def test_gitlab_resume_does_not_drop_page_when_saved_mr_moved(tmp_path: Path) -> None:
    repo = _repo_with_files(tmp_path, {"a.py": b"x = 1\n"})
    store = Store.initialize(tmp_path / "store", InitRequest(repo_path=repo))
    moved = False
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode("ascii")
        calls.append(path)
        if path == "/api/v4/projects/group%2Fapp":
            return httpx.Response(200, json={"id": 9, "visibility": "public"})
        if "/merge_requests?" in path:
            mr1 = {"iid": 1, "title": "one", "updated_at": "2026-02-09T00:00:00Z" if moved else "2026-02-01T00:00:00Z"}
            rest = [
                {"iid": 2, "title": "two", "updated_at": "2026-02-02T00:00:00Z"},
                {"iid": 3, "title": "three", "updated_at": "2026-02-03T00:00:00Z"},
            ]
            return httpx.Response(200, json=[*rest, mr1] if moved else [mr1, *rest])
        if "/discussions?" in path or "/issues?" in path:
            return httpx.Response(200, json=[])
        return httpx.Response(404, json={})

    transport = httpx.MockTransport(handler)
    # Budget 3: project, MR list, MR 1 discussions; pauses before MR 2's discussions.
    with GitLabClient(base_url="https://gitlab.com", token="", transport=transport) as client:
        _u, _s, _v, done = collect_gitlab_merge_requests_and_issues(
            store, client, project_path="group/app", remaining_requests=3
        )
    assert done is False
    with store.connect_read() as conn:
        wm = conn.execute(
            "SELECT high_watermark_us FROM checkpoints WHERE stream_key LIKE '%merge_requests_wm'"
        ).fetchone()
    assert wm["high_watermark_us"] == 1769904000000000  # MR 1 only, not the page max

    moved = True
    calls.clear()
    with GitLabClient(base_url="https://gitlab.com", token="", transport=transport) as client:
        _u, _s, _v, done = collect_gitlab_merge_requests_and_issues(
            store, client, project_path="group/app", remaining_requests=50
        )
    store.close()
    assert done is True
    assert "&updated_after=2026-02-01T00%3A00%3A00Z" in next(c for c in calls if "/merge_requests?" in c)
    discussed = [c.split("/merge_requests/")[1].split("/")[0] for c in calls if "/discussions?" in c]
    assert discussed == ["2", "3", "1"]


# --- GitLab incremental watermarks ------------------------------------------------------------


def test_gitlab_incremental_sync_uses_updated_after_watermark(tmp_path: Path) -> None:
    repo = _repo_with_files(tmp_path, {"a.py": b"x = 1\n"})
    store = Store.initialize(tmp_path / "store", InitRequest(repo_path=repo))
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode("ascii")
        calls.append(path)
        if path == "/api/v4/projects/group%2Fapp":
            return httpx.Response(200, json={"id": 9, "visibility": "public"})
        if "/merge_requests?" in path:
            return httpx.Response(
                200,
                json=[
                    {"iid": 1, "title": "one", "updated_at": "2026-02-01T00:00:00Z"},
                    {"iid": 2, "title": "two", "updated_at": "2026-02-03T04:05:06Z"},
                ],
            )
        if "/discussions?" in path:
            return httpx.Response(200, json=[])
        if "/issues?" in path:
            return httpx.Response(200, json=[{"iid": 5, "title": "bug", "updated_at": "2026-03-01T00:00:00Z"}])
        return httpx.Response(404, json={})

    transport = httpx.MockTransport(handler)
    for _ in range(2):
        with GitLabClient(base_url="https://gitlab.com", token="", transport=transport) as client:
            _u, _s, _v, done = collect_gitlab_merge_requests_and_issues(
                store, client, project_path="group/app", remaining_requests=50
            )
        assert done is True

    mr_lists = [c for c in calls if "/merge_requests?" in c]
    issue_lists = [c for c in calls if "/issues?" in c]
    assert all("order_by=updated_at&sort=asc" in c for c in mr_lists + issue_lists)
    assert "updated_after" not in mr_lists[0]
    assert mr_lists[1].endswith("&updated_after=2026-02-03T04%3A05%3A06Z")
    assert "updated_after" not in issue_lists[0]
    assert issue_lists[1].endswith("&updated_after=2026-03-01T00%3A00%3A00Z")

    with store.connect_read() as conn:
        providers = {r[0] for r in conn.execute("SELECT DISTINCT provider FROM sources")}
        meta = conn.execute("SELECT gitlab_project_id FROM store_meta WHERE singleton = 1").fetchone()
    store.close()
    assert providers == {"gitlab"}
    assert meta["gitlab_project_id"] == 9


# --- Git index stat cache ---------------------------------------------------------------------


def _ls_files_stage(repo: Path) -> dict[bytes, tuple[int, str]]:
    out: dict[bytes, tuple[int, str]] = {}
    for rec in run_git_fixture(repo, ["ls-files", "--stage", "-z"]).split(b"\0"):
        if rec:
            meta, path = rec.split(b"\t", 1)
            mode, oid, _stage = meta.split()
            out[path] = (int(mode, 8), oid.decode())
    return out


@pytest.mark.parametrize("version", ["2", "4"])
def test_parse_git_index_matches_ls_files(tmp_path: Path, version: str) -> None:
    repo = _repo_with_files(
        tmp_path,
        {"src/pkg/alpha.py": b"a\n", "src/pkg/beta.py": b"bb\n", "src/z.py": b"z\n", "top.md": b"t\n"},
    )
    run_git_fixture(repo, ["update-index", f"--index-version={version}"])
    entries, _mtime = read_git_index_stat_cache(repo / ".git" / "index", oid_bytes=20)
    assert {p: (e.mode, e.oid) for p, e in entries.items()} == _ls_files_stage(repo)
    st = os.lstat(repo / "src" / "z.py")
    assert entries[b"src/z.py"].size == st.st_size


def test_parse_git_index_skips_intent_to_add_entries(tmp_path: Path) -> None:
    repo = _repo_with_files(tmp_path, {"a.py": b"a\n"})
    (repo / "new.py").write_bytes(b"new\n")
    run_git_fixture(repo, ["add", "-N", "new.py"])
    entries, _ = read_git_index_stat_cache(repo / ".git" / "index", oid_bytes=20)
    assert set(entries) == {b"a.py"}


def test_corrupt_index_returns_none(tmp_path: Path) -> None:
    repo = _repo_with_files(tmp_path, {"a.py": b"a\n", "b.py": b"b\n"})
    good = (repo / ".git" / "index").read_bytes()
    bad = tmp_path / "bad-index"
    for data in (b"", b"XXXX" + good[4:], good[:40], good[:12] + b"\xff" * (len(good) - 12)):
        bad.write_bytes(data)
        assert read_git_index_stat_cache(bad, oid_bytes=20) is None
    with pytest.raises(ValueError):
        parse_git_index(good[:8] + (10_000).to_bytes(4, "big") + good[12:], oid_bytes=20)


def test_index_entry_matches_stat_rejects_racily_clean(tmp_path: Path) -> None:
    f = tmp_path / "f.txt"
    f.write_bytes(b"hello")
    st = os.lstat(f)
    entry = IndexStatEntry(ctime_ns=0, mtime_ns=st.st_mtime_ns, size=5, mode=0o100644, oid="0" * 40)
    assert index_entry_matches_stat(entry, st, st.st_mtime_ns + 1)
    assert not index_entry_matches_stat(entry, st, st.st_mtime_ns)
    assert not index_entry_matches_stat(entry, st, st.st_mtime_ns - 1)
    assert not index_entry_matches_stat(
        IndexStatEntry(ctime_ns=0, mtime_ns=st.st_mtime_ns, size=6, mode=0o100644, oid="0" * 40),
        st,
        st.st_mtime_ns + 1,
    )


def _record_reads(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    reads: list[bytes] = []
    real = workspace._read_root_confined_regular_file

    def recording(root: Path, rel: bytes, **kwargs: object):
        reads.append(rel)
        return real(root, rel, **kwargs)

    monkeypatch.setattr(workspace, "_read_root_confined_regular_file", recording)
    return reads


def _stat_cached_repo(tmp_path: Path) -> tuple[Path, Store, GitMirror, str]:
    repo = _repo_with_files(tmp_path, {"a.py": b"a = 1\n", "b.py": b"b = 1\n"})
    past = time.time() - 1000
    for name in ("a.py", "b.py"):
        os.utime(repo / name, (past, past))
    run_git_fixture(repo, ["update-index", "--refresh"])
    store = Store.initialize(tmp_path / "store", InitRequest(repo_path=repo))
    mirror = GitMirror(store)
    head = mirror.prepare_snapshot("HEAD")
    return repo, store, mirror, head


def test_unchanged_files_skip_working_tree_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, store, mirror, head = _stat_cached_repo(tmp_path)
    (repo / "b.py").write_bytes(b"b = 22\n")
    reads = _record_reads(monkeypatch)
    snap, _ = mirror._prepare_workspace_revision(repo, head)
    mirror.close()
    store.close()
    assert reads == [b"b.py"]
    assert snap != head


def test_racily_clean_file_is_reread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, store, mirror, head = _stat_cached_repo(tmp_path)
    future = time.time() + 1000
    os.utime(repo / "a.py", (future, future))
    reads = _record_reads(monkeypatch)
    snap, _ = mirror._prepare_workspace_revision(repo, head)
    mirror.close()
    store.close()
    assert reads == [b"a.py"]
    assert snap == head


def test_unusable_index_falls_back_to_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, store, mirror, head = _stat_cached_repo(tmp_path)

    def unusable(_path: Path, *, oid_bytes: int) -> None:
        return None

    monkeypatch.setattr(workspace, "read_git_index_stat_cache", unusable)
    reads = _record_reads(monkeypatch)
    snap, _ = mirror._prepare_workspace_revision(repo, head)
    mirror.close()
    store.close()
    assert sorted(reads) == [b"a.py", b"b.py"]
    assert snap == head


# --- First-parent merge lineage ---------------------------------------------------------------


def test_merge_commit_is_diffed_against_first_parent_only(tmp_path: Path) -> None:
    repo = _repo_with_files(tmp_path, {"a.py": b"a = 1\n"})
    run_git_fixture(repo, ["checkout", "-b", "side"])
    commit_fixture_spec(repo, GitCommitSpec(message="side", files={"b.py": b"b = 1\n"}))
    run_git_fixture(repo, ["checkout", "main"])
    commit_fixture_spec(repo, GitCommitSpec(message="main", files={"a.py": b"a = 2\n"}))
    run_git_fixture(repo, ["merge", "--no-ff", "side", "-m", "Merge side"])
    merge_oid = run_git_fixture(repo, ["rev-parse", "HEAD"]).decode().strip()

    store = Store.initialize(tmp_path / "store", InitRequest(repo_path=repo))
    mirror = GitMirror(store)
    mirror.prepare_snapshot("HEAD")
    index_commit_changes_and_hunks(store, mirror, merge_oid)
    with store.connect_read() as conn:
        rows = conn.execute(
            "SELECT parent_index, new_path FROM changes WHERE commit_oid = ?", (merge_oid,)
        ).fetchall()
    mirror.close()
    store.close()
    assert [(r["parent_index"], bytes(r["new_path"])) for r in rows] == [(0, b"b.py")]


# --- Docs classification ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("doc/architecture/overview.md", "design_doc"),
        ("docs/architecture/storage.md", "design_doc"),
        ("docs/adr/0001-sqlite.md", "adr"),
        ("ARCHITECTURE.md", "design_doc"),
        ("DESIGN.md", "design_doc"),
        ("doc/architecture/diagram.png", None),
        ("src/architecture/notes.md", None),
    ],
)
def test_classify_doc_path(path: str, kind: str | None) -> None:
    assert classify_doc_path(path) == kind


def test_graphql_request_body_is_plain_query(tmp_path: Path) -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"data": {"viewer": {"login": "a"}}})

    with GitHubClient(token="", transport=httpx.MockTransport(handler)) as client:
        res = client.graphql_query("query { viewer { login } }")
    assert res.data["data"]["viewer"]["login"] == "a"
    assert bodies == [{"query": "query { viewer { login } }", "variables": {}}]


# --- Bounded hunk diffing ---------------------------------------------------------------------


def _git_plumbing(repo: Path, args: list[str], stdin: bytes | None = None) -> bytes:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "LC_ALL": "C",
    }
    return subprocess.run(
        ["git", *args], cwd=repo, env=env, input=stdin, capture_output=True, check=True
    ).stdout


def _commit_bytes_tree(repo: Path, files: dict[bytes, bytes], parent: str | None) -> str:
    """Commit through plumbing so paths that the filesystem rejects (non-UTF-8 on APFS) still work."""
    entries = b""
    for path, content in sorted(files.items()):
        oid = _git_plumbing(repo, ["hash-object", "-w", "--stdin"], content).strip()
        entries += b"100644 blob " + oid + b"\t" + path + b"\x00"
    tree = _git_plumbing(repo, ["mktree", "-z"], entries).decode().strip()
    parent_args = ["-p", parent] if parent else []
    commit = _git_plumbing(repo, ["commit-tree", tree, *parent_args, "-m", "c"]).decode().strip()
    _git_plumbing(repo, ["update-ref", "refs/heads/main", commit])
    return commit


def _repetitive_lockfile(lines: int, *, bump_every: int | None = None) -> bytes:
    out = []
    for i in range(lines):
        version = 2 if bump_every and i % bump_every == 0 else 1
        out.append(f'  "dep-{i % 7}": "^{version}.0.0",\n')
    return "".join(out).encode()


def _two_blob_mirror(tmp_path: Path, path: bytes, old: bytes, new: bytes):
    repo = init_fixture_repo(tmp_path / "repo")
    c0 = _commit_bytes_tree(repo, {path: old}, None)
    c1 = _commit_bytes_tree(repo, {path: new}, c0)
    store = Store.initialize(tmp_path / "store", InitRequest(repo_path=repo))
    mirror = GitMirror(store)
    mirror.prepare_snapshot("HEAD")
    (change,) = mirror.diff_tree_parent(c0, c1, parent_index=0)
    return store, mirror, c1, change


def test_large_repetitive_diff_is_fast(tmp_path: Path) -> None:
    old = _repetitive_lockfile(20_000)
    new = _repetitive_lockfile(20_000, bump_every=97)
    store, mirror, _, change = _two_blob_mirror(tmp_path, b"package-lock.json", old, new)
    try:
        start = time.monotonic()
        hunks = compute_unified_zero_hunks(
            old, new, mirror=mirror, old_oid=change.old_blob_oid, new_oid=change.new_blob_oid
        )
        assert time.monotonic() - start < 2.0
    finally:
        mirror.close()
        store.close()
    assert hunks is not None
    assert sum(h.new_count for h in hunks) == len(range(0, 20_000, 97))


def test_large_diff_without_mirror_is_unavailable() -> None:
    old = _repetitive_lockfile(PYTHON_DIFF_MAX_LINES + 1)
    new = _repetitive_lockfile(PYTHON_DIFF_MAX_LINES + 1, bump_every=97)
    start = time.monotonic()
    assert compute_unified_zero_hunks(old, new) is None
    assert time.monotonic() - start < 1.0


def test_large_and_small_paths_agree(tmp_path: Path) -> None:
    # Unique lines leave only one minimal diff, so SequenceMatcher and git's Myers must agree;
    # on repetitive input both are valid but may place hunks differently.
    old_lines = [f"line {i}\n".encode() for i in range(60)]
    new_lines = list(old_lines)
    new_lines[3] = b"changed 3\n"
    del new_lines[10:13]
    new_lines.insert(20, b"inserted\n")
    new_lines[30:32] = [b"a\n", b"b\n", b"c\n"]
    old = b"".join(old_lines)
    new = b"".join(new_lines)[:-1]  # drop the final newline to cover the marker line
    store, mirror, _, change = _two_blob_mirror(tmp_path, b"f.txt", old, new)
    try:
        via_git = mirror.diff_blobs_unified_zero(change.old_blob_oid, change.new_blob_oid)
    finally:
        mirror.close()
        store.close()
    via_python = compute_unified_zero_hunks(old, new)
    assert via_python == via_git
    assert len(via_git) == 5


def test_non_utf8_path_diff_round_trips(tmp_path: Path) -> None:
    path = b"caf\xe9.py"
    old = _repetitive_lockfile(PYTHON_DIFF_MAX_LINES * 2)
    new = _repetitive_lockfile(PYTHON_DIFF_MAX_LINES * 2, bump_every=500)
    store, mirror, c1, _ = _two_blob_mirror(tmp_path, path, old, new)
    try:
        index_commit_changes_and_hunks(store, mirror, c1)
        with store.connect_read() as conn:
            changes = conn.execute(
                "SELECT id, new_path, hunks_state FROM changes WHERE commit_oid = ?", (c1,)
            ).fetchall()
            hunk_count = conn.execute(
                "SELECT COUNT(*) FROM hunks WHERE change_id = ?", (changes[0]["id"],)
            ).fetchone()[0]
    finally:
        mirror.close()
        store.close()
    assert {bytes(r["new_path"]) for r in changes} == {path}
    assert changes[0]["hunks_state"] == "complete"
    assert hunk_count == len(range(0, PYTHON_DIFF_MAX_LINES * 2, 500))
