import hashlib
import json
import re
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any

import httpx
import pytest

from codecontext.errors import CodeContextError
from codecontext.git import GitMirror
from codecontext.github import (
    GITHUB_API_VERSION,
    GitHubClient,
    collect_focused_github_evidence,
    collect_full_and_incremental_github,
    upsert_github_source_and_version,
    verify_and_bind_github_repository,
)
from codecontext.gitlab import GitLabClient, parse_gitlab_remote_url, resolve_gitlab_token
from codecontext.identity import make_src_id, us_to_iso_z
from codecontext.models import InitRequest
from codecontext.store import Store
from tests.fixtures.builders import (
    GitCommitSpec,
    build_f01_fixture,
    commit_fixture_spec,
    init_fixture_repo,
)


def make_mock_github_transport(
    routes: dict[str, tuple[int, Any, dict[str, str]]],
    *,
    expected_user: dict[str, Any] | None = None,
) -> tuple[httpx.MockTransport, list[str]]:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("X-GitHub-Api-Version") == GITHUB_API_VERSION
        assert request.headers.get("Accept") == "application/vnd.github+json"
        path_with_q = request.url.raw_path.decode("ascii")
        requested_paths.append(path_with_q)

        if path_with_q == "/user":
            if expected_user is None:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json=expected_user)

        if path_with_q in routes:
            status, body, headers = routes[path_with_q]
            if isinstance(body, (dict, list)):
                return httpx.Response(status, json=body, headers=headers)
            return httpx.Response(status, content=body, headers=headers)

        return httpx.Response(404, json={"message": "Not Found"})

    return httpx.MockTransport(handler), requested_paths


def test_f01_focused_github_acquisition_pagination_and_resume(tmp_path: Path) -> None:
    """T07: Focus sync acquires F01/F02 PR discussions, review comments, one-hop allowed issues,
    and resumes cleanly after an interrupted page budget without duplicating sources or versions.
    """
    f01 = build_f01_fixture(tmp_path / "f01")
    store = Store.initialize(
        tmp_path / "store",
        InitRequest(
            repo_path=f01.repo_dir,
            github_repository="example/transport",
            allowed_issue_repositories=["example/shared-issues"],
        ),
    )
    mirror = GitMirror(store)
    mirror.prepare_snapshot("HEAD")

    # Add an external issue link into comment 1 body + a disallowlisted repo link
    comments_page_1 = [
        {
            **f01.comments_payload[0],
            "body": f01.comments_payload[0]["body"] + " See example/shared-issues#77 and evil/other#99.",
        }
    ]
    comments_page_2 = [f01.comments_payload[1]]

    inline_review_comments = [
        {
            "id": 40001,
            "pull_request_review_id": 30001,
            "body": "Three retries bounds worker occupancy to 3x timeout.",
            "path": "src/transport.py",
            "commit_id": f01.c1_oid,
            "original_commit_id": f01.c1_oid,
            "line": 1,
            "original_line": 1,
            "side": "RIGHT",
            "diff_hunk": "@@ -0,0 +1,2 @@\n+MAX_RETRIES = 3",
            "user": {"id": 503, "login": "carol"},
            "created_at": "2026-01-02T09:40:00Z",
            "updated_at": "2026-01-02T09:40:00Z",
            "html_url": "https://github.com/example/transport/pull/10#discussion_r40001",
        }
    ]

    routes: dict[str, tuple[int, Any, dict[str, str]]] = {
        "/repos/example/transport": (
            200,
            {"id": 9001, "full_name": "example/transport", "private": False},
            {},
        ),
        f"/repos/example/transport/commits/{f01.c1_oid}/pulls": (
            200,
            [{"number": 10, "id": 10010}],
            {},
        ),
        "/repos/example/transport/pulls/10": (200, f01.pr_payload, {}),
        "/repos/example/transport/issues/10/comments?per_page=100": (
            200,
            comments_page_1,
            {
                "Link": '<https://api.github.com/repos/example/transport/issues/10/comments?per_page=100&page=2>; rel="next"'
            },
        ),
        "/repos/example/transport/issues/10/comments?per_page=100&page=2": (
            200,
            comments_page_2,
            {},
        ),
        "/repos/example/transport/pulls/10/reviews?per_page=100": (
            200,
            f01.reviews_payload,
            {},
        ),
        "/repos/example/transport/pulls/10/comments?per_page=100": (
            200,
            inline_review_comments,
            {},
        ),
        "/repos/example/transport/issues/10/timeline?per_page=100": (200, [], {}),
        "/repos/example/transport/pulls/10/commits?per_page=100": (
            200,
            [{"sha": f01.c1_oid}],
            {},
        ),
        "/repos/example/transport/pulls/10/files?per_page=100": (200, [], {}),
        "/repos/example/shared-issues/issues/77": (
            200,
            {
                "id": 77001,
                "number": 77,
                "state": "open",
                "title": "Proxy duplicates timed-out requests",
                "body": "Legacy proxy retries automatically after 30s timeout.",
                "user": {"id": 504, "login": "dave"},
                "created_at": "2026-01-01T08:00:00Z",
                "updated_at": "2026-01-01T08:00:00Z",
                "html_url": "https://github.com/example/shared-issues/issues/77",
            },
            {},
        ),
    }

    transport, req_paths = make_mock_github_transport(
        routes, expected_user={"id": 7001, "login": "operator"}
    )

    # First call: interrupt mid-way after 4 requests (user + repo + commit_pulls + pr_detail + page 1 of comments = 5; budget 4 stops before page 2)
    with GitHubClient(token="ghp_testtoken1", transport=transport) as client:
        used1, s_add1, v_add1, done1 = collect_focused_github_evidence(
            store,
            client,
            commit_oids=[f01.c1_oid],
            remaining_requests=5,
        )
        assert used1 == 5
        assert done1 is False
        assert s_add1 == 2  # PR #10 + comment 1

    # Resume with remaining budget: should start at page 2 of issue_comments without re-fetching page 1!
    with GitHubClient(token="ghp_testtoken1", transport=transport) as client:
        used2, s_add2, v_add2, done2 = collect_focused_github_evidence(
            store,
            client,
            commit_oids=[f01.c1_oid],
            remaining_requests=20,
        )
        assert done2 is True
        # Page 1 was NOT requested a second time
        assert req_paths.count("/repos/example/transport/issues/10/comments?per_page=100") == 1
        assert req_paths.count("/repos/example/transport/issues/10/comments?per_page=100&page=2") == 1
        # Disallowlisted repo `evil/other#99` was never requested
        assert not any("evil/other" in p for p in req_paths)
        # Allowlisted repo `example/shared-issues#77` was fetched
        assert "/repos/example/shared-issues/issues/77" in req_paths

    # Verify total sources & versions in SQLite (1 PR + 2 issue_comments + 1 review + 1 review_comment + 1 allowlisted issue = 6 sources)
    with store.connect_read() as conn:
        src_count = int(conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0])
        ver_count = int(conn.execute("SELECT COUNT(*) FROM source_versions").fetchone()[0])
        assert src_count == 6
        assert ver_count == 6

    # Repeat sync with unchanged payloads -> 0 new sources, 0 new versions
    with GitHubClient(token="ghp_testtoken1", transport=transport) as client:
        _, s_add3, v_add3, _ = collect_focused_github_evidence(
            store,
            client,
            commit_oids=[f01.c1_oid],
            remaining_requests=20,
        )
        assert s_add3 == 0 and v_add3 == 0

    store.close()


def test_principal_binding_api_caps_and_error_handling(tmp_path: Path) -> None:
    """Verify principal mismatch rejection, installation token rejection, and 250-commit REST cap gap."""
    f01 = build_f01_fixture(tmp_path / "f01")
    store = Store.initialize(
        tmp_path / "store",
        InitRequest(repo_path=f01.repo_dir, github_repository="example/transport"),
    )

    pr_capped = {**f01.pr_payload, "commits": 251}
    routes: dict[str, tuple[int, Any, dict[str, str]]] = {
        "/repos/example/transport": (
            200,
            {"id": 9001, "full_name": "example/transport", "private": True},
            {},
        ),
        f"/repos/example/transport/commits/{f01.c1_oid}/pulls": (
            200,
            [{"number": 10, "id": 10010}],
            {},
        ),
        "/repos/example/transport/pulls/10": (200, pr_capped, {}),
        "/repos/example/transport/issues/10/comments?per_page=100": (200, [], {}),
        "/repos/example/transport/pulls/10/reviews?per_page=100": (200, [], {}),
        "/repos/example/transport/pulls/10/comments?per_page=100": (200, [], {}),
        "/repos/example/transport/issues/10/timeline?per_page=100": (200, [], {}),
        "/repos/example/transport/pulls/10/commits?per_page=100": (200, [], {}),
        "/repos/example/transport/pulls/10/files?per_page=100": (200, [], {}),
    }

    # 1. Bind store to user 7001
    t_user1, _ = make_mock_github_transport(routes, expected_user={"id": 7001, "login": "alice"})
    with GitHubClient(token="ghp_user1", transport=t_user1) as client:
        collect_focused_github_evidence(store, client, commit_oids=[f01.c1_oid])

    # Verify API_RESULT_CAP gap was recorded for PR with 251 commits
    with store.connect_read() as conn:
        cap_gaps = conn.execute(
            "SELECT code, scope FROM coverage_gaps WHERE code = 'API_RESULT_CAP' AND active = 1"
        ).fetchall()
        assert len(cap_gaps) == 1
        assert cap_gaps[0]["scope"] == "github:pr:10:commits"

    # 2. Different principal (user 7002) is rejected with REPOSITORY_MISMATCH
    t_user2, _ = make_mock_github_transport(routes, expected_user={"id": 7002, "login": "bob"})
    with GitHubClient(token="ghp_user2", transport=t_user2) as client:
        with pytest.raises(CodeContextError) as exc_info:
            verify_and_bind_github_repository(store, client)
        assert exc_info.value.code == "REPOSITORY_MISMATCH"

    # 3. Unauthenticated call on a principal-bound store fails with AUTH_REQUIRED
    with GitHubClient(token="", transport=t_user1) as client:
        with pytest.raises(CodeContextError) as exc_info:
            verify_and_bind_github_repository(store, client)
        assert exc_info.value.code == "AUTH_REQUIRED"

    # 4. Installation token (where /user returns 404) fails with AUTH_REQUIRED
    t_inst, _ = make_mock_github_transport(routes, expected_user=None)
    with GitHubClient(token="ghs_installation", transport=t_inst) as client:
        with pytest.raises(CodeContextError) as exc_info:
            verify_and_bind_github_repository(store, client)
        assert exc_info.value.code == "AUTH_REQUIRED"

    # 5. GitLab SSH and HTTPS remote URL parsing (rejects cleartext http://)
    assert parse_gitlab_remote_url("git@gitlab.com:org/sub/project.git") == (
        "https://gitlab.com",
        "org/sub/project",
    )
    assert parse_gitlab_remote_url("https://gitlab.example.internal/team/service.git") == (
        "https://gitlab.example.internal",
        "team/service",
    )
    assert parse_gitlab_remote_url("http://gitlab.example.internal/team/service.git") is None
    assert parse_gitlab_remote_url("https://gitlab.com@evil.example/org/proj.git") is None
    assert parse_gitlab_remote_url("https://user:secret@gitlab.com/org/proj.git") is None
    assert parse_gitlab_remote_url("https://notgitlab.com/org/proj.git") is None

    store.close()


def test_gitlab_env_token_stays_on_named_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GITLAB_TOKEN", "glpat-test")
    monkeypatch.delenv("GLAB_TOKEN", raising=False)
    monkeypatch.delenv("GITLAB_HOST", raising=False)
    monkeypatch.setattr("codecontext.gitlab.Path.home", lambda: tmp_path)
    monkeypatch.setattr(
        "codecontext.gitlab.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, "", ""),
    )
    assert resolve_gitlab_token("https://gitlab.com") == "glpat-test"
    assert resolve_gitlab_token("https://gitlab.example.internal") is None
    assert resolve_gitlab_token("https://gitlab.com@evil.example") is None
    monkeypatch.setenv("GITLAB_HOST", "gitlab.example.internal")
    assert resolve_gitlab_token("https://gitlab.example.internal") == "glpat-test"
    with pytest.raises(CodeContextError) as exc_info:
        GitLabClient(base_url="https://oauth2:secret@gitlab.com@evil.example/org/proj.git", token="dummy-token")
    assert exc_info.value.code == "INVALID_REQUEST"
    assert "secret" not in exc_info.value.message
    assert "dummy-token" not in exc_info.value.message


# --- Incremental discovery against a small fake GitHub -----------------------------------------

OLD_TS = "2026-01-01T00:00:00Z"
API = "https://api.github.com"


def _now_iso() -> str:
    from codecontext.identity import us_to_iso_z
    from codecontext.store import now_us

    return us_to_iso_z(now_us())


class FakeGitHub:
    """Serves `owner/repo` from mutable state so tests can change it between syncs."""

    def __init__(self, page_size: int = 100) -> None:
        self.page_size = page_size
        self.issues: dict[int, dict[str, Any]] = {}
        self.hidden: set[int] = set()
        self.issue_comments: list[dict[str, Any]] = []
        self.review_comments: list[dict[str, Any]] = []
        self.reviews: dict[int, list[dict[str, Any]]] = {}
        self.commit_pulls: dict[str, list[int]] = {}
        self.requested: list[str] = []
        self.calls: list[httpx.Request] = []
        # Raw-path substring -> canned response, for pages that come back broken.
        self.bad: dict[str, httpx.Response] = {}
        self._next_id = 1000

    def add_issue(self, number: int, *, pr: bool = False, updated_at: str = OLD_TS) -> None:
        item: dict[str, Any] = {
            "number": number,
            "title": f"item {number}",
            "body": "",
            "state": "open",
            "user": {"id": 1, "login": "a"},
            "created_at": updated_at,
            "updated_at": updated_at,
        }
        if pr:
            item["pull_request"] = {"url": f"{API}/repos/owner/repo/pulls/{number}"}
            item["head"] = {"sha": "h" * 40}
            item["base"] = {"sha": "b" * 40}
        self.issues[number] = item

    def touch(self, number: int) -> None:
        self.issues[number]["updated_at"] = _now_iso()

    def _comment(self, **extra: Any) -> dict[str, Any]:
        self._next_id += 1
        ts = extra.pop("updated_at", _now_iso())
        return {"id": self._next_id, "body": "c", "user": {"id": 2, "login": "b"},
                "created_at": ts, "updated_at": ts, **extra}

    def add_issue_comment(self, number: int, **extra: Any) -> None:
        self.issue_comments.append(
            self._comment(issue_url=f"{API}/repos/owner/repo/issues/{number}", **extra)
        )

    def add_review(self, pr: int) -> None:
        self._next_id += 1
        self.reviews.setdefault(pr, []).append(
            {"id": self._next_id, "body": "lgtm", "state": "APPROVED",
             "user": {"id": 3, "login": "c"}, "submitted_at": _now_iso()}
        )

    def _page(self, request: httpx.Request, items: list[dict[str, Any]]) -> httpx.Response:
        query = dict(urllib.parse.parse_qsl(request.url.query.decode("ascii")))
        page = int(query.get("page", "1"))
        start = (page - 1) * self.page_size
        headers = {}
        if start + self.page_size < len(items):
            query["page"] = str(page + 1)
            nxt = f"{API}{request.url.path}?{urllib.parse.urlencode(query, safe=':')}"
            headers["Link"] = f'<{nxt}>; rel="next"'
        return httpx.Response(200, json=items[start : start + self.page_size], headers=headers)

    @staticmethod
    def _since(request: httpx.Request, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query = dict(urllib.parse.parse_qsl(request.url.query.decode("ascii")))
        if query.get("sort") == "updated":
            items = sorted(items, key=lambda i: i["updated_at"])
        since = query.get("since")
        return [i for i in items if since is None or i["updated_at"] >= since]

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        raw = request.url.raw_path.decode("ascii")
        self.requested.append(raw)
        self.calls.append(request)
        for needle, broken in self.bad.items():
            if needle in raw:
                return broken
        base = "/repos/owner/repo"
        if path == "/user":
            return httpx.Response(200, json={"id": 5, "login": "op"})
        if path == base:
            return httpx.Response(200, json={"id": 77, "full_name": "owner/repo", "private": False})
        if path == f"{base}/issues":
            listed = [i for n, i in sorted(self.issues.items()) if n not in self.hidden]
            return self._page(request, self._since(request, listed))
        if path == f"{base}/issues/comments":
            return self._page(request, self._since(request, self.issue_comments))
        if path == f"{base}/pulls/comments":
            return self._page(request, self._since(request, self.review_comments))
        m = re.fullmatch(rf"{base}/pulls/(\d+)", path)
        if m and int(m.group(1)) in self.issues:
            item = self.issues[int(m.group(1))]
            etag = '"' + hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()[:16] + '"'
            if request.headers.get("If-None-Match") == etag:
                return httpx.Response(304, headers={"ETag": etag})
            return httpx.Response(200, json=item, headers={"ETag": etag})
        m = re.fullmatch(rf"{base}/issues/(\d+)/comments", path)
        if m:
            url = f"{API}{base}/issues/{m.group(1)}"
            return self._page(request, [c for c in self.issue_comments if c["issue_url"] == url])
        m = re.fullmatch(rf"{base}/pulls/(\d+)/reviews", path)
        if m:
            return self._page(request, self.reviews.get(int(m.group(1)), []))
        m = re.fullmatch(rf"{base}/pulls/(\d+)/comments", path)
        if m:
            url = f"{API}{base}/pulls/{m.group(1)}"
            return self._page(request, [c for c in self.review_comments if c["pull_request_url"] == url])
        if re.fullmatch(rf"{base}/(issues/\d+/timeline|pulls/\d+/(commits|files))", path):
            return httpx.Response(200, json=[])
        m = re.fullmatch(rf"{base}/commits/([0-9a-f]+)/pulls", path)
        if m and m.group(1) in self.commit_pulls:
            return httpx.Response(200, json=[{"number": n} for n in self.commit_pulls[m.group(1)]])
        return httpx.Response(404, json={"message": "Not Found"})


def _fake_store(tmp_path: Path) -> Store:
    repo = init_fixture_repo(tmp_path / "repo")
    commit_fixture_spec(repo, GitCommitSpec(message="init", files={"a.py": b"x = 1\n"}))
    return Store.initialize(
        tmp_path / "store", InitRequest(repo_path=repo, github_repository="owner/repo")
    )


def _sync(store: Store, fake: FakeGitHub, **kwargs: Any) -> tuple[int, int, int, bool]:
    with GitHubClient(token="", transport=httpx.MockTransport(fake.handler)) as client:
        return collect_full_and_incremental_github(store, client, **kwargs)


def _sid(kind: str, number: int | str) -> str:
    return make_src_id(provider="github", repository_provider_id="77", kind=kind, provider_id=str(number))


def test_new_issue_found_after_backfill(tmp_path: Path) -> None:
    store = _fake_store(tmp_path)
    fake = FakeGitHub()
    fake.add_issue(1)
    assert _sync(store, fake)[3] is True
    fake.add_issue(42, updated_at=_now_iso())
    assert _sync(store, fake)[3] is True
    with store.connect_read() as conn:
        found = conn.execute("SELECT 1 FROM sources WHERE id = ?", (_sid("issue", 42),)).fetchone()
        ckpt = conn.execute(
            "SELECT cursor_json, high_watermark_us FROM checkpoints WHERE stream_key = ?",
            ("github:77:issues_refresh",),
        ).fetchone()
        all_wm = conn.execute(
            "SELECT high_watermark_us FROM checkpoints WHERE stream_key = ?",
            ("github:77:issues_all",),
        ).fetchone()["high_watermark_us"]
    store.close()
    assert found is not None
    assert json.loads(ckpt["cursor_json"])["resume_since"] is None
    assert ckpt["high_watermark_us"] is not None
    refresh = [p for p in fake.requested if "sort=updated" in p and p.startswith("/repos/owner/repo/issues?")]
    expected_since = us_to_iso_z(all_wm - 300 * 1_000_000)
    assert refresh and all(p.endswith(f"&since={expected_since}") for p in refresh)


def test_changed_pr_reschedules_child_streams(tmp_path: Path) -> None:
    store = _fake_store(tmp_path)
    fake = FakeGitHub()
    fake.add_issue(7, pr=True)
    fake.add_issue(8, pr=True)
    _sync(store, fake)
    fake.add_review(pr=7)
    fake.add_review(pr=8)
    fake.touch(7)
    assert _sync(store, fake)[3] is True
    with store.connect_read() as conn:
        counts = {
            pr: int(conn.execute(
                "SELECT COUNT(*) FROM sources WHERE kind = 'review' AND parent_source_id = ?",
                (_sid("pull_request", pr),),
            ).fetchone()[0])
            for pr in (7, 8)
        }
    store.close()
    # PR 8 did not change, so its finished child streams stay finished.
    assert counts == {7: 1, 8: 0}


def test_comment_stream_resumes_mid_pages(tmp_path: Path) -> None:
    store = _fake_store(tmp_path)
    fake = FakeGitHub(page_size=2)
    for i in range(5):
        fake.add_issue_comment(3, updated_at=f"2026-01-01T00:00:0{i}Z")
    # repo + issues backfill page + issues refresh page + two comment pages
    assert _sync(store, fake, remaining_requests=5)[3] is False
    assert fake.requested[-1].endswith("page=2")
    fake.requested.clear()
    assert _sync(store, fake)[3] is True
    comment_urls = [p for p in fake.requested if p.startswith("/repos/owner/repo/issues/comments")]
    # Restarts at the fourth comment's timestamp, so only one page is read again.
    assert comment_urls == [
        "/repos/owner/repo/issues/comments?sort=updated&direction=asc&per_page=100"
        "&since=2026-01-01T00:00:03Z"
    ]
    with store.connect_read() as conn:
        n = int(conn.execute("SELECT COUNT(*) FROM sources WHERE kind = 'issue_comment'").fetchone()[0])
    store.close()
    assert n == 5


def test_comment_stream_sets_parent_of_stored_pr(tmp_path: Path) -> None:
    store = _fake_store(tmp_path)
    fake = FakeGitHub()
    fake.add_issue(7, pr=True)
    _sync(store, fake)
    # PR 7 is unchanged, so only the repository-wide comment stream sees the new comment.
    fake.add_issue_comment(7)
    _sync(store, fake)
    with store.connect_read() as conn:
        parent = conn.execute(
            "SELECT parent_source_id FROM sources WHERE id = ?",
            (_sid("issue_comment", fake.issue_comments[0]["id"]),),
        ).fetchone()["parent_source_id"]
    store.close()
    assert parent == _sid("pull_request", 7)


def test_comment_before_pr_keeps_parent_and_survives_reconcile(tmp_path: Path) -> None:
    store = _fake_store(tmp_path)
    fake = FakeGitHub()
    fake.add_issue(7, pr=True)
    fake.hidden.add(7)
    fake.add_issue_comment(7)
    comment_sid = _sid("issue_comment", fake.issue_comments[0]["id"])
    _sync(store, fake)  # comment stored before its PR
    fake.hidden.clear()
    assert _sync(store, fake, reconcile=True)[3] is True
    with store.connect_read() as conn:
        row = conn.execute(
            "SELECT availability, parent_source_id FROM sources WHERE id = ?", (comment_sid,)
        ).fetchone()
    store.close()
    assert row["availability"] == "available"
    assert row["parent_source_id"] == _sid("pull_request", 7)


def test_upsert_without_parent_keeps_existing_parent(tmp_path: Path) -> None:
    store = _fake_store(tmp_path)
    fake = FakeGitHub()
    fake.add_issue(7, pr=True)
    fake.add_issue_comment(7)
    _sync(store, fake)
    comment = fake.issue_comments[0]
    with store.write_transaction(bump_generation=False) as conn:
        upsert_github_source_and_version(
            conn, repository_id=store.config.repository_id, repository_provider_id="77",
            kind="issue_comment", provider_id=str(comment["id"]), parent_source_id=None,
            title=None, body="edited", author_dict=None, source_url=None, authored_at_us=None,
            updated_at_us=None, anchors={}, lifecycle_fields={}, raw_payload=comment,
        )
        parent = conn.execute(
            "SELECT parent_source_id FROM sources WHERE id = ?",
            (_sid("issue_comment", comment["id"]),),
        ).fetchone()["parent_source_id"]
    store.close()
    assert parent == _sid("pull_request", 7)


def test_commit_pr_lookup_retried_after_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import codecontext.github as gh

    store = _fake_store(tmp_path)
    fake = FakeGitHub()
    fake.add_issue(9, pr=True)
    oid = "c" * 40
    lookup = f"/repos/owner/repo/commits/{oid}/pulls"

    def focus() -> None:
        with GitHubClient(token="", transport=httpx.MockTransport(fake.handler)) as client:
            collect_focused_github_evidence(store, client, commit_oids=[oid])

    focus()  # 404
    fake.commit_pulls[oid] = [9]
    focus()  # still inside the re-check delay: no new lookup
    assert fake.requested.count(lookup) == 1
    real_now = gh.now_us
    monkeypatch.setattr(gh, "now_us", lambda: real_now() + gh.COMMIT_PULLS_RECHECK_US + 1)
    focus()
    assert fake.requested.count(lookup) == 2
    focus()  # done: never asked again
    assert fake.requested.count(lookup) == 2
    with store.connect_read() as conn:
        rel = conn.execute(
            "SELECT 1 FROM relations WHERE from_id = ? AND relation_kind = 'associated_pr' AND to_id = ?",
            (oid, _sid("pull_request", 9)),
        ).fetchone()
    store.close()
    assert rel is not None


def test_paused_refresh_does_not_skip_item_that_moved(tmp_path: Path) -> None:
    from codecontext.store import now_us

    store = _fake_store(tmp_path)
    fake = FakeGitHub(page_size=2)
    for n in range(1, 5):
        fake.add_issue(n)
    assert _sync(store, fake)[3] is True
    base_us = now_us() - 100 * 1_000_000

    def edit(n: int, seconds: int) -> None:
        fake.issues[n]["updated_at"] = us_to_iso_z(base_us - base_us % 1_000_000 + seconds * 1_000_000)
        fake.issues[n]["title"] = f"edited {n} at {seconds}"

    for n in range(1, 5):
        edit(n, n)
    # repo + refresh page 1 + comments and timeline for its two issues
    assert _sync(store, fake, remaining_requests=6)[3] is False
    edit(1, 10)  # moves issue 1 to the end of the list, shifting the pages
    assert _sync(store, fake)[3] is True
    with store.connect_read() as conn:
        titles = {
            n: conn.execute(
                "SELECT v.title FROM sources s JOIN source_versions v ON v.id = s.current_version_id"
                " WHERE s.id = ?",
                (_sid("issue", n),),
            ).fetchone()["title"]
            for n in range(1, 5)
        }
    store.close()
    assert titles == {n: fake.issues[n]["title"] for n in range(1, 5)}


_SWEEPS = {
    "backfill": ("issues?state=all&sort=created", "github:77:issues_all"),
    "refresh": ("issues?state=all&sort=updated", "github:77:issues_refresh"),
    "comments": ("/issues/comments?", "github:77:issue_comments_inc"),
}


@pytest.mark.parametrize("sweep", sorted(_SWEEPS))
@pytest.mark.parametrize("broken", ["too_large", "not_a_list"])
def test_broken_listing_page_records_gap_and_keeps_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sweep: str, broken: str
) -> None:
    import codecontext.github as gh

    monkeypatch.setattr(gh, "MAX_HTTP_BODY_BYTES", 64 * 1024)
    needle, stream_key = _SWEEPS[sweep]
    store = _fake_store(tmp_path)
    fake = FakeGitHub()
    fake.add_issue(1)
    fake.add_issue_comment(1, updated_at=OLD_TS)
    if sweep == "refresh":
        assert _sync(store, fake)[3] is True
        fake.add_issue(2, updated_at=_now_iso())

    def cursor() -> Any:
        with store.connect_read() as conn:
            row = conn.execute(
                "SELECT cursor_json FROM checkpoints WHERE stream_key = ?", (stream_key,)
            ).fetchone()
        return None if row is None else row["cursor_json"]

    before = cursor()
    fake.bad[needle] = (
        httpx.Response(200, content=b"[" + b" " * (100 * 1024) + b"]")
        if broken == "too_large"
        else httpx.Response(200, json={"message": "not a page"})
    )
    assert _sync(store, fake)[3] is False
    code = "HTTP_BODY_TOO_LARGE" if broken == "too_large" else "SOURCE_UNAVAILABLE"
    with store.connect_read() as conn:
        active = conn.execute(
            "SELECT 1 FROM coverage_gaps WHERE code = ? AND scope = ? AND active = 1",
            (code, stream_key),
        ).fetchone()
    assert active is not None
    assert cursor() == before

    fake.bad.clear()
    assert _sync(store, fake)[3] is True
    with store.connect_read() as conn:
        still_active = conn.execute(
            "SELECT 1 FROM coverage_gaps WHERE scope = ? AND active = 1", (stream_key,)
        ).fetchone()
    store.close()
    assert still_active is None


def test_reconcile_marks_plain_issue_comments_seen(tmp_path: Path) -> None:
    store = _fake_store(tmp_path)
    fake = FakeGitHub()
    fake.add_issue(3)
    fake.add_issue_comment(3, updated_at=OLD_TS)
    comment_sid = _sid("issue_comment", fake.issue_comments[0]["id"])
    _sync(store, fake)
    assert _sync(store, fake, reconcile=True)[3] is True
    with store.connect_read() as conn:
        row = conn.execute(
            "SELECT availability, parent_source_id FROM sources WHERE id = ?", (comment_sid,)
        ).fetchone()
    store.close()
    assert row["availability"] == "available"
    assert row["parent_source_id"] == _sid("issue", 3)


def test_secondary_rate_limit_is_retryable() -> None:
    body = b'{"message": "You have exceeded a secondary rate limit. Please wait."}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, content=body)

    with GitHubClient(token="", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(CodeContextError) as exc_info:
            client.get_json("/repos/owner/repo")
    err = exc_info.value
    assert err.code == "RATE_LIMITED" and err.retryable
    assert err.details["wait_seconds"] >= 60
    assert "secondary" not in err.message and "secondary" not in json.dumps(err.details)


def test_plain_403_is_still_access_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "Resource not accessible by integration"})

    with GitHubClient(token="", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(CodeContextError) as exc_info:
            client.get_json("/repos/owner/repo")
    assert exc_info.value.code == "ACCESS_BLOCKED"


@pytest.mark.parametrize(
    "graphql_response",
    [
        httpx.Response(403, json={"message": "You have exceeded a secondary rate limit."}),
        httpx.Response(200, json={"errors": [{"type": "RATE_LIMITED", "message": "limit"}]}),
    ],
)
def test_graphql_rate_limit_stops_without_rest_fallback(
    tmp_path: Path, graphql_response: httpx.Response
) -> None:
    store = _fake_store(tmp_path)
    fake = FakeGitHub()
    fake.add_issue(1)
    fake.bad["/graphql"] = graphql_response
    with GitHubClient(token="", transport=httpx.MockTransport(fake.handler)) as client:
        with pytest.raises(CodeContextError) as exc_info:
            collect_full_and_incremental_github(store, client, use_graphql=True)
    store.close()
    assert exc_info.value.code == "RATE_LIMITED" and exc_info.value.retryable
    gql_at = fake.requested.index("/graphql")
    assert fake.requested[gql_at + 1 :] == []


def test_pr_detail_sends_etag_and_skips_on_304(tmp_path: Path) -> None:
    store = _fake_store(tmp_path)
    fake = FakeGitHub()
    fake.add_issue(9, pr=True)
    oid = "c" * 40
    fake.commit_pulls[oid] = [9]

    def focus() -> None:
        with GitHubClient(token="", transport=httpx.MockTransport(fake.handler)) as client:
            assert collect_focused_github_evidence(store, client, commit_oids=[oid])[3] is True

    focus()
    fake.calls.clear()
    focus()
    pr_calls = [r for r in fake.calls if r.url.path == "/repos/owner/repo/pulls/9"]
    with store.connect_read() as conn:
        versions = int(conn.execute(
            "SELECT COUNT(*) FROM source_versions WHERE source_id = ?", (_sid("pull_request", 9),)
        ).fetchone()[0])
        rel = conn.execute(
            "SELECT 1 FROM relations WHERE from_id = ? AND relation_kind = 'associated_pr' AND to_id = ?",
            (oid, _sid("pull_request", 9)),
        ).fetchone()
    store.close()
    assert len(pr_calls) == 1 and "If-None-Match" in pr_calls[0].headers
    assert versions == 1
    assert rel is not None
    # A 304 leaves the finished child streams alone.
    assert [r.url.path for r in fake.calls if "/pulls/9/" in r.url.path] == []


def test_repository_verified_once_per_sync(tmp_path: Path) -> None:
    store = _fake_store(tmp_path)
    fake = FakeGitHub()
    fake.add_issue(9, pr=True)
    oid = "c" * 40
    fake.commit_pulls[oid] = [9]
    with GitHubClient(token="ghp_test", transport=httpx.MockTransport(fake.handler)) as client:
        collect_focused_github_evidence(store, client, commit_oids=[oid])
        collect_full_and_incremental_github(store, client)
    store.close()
    assert fake.requested.count("/user") == 1
    assert fake.requested.count("/repos/owner/repo") == 1
