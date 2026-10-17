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
    verify_and_bind_github_repository,
)
from codecontext.gitlab import parse_gitlab_remote_url
from codecontext.models import InitRequest
from codecontext.store import Store
from tests.fixtures.builders import build_f01_fixture


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

    # 5. GitLab SSH and HTTPS remote URL parsing
    assert parse_gitlab_remote_url("git@gitlab.com:org/sub/project.git") == (
        "https://gitlab.com",
        "org/sub/project",
    )
    assert parse_gitlab_remote_url("https://gitlab.example.internal/team/service.git") == (
        "https://gitlab.example.internal",
        "team/service",
    )

    store.close()
