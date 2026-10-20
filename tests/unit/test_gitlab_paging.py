import httpx

from codecontext.gitlab import GitLabClient, collect_gitlab_merge_requests_and_issues
from codecontext.models import InitRequest
from codecontext.store import Store
from tests.fixtures.builders import GitCommitSpec, commit_fixture_spec, init_fixture_repo


def test_gitlab_discussion_pages_resume(tmp_path) -> None:
    repo = tmp_path / "repo"
    init_fixture_repo(repo)
    commit_fixture_spec(repo, GitCommitSpec(message="init", files={"src/a.py": b"def hello():\n    return 1\n"}))
    store = Store.initialize(tmp_path / "store", InitRequest(repo_path=repo))
    calls: list[str] = []
    title = "mr"

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode("ascii")
        calls.append(path)
        if path == "/api/v4/projects/group%2Fapp":
            return httpx.Response(200, json={"id": 9, "visibility": "public"})
        if "/merge_requests?" in path and "/discussions" not in path:
            return httpx.Response(200, json=[{"iid": 4, "title": title, "sha": "abc"}])
        if "/discussions?" in path and "page=2" in path:
            return httpx.Response(
                200,
                json=[{"id": "d2", "notes": [{"id": 8, "system": False, "body": "later"}]}],
            )
        if "/discussions?" in path:
            return httpx.Response(
                200,
                json=[{"id": "d1", "notes": [{"id": 7, "system": False, "body": "first"}]}],
                headers={"X-Next-Page": "2"},
            )
        if "/issues?" in path:
            return httpx.Response(200, json=[])
        return httpx.Response(404, json={"message": "missing"})

    transport = httpx.MockTransport(handler)
    with GitLabClient(base_url="https://gitlab.com", token="", transport=transport) as client:
        used, _sources, _versions, done = collect_gitlab_merge_requests_and_issues(
            store,
            client,
            project_path="group/app",
            remaining_requests=3,
        )
    assert used == 3
    assert done is False
    assert not any("page=2" in path for path in calls)

    with GitLabClient(base_url="https://gitlab.com", token="", transport=transport) as client:
        _used, _sources, _versions, done = collect_gitlab_merge_requests_and_issues(
            store,
            client,
            project_path="group/app",
            remaining_requests=10,
        )
    assert done is True
    assert any("/discussions?per_page=100&page=2" in path for path in calls)
    with store.connect_read() as conn:
        note = conn.execute(
            "SELECT 1 FROM sources WHERE provider_id = '8'"
        ).fetchone()
    assert note is not None
    title = "updated mr"
    calls.clear()
    with GitLabClient(base_url="https://gitlab.com", token="", transport=transport) as client:
        _used, _sources, versions, done = collect_gitlab_merge_requests_and_issues(
            store, client, project_path="group/app", remaining_requests=10,
        )
    assert done is True
    assert versions == 1
    assert any("/merge_requests?" in path for path in calls)
    with store.connect_read() as conn:
        current = conn.execute(
            "SELECT v.title FROM sources s JOIN source_versions v ON v.id = s.current_version_id "
            "WHERE s.kind = 'pull_request' AND s.provider_id = '4'"
        ).fetchone()
    assert current["title"] == title
    store.close()
