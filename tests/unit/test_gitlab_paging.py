import httpx
import pytest

from codecontext.deadline import sync_deadline
from codecontext.errors import CodeContextError
from codecontext.gitlab import (
    GITLAB_MAX_RETRY_AFTER_SECONDS,
    GitLabClient,
    collect_gitlab_merge_requests_and_issues,
)
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


def _gitlab_store(tmp_path):
    repo = tmp_path / "repo"
    init_fixture_repo(repo)
    commit_fixture_spec(repo, GitCommitSpec(message="init", files={"src/a.py": b"x = 1\n"}))
    return Store.initialize(tmp_path / "store", InitRequest(repo_path=repo))


def test_late_visible_item_inside_overlap_is_ingested(tmp_path) -> None:
    store = _gitlab_store(tmp_path)
    visible = [{"iid": 1, "title": "one", "updated_at": "2026-01-01T10:00:00Z"}]
    late = {"iid": 2, "title": "two", "updated_at": "2026-01-01T09:59:59Z"}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode("ascii")
        if path == "/api/v4/projects/group%2Fapp":
            return httpx.Response(200, json={"id": 9, "visibility": "public"})
        if "/merge_requests?" in path and "/discussions" not in path:
            since = request.url.params.get("updated_after")
            return httpx.Response(
                200, json=[m for m in visible if since is None or m["updated_at"] >= since[:19] + "Z"]
            )
        if "/issues?" in path:
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)
    for _ in range(2):
        with GitLabClient(base_url="https://gitlab.com", token="", transport=transport) as client:
            assert collect_gitlab_merge_requests_and_issues(
                store, client, project_path="group/app", remaining_requests=20
            )[3]
        if len(visible) == 1:
            visible.append(late)
    with store.connect_read() as conn:
        row = conn.execute("SELECT 1 FROM sources WHERE kind = 'pull_request' AND provider_id = '2'").fetchone()
    assert row is not None
    store.close()


def _client(handler, sleeps=None) -> GitLabClient:
    return GitLabClient(
        base_url="https://gitlab.com",
        token="",
        transport=httpx.MockTransport(handler),
        sleep_fn=(sleeps.append if sleeps is not None else lambda _s: None),
    )


def test_invalid_json_is_code_context_error() -> None:
    with _client(lambda _r: httpx.Response(200, content=b"<html>secret-body")) as client:
        with pytest.raises(CodeContextError) as err:
            client.get_json("/projects/1")
    assert err.value.code == "IO_ERROR"
    assert "secret-body" not in err.value.message


def test_429_honours_retry_after() -> None:
    sleeps: list[float] = []
    responses = iter([httpx.Response(429, headers={"Retry-After": "2"}), httpx.Response(200, json=[])])
    with _client(lambda _r: next(responses), sleeps) as client:
        status, data, _next = client.get_json("/projects/1")
    assert (status, data) == (200, [])
    assert sleeps == [2]


def test_429_without_retry_after_raises_rate_limited() -> None:
    with _client(lambda _r: httpx.Response(429)) as client:
        with pytest.raises(CodeContextError) as err:
            client.get_json("/projects/1")
    assert err.value.code == "RATE_LIMITED"


def test_transport_error_retried_once() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json={"ok": True})

    with _client(handler) as client:
        assert client.get_json("/projects/1")[1] == {"ok": True}
    assert len(calls) == 2

    calls.clear()
    with _client(lambda r: (_ for _ in ()).throw(httpx.ConnectError("boom", request=r))) as client:
        with pytest.raises(CodeContextError) as err:
            client.get_json("/projects/1")
    assert err.value.code == "IO_ERROR"


def test_transport_retry_counts_against_budget() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with _client(handler) as client:
        with pytest.raises(CodeContextError):
            client.get_json("/projects/1")
        assert client.requests_attempted == 2
    with _client(handler) as client:
        with pytest.raises(CodeContextError):
            client.get_json("/projects/1", remaining_requests=1)
        assert client.requests_attempted == 1


def test_second_429_after_retry_is_rate_limited() -> None:
    sleeps: list[float] = []
    with _client(lambda _r: httpx.Response(429, headers={"Retry-After": "1"}), sleeps) as client:
        with pytest.raises(CodeContextError) as err:
            client.get_json("/projects/1")
    assert err.value.code == "RATE_LIMITED"
    assert sleeps == [1]
    assert client.requests_attempted == 2


def test_429_without_budget_for_retry_does_not_sleep() -> None:
    sleeps: list[float] = []
    with _client(lambda _r: httpx.Response(429, headers={"Retry-After": "1"}), sleeps) as client:
        with pytest.raises(CodeContextError) as err:
            client.get_json("/projects/1", remaining_requests=1)
    assert err.value.code == "RATE_LIMITED"
    assert sleeps == []


def test_retry_after_non_ascii_digit_is_rate_limited() -> None:
    sleeps: list[float] = []
    with _client(lambda _r: httpx.Response(429, headers={"Retry-After": "\u00b2".encode()}), sleeps) as client:
        with pytest.raises(CodeContextError) as err:
            client.get_json("/projects/1")
    assert err.value.code == "RATE_LIMITED"
    assert sleeps == []


@pytest.mark.parametrize("value", ["9" * 400, "86400"])
def test_retry_after_is_clamped(value: str) -> None:
    sleeps: list[float] = []
    responses = iter([httpx.Response(429, headers={"Retry-After": value}), httpx.Response(200, json=[])])
    with _client(lambda _r: next(responses), sleeps) as client:
        client.get_json("/projects/1")
    assert sleeps == [GITLAB_MAX_RETRY_AFTER_SECONDS]


def test_retry_after_is_capped_by_remaining_deadline() -> None:
    sleeps: list[float] = []
    responses = iter([httpx.Response(429, headers={"Retry-After": "50"}), httpx.Response(200, json=[])])
    with sync_deadline(5), _client(lambda _r: next(responses), sleeps) as client:
        client.get_json("/projects/1")
    assert 0 < sleeps[0] <= 5
