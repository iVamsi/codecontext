import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from codecontext.identity import (
    canonical_json,
    datetime_to_us,
    default_token_estimator,
    path_bytes_to_display,
    sha256_bytes,
)
from codecontext.models import (
    AwareDatetime,
    ContextRequest,
    GetRequest,
    SearchRequest,
)
from codecontext.retrieval import execute_context, execute_get, execute_search
from codecontext.store import Store


class EvalStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ToolSpec(EvalStrictModel):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    description: Annotated[str, Field(min_length=1, max_length=2000)]
    input_schema: dict[str, Any]


class EvalRequest(EvalStrictModel):
    run_id: Annotated[str, Field(min_length=1, max_length=100)]
    question_id: Annotated[str, Field(min_length=1, max_length=100)]
    family_id: Annotated[str, Field(min_length=1, max_length=100)]
    arm: Literal["ordinary", "lexical", "memory"]
    question: Annotated[str, Field(min_length=1, max_length=4000)]
    target_revision: Annotated[str, Field(min_length=40, max_length=64)]
    knowledge_cutoff: AwareDatetime | None = None
    evidence_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    model_name: Annotated[str, Field(min_length=1, max_length=200)]
    model_revision: Annotated[str, Field(min_length=1, max_length=200)]
    model_options: dict[str, str | int | bool]
    system_prompt: Annotated[str, Field(min_length=1, max_length=8000)]
    max_tool_calls: Annotated[int, Field(ge=1, le=100)] = 20
    max_tool_result_tokens: Annotated[int, Field(ge=1, le=1_000_000)] = 30000
    max_answer_tokens: Annotated[int, Field(ge=1, le=100_000)] = 2000
    timeout_seconds: Annotated[int, Field(ge=1, le=3600)] = 300
    tokenizer_id: Annotated[str, Field(min_length=1, max_length=200)] = "utf8_bytes_div3_v1"
    repetition: Annotated[int, Field(ge=1)] = 1
    seed: int | None = None


class ToolEvent(EvalStrictModel):
    ordinal: Annotated[int, Field(ge=1)]
    name: Annotated[str, Field(min_length=1, max_length=100)]
    arguments: dict[str, Any]
    result: dict[str, Any] | None = None
    error_code: str | None = None
    started_at: AwareDatetime
    duration_ms: Annotated[int, Field(ge=0)]
    result_tokens: Annotated[int, Field(ge=0)]
    result_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None


class EvalResult(EvalStrictModel):
    run_id: Annotated[str, Field(min_length=1, max_length=100)]
    state: Literal["complete", "timeout", "budget_exhausted", "runner_error", "protocol_invalid"]
    final_answer: str | None = None
    events: list[ToolEvent]
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    cost_usd: Decimal | None = None
    provider_request_ids: list[str]
    started_at: AwareDatetime
    finished_at: AwareDatetime
    error_code: str | None = None
    runner_fingerprint: Annotated[str, Field(min_length=1, max_length=200)]


class EvalTools:
    """Dispatcher owning tool list, cutoff eligibility, shared tokenizer accounting, timestamps,
    events, and budget checks (`max_tool_calls=20`, `max_tool_result_tokens=30000`, `timeout_seconds=300`).
    """

    def __init__(
        self,
        store: Store,
        request: EvalRequest,
        *,
        token_counter: Callable[[str], int] = default_token_estimator,
    ) -> None:
        from codecontext.git import GitMirror

        self._store = store
        self._mirror = GitMirror(store)
        self._request = request
        self._token_counter = token_counter
        self._deadline_mono = time.monotonic() + float(request.timeout_seconds)
        self._events: list[ToolEvent] = []
        self._total_result_tokens = 0
        self._cutoff_us = (
            datetime_to_us(request.knowledge_cutoff)
            if request.knowledge_cutoff is not None
            else None
        )
        self.budget_refused = False
        self.timeout_refused = False

    @property
    def events(self) -> list[ToolEvent]:
        return list(self._events)

    def list_tools(self) -> list[ToolSpec]:
        baseline = [
            ToolSpec(
                name="list_files",
                description="List tracked files in the frozen snapshot.",
                input_schema={"type": "object", "properties": {"prefix": {"type": "string"}}},
            ),
            ToolSpec(
                name="read_file",
                description="Read a tracked file from the frozen snapshot.",
                input_schema={"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
            ),
            ToolSpec(
                name="search_text",
                description="Search file and discussion text in the frozen corpus.",
                input_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}},
            ),
            ToolSpec(
                name="git_log",
                description="List reachable commits in the frozen snapshot.",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}},
            ),
            ToolSpec(
                name="git_show",
                description="Show commit message and changes for a commit OID.",
                input_schema={"type": "object", "required": ["oid"], "properties": {"oid": {"type": "string"}}},
            ),
            ToolSpec(
                name="list_discussions",
                description="List eligible issues and PRs in the frozen corpus.",
                input_schema={"type": "object", "properties": {"limit": {"type": "integer"}}},
            ),
            ToolSpec(
                name="read_discussion",
                description="Read an eligible discussion source and its comments.",
                input_schema={"type": "object", "required": ["source_id"], "properties": {"source_id": {"type": "string"}}},
            ),
        ]
        if self._request.arm in ("lexical", "memory"):
            baseline.extend(
                [
                    ToolSpec(
                        name="search",
                        description="CodeContext search tool.",
                        input_schema=SearchRequest.model_json_schema(),
                    ),
                    ToolSpec(
                        name="get",
                        description="CodeContext get tool.",
                        input_schema=GetRequest.model_json_schema(),
                    ),
                    ToolSpec(
                        name="context",
                        description="CodeContext context tool.",
                        input_schema=ContextRequest.model_json_schema(),
                    ),
                ]
            )
        return baseline

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        started_dt = datetime.now(UTC)
        start_m = time.monotonic()
        ordinal = len(self._events) + 1

        # 1. Check deadline before dispatch
        if start_m >= self._deadline_mono:
            self.timeout_refused = True
            err_res = {"error": {"code": "BUDGET_EXHAUSTED", "reason": "timeout_exceeded"}}
            self._record_event(ordinal, name, arguments, err_res, "BUDGET_EXHAUSTED", started_dt, start_m)
            return err_res

        # 2. Check max_tool_calls (refuse 21st call)
        if ordinal > self._request.max_tool_calls:
            self.budget_refused = True
            err_res = {"error": {"code": "BUDGET_EXHAUSTED", "reason": "max_tool_calls_exceeded"}}
            self._record_event(ordinal, name, arguments, err_res, "BUDGET_EXHAUSTED", started_dt, start_m)
            return err_res

        allowed_names = {t.name for t in self.list_tools()}
        if name not in allowed_names:
            err_res = {"error": {"code": "INVALID_REQUEST", "reason": f"Tool {name!r} is not available in arm {self._request.arm!r}"}}
            self._record_event(ordinal, name, arguments, err_res, "INVALID_REQUEST", started_dt, start_m)
            return err_res

        try:
            raw_res = self._dispatch_impl(name, arguments)
            error_code: str | None = None
        except Exception as exc:
            error_code = getattr(exc, "code", "IO_ERROR")
            raw_res = {"error": {"code": error_code, "message": str(exc)[:500]}}

        res_json = canonical_json(raw_res)
        res_tokens = int(self._token_counter(res_json))

        # 3. Check aggregate max_tool_result_tokens
        if (self._total_result_tokens + res_tokens) > self._request.max_tool_result_tokens:
            self.budget_refused = True
            err_res = {
                "error": {
                    "code": "BUDGET_EXHAUSTED",
                    "reason": "max_tool_result_tokens_exceeded",
                    "remaining_tokens": max(0, self._request.max_tool_result_tokens - self._total_result_tokens),
                    "result_tokens": res_tokens,
                }
            }
            self._record_event(ordinal, name, arguments, err_res, "BUDGET_EXHAUSTED", started_dt, start_m)
            return err_res

        self._total_result_tokens += res_tokens
        self._record_event(ordinal, name, arguments, raw_res, error_code, started_dt, start_m, res_tokens, res_json)
        return raw_res

    def _record_event(
        self,
        ordinal: int,
        name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        error_code: str | None,
        started_dt: datetime,
        start_m: float,
        res_tokens: int | None = None,
        res_json: str | None = None,
    ) -> None:
        dur_ms = max(0, int((time.monotonic() - start_m) * 1000))
        s_json = res_json if res_json is not None else canonical_json(result)
        toks = res_tokens if res_tokens is not None else int(self._token_counter(s_json))
        sha = sha256_bytes(s_json.encode("utf-8"))
        self._events.append(
            ToolEvent(
                ordinal=ordinal,
                name=name,
                arguments=arguments,
                result=result,
                error_code=error_code,
                started_at=started_dt,
                duration_ms=dur_ms,
                result_tokens=toks,
                result_sha256=sha,
            )
        )

    def _dispatch_impl(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        exclude_enriched = self._request.arm == "lexical"

        if name == "search":
            req_d = dict(arguments.get("request", arguments))
            req_d["knowledge_cutoff"] = self._request.knowledge_cutoff
            s_req = SearchRequest.model_validate(req_d)
            return execute_search(
                self._store,
                self._mirror,
                s_req,
                token_counter=self._token_counter,
                exclude_enriched=exclude_enriched,
            ).model_dump(mode="json")

        if name == "get":
            req_d = dict(arguments.get("request", arguments))
            req_d["knowledge_cutoff"] = self._request.knowledge_cutoff
            g_req = GetRequest.model_validate(req_d)
            return execute_get(
                self._store,
                self._mirror,
                g_req,
                token_counter=self._token_counter,
            ).model_dump(mode="json")

        if name == "context":
            req_d = dict(arguments.get("request", arguments))
            req_d["knowledge_cutoff"] = self._request.knowledge_cutoff
            c_req = ContextRequest.model_validate(req_d)
            return execute_context(
                self._store,
                self._mirror,
                c_req,
                token_counter=self._token_counter,
                exclude_enriched=exclude_enriched,
            ).model_dump(mode="json")

        # Baseline tools over frozen bytes and owned Git objects
        with self._store.connect_read() as conn:
            if name == "list_files":
                prefix = str(arguments.get("prefix") or "")
                rows = conn.execute("SELECT path, blob_oid FROM snapshot_files ORDER BY path ASC LIMIT 200").fetchall()
                files = []
                for r in rows:
                    disp, _ = path_bytes_to_display(bytes(r["path"]))
                    if not prefix or disp.startswith(prefix):
                        files.append({"blob_oid": str(r["blob_oid"]), "path": disp})
                return {"files": files}

            if name == "read_file":
                target_p = str(arguments.get("path") or "").encode("utf-8")
                sf = conn.execute("SELECT blob_oid FROM snapshot_files WHERE path = ?", (target_p,)).fetchone()
                if not sf:
                    return {"error": "file_not_found"}
                with self._mirror.open_batch_session() as batch:
                    obj = batch.read_object(str(sf["blob_oid"]))
                if obj is None:
                    return {"error": "missing_blob"}
                return {
                    "blob_oid": str(sf["blob_oid"]),
                    "content": obj[2][:16384].decode("utf-8", errors="replace"),
                }

            if name == "search_text":
                q = str(arguments.get("query") or "")
                s_res = execute_search(
                    self._store,
                    self._mirror,
                    SearchRequest(
                        query=q,
                        kind="all",
                        limit=10,
                        knowledge_cutoff=self._request.knowledge_cutoff,
                    ),
                    token_counter=self._token_counter,
                    exclude_enriched=True,
                )
                return {
                    "hits": [
                        {"id": h.id, "kind": h.kind, "snippet": h.snippet, "title": h.title}
                        for h in s_res.hits
                    ]
                }

            if name == "git_log":
                lim = min(50, max(1, int(arguments.get("limit", 10))))
                rows = conn.execute(
                    """
                    SELECT c.oid, c.author_name, c.committed_at_us, c.message
                    FROM commits c
                    JOIN snapshot_commits sc ON sc.commit_oid = c.oid
                    ORDER BY COALESCE(c.committed_at_us, c.authored_at_us, 0) DESC, c.oid ASC
                    LIMIT ?
                    """,
                    (lim,),
                ).fetchall()
                commits = []
                for r in rows:
                    if self._cutoff_us is not None and r["committed_at_us"] and int(r["committed_at_us"]) > self._cutoff_us:
                        continue
                    commits.append(
                        {
                            "author": r["author_name"],
                            "message": str(r["message"])[:500],
                            "oid": str(r["oid"]),
                        }
                    )
                return {"commits": commits}

            if name == "git_show":
                oid = str(arguments.get("oid") or "")
                crow = conn.execute(
                    "SELECT oid, author_name, committed_at_us, message FROM commits WHERE oid = ?",
                    (oid,),
                ).fetchone()
                if not crow:
                    return {"error": "commit_not_found"}
                if self._cutoff_us is not None and crow["committed_at_us"] and int(crow["committed_at_us"]) > self._cutoff_us:
                    return {"error": "commit_after_cutoff"}
                h_rows = conn.execute(
                    "SELECT h.patch_text FROM hunks h JOIN changes c ON c.id = h.change_id WHERE c.commit_oid = ? ORDER BY h.ordinal ASC LIMIT 10",
                    (oid,),
                ).fetchall()
                return {
                    "message": str(crow["message"]),
                    "oid": oid,
                    "patches": [str(hr["patch_text"])[:2000] for hr in h_rows if hr["patch_text"]],
                }

            if name == "list_discussions":
                lim = min(50, max(1, int(arguments.get("limit", 10))))
                s_rows = conn.execute(
                    "SELECT id, kind, provider_id FROM sources WHERE availability = 'available' AND kind IN ('pull_request', 'issue') ORDER BY id ASC LIMIT ?",
                    (lim,),
                ).fetchall()
                from codecontext.retrieval import _select_eligible_version_for_source

                discs = []
                for sr in s_rows:
                    sel_v, _ = _select_eligible_version_for_source(conn, str(sr["id"]), self._cutoff_us)
                    if sel_v:
                        vr = conn.execute("SELECT title FROM source_versions WHERE id = ?", (sel_v,)).fetchone()
                        discs.append(
                            {
                                "kind": str(sr["kind"]),
                                "number": str(sr["provider_id"]),
                                "source_id": str(sr["id"]),
                                "title": vr["title"] if vr else None,
                            }
                        )
                return {"discussions": discs}

            if name == "read_discussion":
                sid = str(arguments.get("source_id") or "")
                from codecontext.retrieval import _select_eligible_version_for_source

                sel_v, _ = _select_eligible_version_for_source(conn, sid, self._cutoff_us)
                if not sel_v:
                    return {"error": "discussion_not_eligible_or_not_found"}
                vr = conn.execute("SELECT title, body FROM source_versions WHERE id = ?", (sel_v,)).fetchone()
                child_rows = conn.execute(
                    "SELECT id, kind FROM sources WHERE parent_source_id = ? AND availability = 'available' ORDER BY id ASC",
                    (sid,),
                ).fetchall()
                comments = []
                for cr in child_rows:
                    c_ver, _ = _select_eligible_version_for_source(conn, str(cr["id"]), self._cutoff_us)
                    if c_ver:
                        cvr = conn.execute("SELECT body FROM source_versions WHERE id = ?", (c_ver,)).fetchone()
                        if cvr and cvr["body"]:
                            comments.append({"kind": str(cr["kind"]), "text": str(cvr["body"])[:2000]})
                return {
                    "body": str(vr["body"])[:4000] if vr and vr["body"] else None,
                    "comments": comments,
                    "source_id": sid,
                    "title": vr["title"] if vr else None,
                }

        return {"error": "unknown_tool"}


@runtime_checkable
class EvaluationRunner(Protocol):
    def run(self, request: EvalRequest, tools: EvalTools) -> EvalResult: ...


class ScriptedEvaluationRunner:
    """Deterministic scripted runner for harness and refusal-boundary testing."""

    def __init__(
        self,
        script: list[tuple[str, dict[str, Any]]] | None = None,
        *,
        final_answer: str = "Scripted evidence-backed explanation.",
        forge_events: list[ToolEvent] | None = None,
    ) -> None:
        self._script = script
        self._final_answer = final_answer
        self._forge_events = forge_events

    def run(self, request: EvalRequest, tools: EvalTools) -> EvalResult:
        started_at = datetime.now(UTC)
        actions = (
            self._script
            if self._script is not None
            else (
                [("context", {"query": request.question})]
                if request.arm in ("lexical", "memory")
                else [("list_files", {}), ("git_log", {"limit": 5})]
            )
        )
        for tool_name, tool_args in actions:
            res = tools.call(tool_name, tool_args)
            if isinstance(res, dict) and "error" in res:
                err_c = (res["error"] or {}).get("code")
                if err_c == "BUDGET_EXHAUSTED":
                    break

        finished_at = datetime.now(UTC)
        events = self._forge_events if self._forge_events is not None else tools.events
        return EvalResult(
            run_id=request.run_id,
            state="complete",
            final_answer=self._final_answer,
            events=events,
            input_tokens=100,
            output_tokens=50,
            cost_usd=Decimal("0"),
            provider_request_ids=[f"script_{request.run_id}"],
            started_at=started_at,
            finished_at=finished_at,
            error_code=None,
            runner_fingerprint="scripted_runner_v1",
        )


def execute_eval_run(
    store: Store,
    request: EvalRequest,
    runner: EvaluationRunner,
    *,
    output_dir: Path | None = None,
    token_counter: Callable[[str], int] = default_token_estimator,
) -> EvalResult:
    """Execute one evaluation run through the harness dispatcher `EvalTools`.
    Enforces authoritative event logs, answer token limits, and timeout/budget states.
    """
    tools = EvalTools(store, request, token_counter=token_counter)
    started_at = datetime.now(UTC)
    try:
        runner_res = runner.run(request, tools)
    except Exception as exc:
        finished_at = datetime.now(UTC)
        res = EvalResult(
            run_id=request.run_id,
            state="runner_error",
            final_answer=None,
            events=tools.events,
            input_tokens=None,
            output_tokens=None,
            cost_usd=None,
            provider_request_ids=[],
            started_at=started_at,
            finished_at=finished_at,
            error_code=type(exc).__name__,
            runner_fingerprint="unknown",
        )
        if output_dir is not None:
            _append_eval_run_log(output_dir, request, res)
        return res

    # Verify runner did not forge or bypass authoritative `tools.events`
    auth_events = tools.events
    state: Literal["complete", "timeout", "budget_exhausted", "runner_error", "protocol_invalid"] = (
        runner_res.state
    )
    error_code = runner_res.error_code

    if [e.model_dump() for e in runner_res.events] != [e.model_dump() for e in auth_events]:
        state = "protocol_invalid"
        error_code = "RUNNER_FORGED_EVENTS"
    elif runner_res.final_answer and int(token_counter(runner_res.final_answer)) > request.max_answer_tokens:
        state = "protocol_invalid"
        error_code = "ANSWER_TOKEN_LIMIT_EXCEEDED"
    elif tools.timeout_refused:
        state = "timeout"
        error_code = "TIMEOUT_EXCEEDED"
    elif tools.budget_refused:
        state = "budget_exhausted"
        error_code = "BUDGET_EXHAUSTED"

    verified_result = EvalResult(
        run_id=request.run_id,
        state=state,
        final_answer=runner_res.final_answer if state != "protocol_invalid" else None,
        events=auth_events,
        input_tokens=runner_res.input_tokens,
        output_tokens=runner_res.output_tokens,
        cost_usd=runner_res.cost_usd,
        provider_request_ids=runner_res.provider_request_ids,
        started_at=runner_res.started_at,
        finished_at=runner_res.finished_at,
        error_code=error_code,
        runner_fingerprint=runner_res.runner_fingerprint,
    )
    if output_dir is not None:
        _append_eval_run_log(output_dir, request, verified_result)
    return verified_result


def _append_eval_run_log(output_dir: Path, request: EvalRequest, result: EvalResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_file = output_dir / "runs.jsonl"
    record = {
        "request": request.model_dump(mode="json"),
        "result": result.model_dump(mode="json"),
    }
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with runs_file.open("a", encoding="utf-8") as f:
        f.write(line)
