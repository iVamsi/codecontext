import json
from decimal import Decimal
from typing import Any

from codecontext.models import (
    ModelAdapter,
    ModelIdentity,
    ModelRequest,
    ModelResponse,
)


class DeterministicFixtureAdapter:
    """Runnable deterministic fixture adapter for tests and first-use demonstrations.
    Returns known fixture claims for recognized fixture inputs, has zero prices,
    and raises for unknown inputs when `strict_fixture_only=True`.
    """

    def __init__(
        self,
        *,
        strict_fixture_only: bool = True,
        input_price_per_million: Decimal | None = Decimal("0"),
        output_price_per_million: Decimal | None = Decimal("0"),
    ) -> None:
        self._strict = strict_fixture_only
        self._identity = ModelIdentity(
            adapter_name="fixture_adapter",
            model_name="deterministic-fixture",
            model_revision="1.0.0",
            configuration_fingerprint="deterministic_v1",
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            framing_input_tokens=10,
        )

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload = json.loads(request.input_json)
        if "original_input" in payload:
            payload = payload["original_input"]

        slots: list[dict[str, Any]] = payload.get("slots", [])
        anchors: list[dict[str, Any]] = payload.get("anchor_options", [])
        all_slot_ids = [str(s["slot"]) for s in slots]
        target_keys = [
            str(a["key"])
            for a in anchors
            if "undocumented" not in str(a.get("symbol_id", ""))
            and a.get("line_start") != 16
            and a.get("line_start") != 17
        ]
        if not target_keys and anchors:
            target_keys = [str(anchors[0]["key"])]

        # Check if this is a summary extraction request
        if "statements" in request.output_schema.get("properties", {}):
            stmts = []
            for s in slots[:3]:
                txt = str(s.get("text", "")).strip()
                first_line = txt.splitlines()[0].strip() if txt else ""
                if len(first_line) >= 10:
                    stmts.append(
                        {
                            "evidence": [{"quote": first_line, "slot": s["slot"]}],
                            "text": f"Documented note: {first_line[:200]}",
                        }
                    )
            out_str = json.dumps(
                {
                    "inspected_slots": all_slot_ids,
                    "statements": stmts,
                    "unresolved": [],
                },
                sort_keys=True,
            )
            return ModelResponse(
                output_json=out_str,
                input_tokens=max(1, len(request.input_json) // 4),
                output_tokens=max(1, len(out_str) // 4),
                provider_request_id=f"fix_{request.request_id}",
            )

        # Episode claim extraction: inspect slots for F01/F02/F05 fixture patterns
        rationale_slot: str | None = None
        rationale_quote: str | None = None
        rejection_slot: str | None = None
        rejection_quote: str | None = None
        corroboration_slot: str | None = None
        corroboration_quote: str | None = None
        corroboration_key: str | None = None

        for s in slots:
            stext = str(s.get("text", ""))
            sid = str(s["slot"])
            if "Limit retries to three because each retry holds a worker" in stext:
                rationale_slot = sid
                rationale_quote = (
                    "Limit retries to three because each retry holds a worker and the legacy proxy "
                    "duplicates requests after its timeout."
                )
            elif "Rejected unlimited retries: holding workers indefinitely" in stext:
                rejection_slot = sid
                rejection_quote = (
                    "Rejected unlimited retries: holding workers indefinitely exhausts the worker pool "
                    "under proxy brownouts."
                )
            elif "ADR-001: The 3-attempt retry limit remains required at revision" in stext:
                corroboration_slot = sid
                corroboration_quote = stext.strip().splitlines()[0]
                if anchors:
                    corroboration_key = str(anchors[0]["key"])
                    for a in anchors:
                        if str(a.get("revision")) in corroboration_quote:
                            corroboration_key = str(a["key"])
                            break

        if rationale_slot and rationale_quote:
            ev_list = [{"quote": rationale_quote, "slot": rationale_slot}]
            lc_list = [{"quote": rationale_quote, "slot": rationale_slot}]
            counter_list = (
                [{"quote": rejection_quote, "slot": rejection_slot}]
                if (rejection_slot and rejection_quote)
                else []
            )
            corr_list = (
                [
                    {
                        "quote": corroboration_quote,
                        "slot": corroboration_slot,
                        "target_anchor_key": corroboration_key,
                    }
                ]
                if (corroboration_slot and corroboration_quote and corroboration_key)
                else []
            )
            claims = [
                {
                    "alternatives": ["Unlimited retries with exponential backoff"],
                    "constraints": [
                        "Each retry holds a worker",
                        "Legacy proxy duplicates requests after its timeout",
                    ],
                    "corroboration": corr_list,
                    "counterevidence": counter_list,
                    "evidence": ev_list,
                    "lifecycle": "adopted",
                    "lifecycle_evidence": lc_list,
                    "rationale": (
                        "Each retry holds a worker and the legacy proxy duplicates requests after its timeout."
                    ),
                    "statement": "Bound send retries to three attempts.",
                    "support": "explicit",
                    "target_anchor_keys": target_keys,
                }
            ]
            out_str = json.dumps(
                {
                    "claims": claims,
                    "inspected_slots": all_slot_ids,
                    "unresolved": [],
                },
                sort_keys=True,
            )
            return ModelResponse(
                output_json=out_str,
                input_tokens=max(1, len(request.input_json) // 4),
                output_tokens=max(1, len(out_str) // 4),
                provider_request_id=f"fix_{request.request_id}",
            )

        # Known undocumented fixture commit / function
        if any(
            "Initial transport send implementation" in str(s.get("text", ""))
            or "Bound send retries to 3 attempts" in str(s.get("text", ""))
            or "Update README" in str(s.get("text", ""))
            or "undocumented" in str(s.get("text", "")).lower()
            for s in slots
        ):
            out_str = json.dumps(
                {
                    "claims": [],
                    "inspected_slots": all_slot_ids,
                    "unresolved": [],
                },
                sort_keys=True,
            )
            return ModelResponse(
                output_json=out_str,
                input_tokens=max(1, len(request.input_json) // 4),
                output_tokens=max(1, len(out_str) // 4),
                provider_request_id=f"fix_{request.request_id}",
            )

        if self._strict:
            raise RuntimeError("DeterministicFixtureAdapter received unrecognized non-fixture input.")

        out_str = json.dumps(
            {
                "claims": [],
                "inspected_slots": all_slot_ids,
                "unresolved": [],
            },
            sort_keys=True,
        )
        return ModelResponse(
            output_json=out_str,
            input_tokens=max(1, len(request.input_json) // 4),
            output_tokens=max(1, len(out_str) // 4),
            provider_request_id=f"fix_{request.request_id}",
        )


def create_adapter() -> ModelAdapter:
    return DeterministicFixtureAdapter(strict_fixture_only=False)
