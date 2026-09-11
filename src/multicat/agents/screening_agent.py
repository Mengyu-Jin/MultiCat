from __future__ import annotations

from pathlib import Path

from multicat.agents.io import read_agent_input, write_agent_json
from multicat.validator.schema_utils import validate_with_schema


def run_screening_agent(paper_dir: Path, llm_client, system_prompt: str) -> dict:
    agent_input = read_agent_input(paper_dir)
    result = _normalize_screening_result(
        llm_client.chat_json(system_prompt=system_prompt, user_prompt=agent_input)
    )
    result = _apply_deterministic_screening_overrides(result, agent_input)
    _validate_screening_result(result)
    write_agent_json(paper_dir, "06_screening.json", result)
    return result


def _normalize_screening_result(result: dict) -> dict:
    reason = result.get("reason", result.get("justification"))
    normalized = {
        "decision": result.get("decision"),
        "reason": reason,
    }
    if result.get("decision") == "keep":
        normalized["figure_labels"] = result.get("figure_labels") or {}
    return normalized


def _validate_screening_result(result: dict) -> None:
    validate_with_schema(result, "screening.schema.json", label="Screening Agent")


def _apply_deterministic_screening_overrides(result: dict, agent_input: str) -> dict:
    if result.get("decision") != "keep":
        return result

    reason = str(result.get("reason", ""))
    combined_text = f"{agent_input}\n{reason}".lower()
    homogeneous_salt_markers = [
        "alcl3",
        "aluminum chloride",
        "aluminium chloride",
        "metal chloride",
        "metal chlorides",
        "triflate",
        "triflates",
    ]
    dissolved_markers = [
        "dissolved",
        "dissolve",
        "in solution",
        "in aqueous phase",
        "in phosphate buffer",
    ]
    if any(marker in combined_text for marker in homogeneous_salt_markers) and any(
        marker in combined_text for marker in dissolved_markers
    ):
        return {
            "decision": "skip",
            "reason": (
                "deterministic override: soluble acid/salt catalyst appears to be dissolved, "
                "so the system is treated as homogeneous rather than heterogeneous. "
                f"Original screening reason: {reason}"
            ),
        }
    return result
