from __future__ import annotations

import json
from pathlib import Path

from multicat.agents.run_agent_cli import run_agent


class DummyClient:
    def __init__(self, response: dict):
        self.response = response

    def chat_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        assert system_prompt
        assert "reaction table" in user_prompt
        return self.response


def test_run_agent_runs_screening_with_injected_client(tmp_path: Path):
    project_root = tmp_path
    (project_root / "prompts").mkdir()
    (project_root / "prompts" / "screening_prompt.md").write_text("screening prompt", encoding="utf-8")
    paper_dir = project_root / "papers" / "P000001"
    paper_dir.mkdir(parents=True)
    (paper_dir / "05_agent_input.md").write_text("reaction table", encoding="utf-8")

    result = run_agent(
        kind="screening",
        paper_dir=paper_dir,
        project_root=project_root,
        llm_client=DummyClient({"decision": "keep", "reason": "contains reaction table"}),
    )

    assert result == {
        "decision": "keep",
        "reason": "contains reaction table",
        "figure_labels": {},
    }
    saved = json.loads((paper_dir / "06_screening.json").read_text(encoding="utf-8"))
    assert saved["decision"] == "keep"


def test_run_agent_runs_extraction_with_injected_client(tmp_path: Path):
    project_root = tmp_path
    (project_root / "prompts").mkdir()
    (project_root / "prompts" / "extraction_prompt.md").write_text("extraction prompt", encoding="utf-8")
    paper_dir = project_root / "papers" / "P000001"
    paper_dir.mkdir(parents=True)
    (paper_dir / "05_agent_input.md").write_text("reaction table", encoding="utf-8")

    payload = {
        "paper": {},
        "catalysts": [],
        "reactions": [],
        "extraction_meta": {"status": "empty"},
    }
    result = run_agent(
        kind="extraction",
        paper_dir=paper_dir,
        project_root=project_root,
        llm_client=DummyClient(payload),
    )

    assert result == payload
    saved = json.loads((paper_dir / "07_extraction.json").read_text(encoding="utf-8"))
    assert saved["extraction_meta"]["status"] == "empty"


def test_run_agent_runs_judge_with_injected_client(tmp_path: Path):
    project_root = tmp_path
    (project_root / "prompts").mkdir()
    (project_root / "prompts" / "judge_prompt.md").write_text("judge prompt", encoding="utf-8")
    paper_dir = project_root / "papers" / "P000001"
    paper_dir.mkdir(parents=True)
    (paper_dir / "05_agent_input.md").write_text("reaction table", encoding="utf-8")
    (paper_dir / "07_extraction.json").write_text(
        json.dumps({"paper": {}, "catalysts": [], "reactions": [], "extraction_meta": {}}),
        encoding="utf-8",
    )
    (paper_dir / "08_validation.json").write_text(
        json.dumps({"status": "pass", "issues": []}),
        encoding="utf-8",
    )

    result = run_agent(
        kind="judge",
        paper_dir=paper_dir,
        project_root=project_root,
        llm_client=DummyClient({"status": "pass", "issues": [], "summary": "ok"}),
    )

    assert result["status"] == "pass"
    saved = json.loads((paper_dir / "09_judge.json").read_text(encoding="utf-8"))
    assert saved["summary"] == "ok"


def test_run_agent_runs_repair_with_injected_client(tmp_path: Path):
    project_root = tmp_path
    (project_root / "prompts").mkdir()
    (project_root / "prompts" / "repair_prompt.md").write_text("repair prompt", encoding="utf-8")
    paper_dir = project_root / "papers" / "P000001"
    paper_dir.mkdir(parents=True)
    (paper_dir / "05_agent_input.md").write_text("reaction table", encoding="utf-8")
    extraction = {
        "paper": {},
        "catalysts": [],
        "reactions": [],
        "extraction_meta": {"schema_version": "v1"},
    }
    (paper_dir / "07_extraction.json").write_text(json.dumps(extraction), encoding="utf-8")
    (paper_dir / "08_validation.json").write_text(
        json.dumps({"status": "fail", "issues": [{"message": "missing"}]}),
        encoding="utf-8",
    )

    result = run_agent(
        kind="repair",
        paper_dir=paper_dir,
        project_root=project_root,
        llm_client=DummyClient({"repaired_extraction": extraction, "repair_log": []}),
    )

    assert result == extraction
    saved = json.loads((paper_dir / "07_extraction.json").read_text(encoding="utf-8"))
    assert saved == extraction


def test_run_agent_writes_llm_call_log_with_token_usage(tmp_path: Path, monkeypatch):
    """Regression test: every real (non-injected) LLM call must append a
    JSON line to outputs/llm_call_log.jsonl recording paper_id, agent_kind,
    model, token usage, and duration -- this is the raw data needed to
    report total cost and average call time in the paper's methods section.
    pipeline_run_report.csv only has per-paper aggregates, not per-call data.
    """
    project_root = tmp_path
    (project_root / "prompts").mkdir()
    (project_root / "prompts" / "screening_prompt.md").write_text(
        "screening prompt", encoding="utf-8"
    )
    paper_dir = project_root / "papers" / "P000001"
    paper_dir.mkdir(parents=True)
    (paper_dir / "05_agent_input.md").write_text("reaction table", encoding="utf-8")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": '{"decision": "keep", "reason": "ok"}'}}
                ],
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 50,
                    "total_tokens": 550,
                },
            }

    monkeypatch.setattr(
        "multicat.llm.openrouter_client._requests.post",
        lambda url, *, json=None, headers=None, timeout=None: FakeResponse(),
    )

    log_path = project_root / "outputs" / "llm_call_log.jsonl"
    result = run_agent(
        kind="screening",
        paper_dir=paper_dir,
        project_root=project_root,
        llm_call_log_path=log_path,
    )

    assert result["decision"] == "keep"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["paper_id"] == "P000001"
    assert record["agent_kind"] == "screening"
    assert record["prompt_tokens"] == 500
    assert record["completion_tokens"] == 50
    assert record["total_tokens"] == 550
    assert record["succeeded"] is True
    assert "duration_seconds" in record
