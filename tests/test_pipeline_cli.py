from __future__ import annotations

import csv
import json
from pathlib import Path

from multicat.pipeline.run_pipeline_cli import (
    _screening_is_skip,
    run_pipeline,
)


class FakeRunner:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.validation_calls = 0

    def run_agent(self, kind: str, paper_dir: Path, project_root: Path) -> dict:
        self.calls.append((kind, paper_dir.name))
        if kind == "screening":
            report = {"decision": "keep", "reason": "ok"}
            (paper_dir / "06_screening.json").write_text(json.dumps(report), encoding="utf-8")
            return report
        if kind == "extraction":
            payload = {
                "paper": {
                    "paper_id": paper_dir.name,
                    "doi": "10.0000/example",
                    "title": "Example",
                    "journal": "Journal",
                    "year": 2026,
                    "publisher": "publisher",
                    "xml_source": "pmc",
                },
                "catalysts": [],
                "reactions": [],
                "extraction_meta": {"schema_version": "v1", "status": "complete", "notes": None},
            }
            (paper_dir / "07_extraction.json").write_text(json.dumps(payload), encoding="utf-8")
            return payload
        if kind == "repair":
            payload = json.loads((paper_dir / "07_extraction.json").read_text(encoding="utf-8"))
            payload["catalysts"] = [
                {
                    "catalyst_id": "cat_001",
                    "catalyst_name": "Amberlyst-15",
                    "support_status": "unsupported",
                    "support_class": "none",
                    "unsupported_class": "polymer_resin",
                    "composition_raw": "Amberlyst-15",
                    "metals": [],
                    "synthesis": {},
                    "characterization": {},
                    "evidence": "Table 1",
                }
            ]
            payload["reactions"] = [
                {
                    "reaction_id": "rxn_001",
                    "paper_id": paper_dir.name,
                    "catalyst_id": "cat_001",
                    "substrate": {"substrate_name": "cellulose"},
                    "conditions": {},
                    "performance_metrics": [
                        {
                            "metric_name": "yield",
                            "product_name": "levulinic_acid",
                            "value": 29.91,
                            "unit": "%",
                            "basis": "unknown",
                            "original_label": "LA/%",
                            "evidence": "Table 1",
                        }
                    ],
                    "source_type": "table",
                    "evidence": "Table 1",
                }
            ]
            (paper_dir / "07_extraction.json").write_text(json.dumps(payload), encoding="utf-8")
            (paper_dir / "10_repair_log.json").write_text(
                json.dumps([{"action": "added missing reaction"}]),
                encoding="utf-8",
            )
            return payload
        if kind == "judge":
            report = {"status": "pass", "issues": [], "summary": "ok"}
            (paper_dir / "09_judge.json").write_text(json.dumps(report), encoding="utf-8")
            return report
        raise AssertionError(f"Unexpected agent kind: {kind}")

    def validate(self, paper_dir: Path) -> dict:
        self.validation_calls += 1
        status = "fail" if self.validation_calls == 1 else "pass"
        report = {
            "status": status,
            "issues": [] if status == "pass" else [{"message": "missing row"}],
            "stats": {"catalysts_count": 0, "reactions_count": 0},
        }
        (paper_dir / "08_validation.json").write_text(json.dumps(report), encoding="utf-8")
        return report


def test_run_pipeline_repairs_failed_validation_and_writes_reports(tmp_path: Path):
    papers_root = tmp_path / "papers"
    output_root = tmp_path / "outputs"
    paper_dir = papers_root / "P000001"
    paper_dir.mkdir(parents=True)
    (paper_dir / "05_agent_input.md").write_text(
        "# Agent Input\n## Textual Tables\n| Catalyst | yield |", encoding="utf-8"
    )
    runner = FakeRunner()

    result = run_pipeline(
        project_root=tmp_path,
        papers_root=papers_root,
        output_root=output_root,
        paper_ids=["P000001"],
        max_repair_loops=2,
        runner=runner,
    )

    assert result.accepted == ["P000001"]
    assert runner.calls == [
        ("screening", "P000001"),
        ("extraction", "P000001"),
        ("repair", "P000001"),
        ("judge", "P000001"),
    ]
    report_path = output_root / "pipeline_run_report.csv"
    with report_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["paper_id"] == "P000001"
    assert rows[0]["final_status"] == "accepted"
    assert rows[0]["validation_repair_attempts"] == "1"
    assert rows[0]["judge_repair_attempts"] == "0"
    assert "extraction_model" in rows[0]
    assert "judge_model" in rows[0]
    assert "repair_model" in rows[0]
    assert (output_root / "reaction_audit_table_v1.csv").exists()
    assert (output_root / "ml_ready_table_v1.csv").exists()


def test_run_pipeline_report_preserves_rows_from_earlier_batches(tmp_path: Path):
    """Regression test: an overnight run is often split into several
    --paper-ids batches. A later batch must merge into the existing
    pipeline_run_report.csv by paper_id, not overwrite it -- otherwise
    every paper from an earlier batch silently disappears from the report
    even though its papers/<id>/08_validation.json and 09_judge.json are
    still on disk and its status hasn't changed.
    """
    papers_root = tmp_path / "papers"
    output_root = tmp_path / "outputs"
    output_root.mkdir(parents=True)
    (output_root / "pipeline_run_report.csv").write_text(
        "paper_id,final_status,validation_status,judge_status,"
        "validation_repair_attempts,judge_repair_attempts,"
        "extraction_model,judge_model,repair_model,notes\n"
        "P000999,accepted,pass,pass,0,0,modelA,modelB,modelC,\n",
        encoding="utf-8-sig",
    )
    paper_dir = papers_root / "P000001"
    paper_dir.mkdir(parents=True)
    (paper_dir / "05_agent_input.md").write_text(
        "# Agent Input\n## Textual Tables\n| Catalyst | yield |", encoding="utf-8"
    )
    runner = FakeRunner()

    run_pipeline(
        project_root=tmp_path,
        papers_root=papers_root,
        output_root=output_root,
        paper_ids=["P000001"],
        max_repair_loops=2,
        runner=runner,
    )

    report_path = output_root / "pipeline_run_report.csv"
    with report_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    paper_ids = {row["paper_id"] for row in rows}
    assert paper_ids == {"P000999", "P000001"}
    old_row = next(row for row in rows if row["paper_id"] == "P000999")
    assert old_row["final_status"] == "accepted"


def test_run_pipeline_can_skip_existing_passed_papers(tmp_path: Path):
    papers_root = tmp_path / "papers"
    output_root = tmp_path / "outputs"
    paper_dir = papers_root / "P000001"
    paper_dir.mkdir(parents=True)
    (paper_dir / "05_agent_input.md").write_text(
        "# Agent Input\n## Textual Tables\n| Catalyst | yield |", encoding="utf-8"
    )
    payload = {
        "paper": {"paper_id": "P000001"},
        "catalysts": [],
        "reactions": [],
        "extraction_meta": {"schema_version": "v1"},
    }
    (paper_dir / "07_extraction.json").write_text(json.dumps(payload), encoding="utf-8")
    (paper_dir / "08_validation.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    (paper_dir / "09_judge.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    runner = FakeRunner()

    result = run_pipeline(
        project_root=tmp_path,
        papers_root=papers_root,
        output_root=output_root,
        paper_ids=["P000001"],
        max_repair_loops=2,
        skip_existing_pass=True,
        runner=runner,
    )

    assert result.accepted == ["P000001"]
    assert runner.calls == []


def test_run_pipeline_marks_screening_skip_as_incomplete_without_agents(tmp_path: Path):
    papers_root = tmp_path / "papers"
    output_root = tmp_path / "outputs"
    paper_dir = papers_root / "P000001"
    paper_dir.mkdir(parents=True)
    (paper_dir / "05_agent_input.md").write_text(
        "# Agent Input\n## Textual Tables\n| Catalyst | yield |", encoding="utf-8"
    )

    class SkippingFakeRunner(FakeRunner):
        def run_agent(self, kind: str, paper_dir: Path, project_root: Path) -> dict:
            if kind == "screening":
                self.calls.append((kind, paper_dir.name))
                report = {"decision": "skip", "reason": "out of scope"}
                (paper_dir / "06_screening.json").write_text(json.dumps(report), encoding="utf-8")
                return report
            return super().run_agent(kind, paper_dir, project_root)

    runner = SkippingFakeRunner()

    result = run_pipeline(
        project_root=tmp_path,
        papers_root=papers_root,
        output_root=output_root,
        paper_ids=["P000001"],
        runner=runner,
    )

    assert result.incomplete == ["P000001"]
    assert runner.calls == [("screening", "P000001")]
    with (output_root / "incomplete_papers.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["paper_id"] == "P000001"
    assert rows[0]["notes"] == "screening_skip"


def test_run_pipeline_skips_screening_agent_when_cached_decision_is_keep(tmp_path: Path):
    """Regression test (see memory bug #14, fixed 2026-06-26): re-running the
    pipeline on a paper that already has a cached 06_screening.json with
    decision=keep must NOT re-invoke the Screening Agent -- doing so wasted
    an LLM call on every batch re-run for papers whose scope decision was
    already settled. The cache is only trusted for decision=keep; anything
    else still triggers a real call (see the next test).
    """
    papers_root = tmp_path / "papers"
    output_root = tmp_path / "outputs"
    paper_dir = papers_root / "P000001"
    paper_dir.mkdir(parents=True)
    (paper_dir / "05_agent_input.md").write_text(
        "# Agent Input\n## Textual Tables\n| Catalyst | yield |", encoding="utf-8"
    )
    (paper_dir / "06_screening.json").write_text(
        json.dumps({"decision": "keep", "reason": "prior run"}), encoding="utf-8"
    )
    runner = FakeRunner()

    run_pipeline(
        project_root=tmp_path,
        papers_root=papers_root,
        output_root=output_root,
        paper_ids=["P000001"],
        runner=runner,
    )

    assert ("screening", "P000001") not in runner.calls
    assert (paper_dir / "06_screening.json").exists()


def test_run_pipeline_calls_screening_agent_when_cached_decision_is_not_keep(tmp_path: Path):
    """Complement to the cache-skip test above: a cached screening result
    that is NOT decision=keep (e.g. a stale/incorrect skip, or malformed
    JSON) must not be trusted blindly -- the pipeline should still call the
    Screening Agent so its scope rules get a chance to run.
    """
    papers_root = tmp_path / "papers"
    output_root = tmp_path / "outputs"
    paper_dir = papers_root / "P000001"
    paper_dir.mkdir(parents=True)
    (paper_dir / "05_agent_input.md").write_text(
        "# Agent Input\n## Textual Tables\n| Catalyst | yield |", encoding="utf-8"
    )
    (paper_dir / "06_screening.json").write_text(
        json.dumps({"decision": "skip", "reason": "stale prior run"}), encoding="utf-8"
    )
    runner = FakeRunner()

    run_pipeline(
        project_root=tmp_path,
        papers_root=papers_root,
        output_root=output_root,
        paper_ids=["P000001"],
        runner=runner,
    )

    assert ("screening", "P000001") in runner.calls


def test_run_pipeline_isolates_one_paper_crash_from_the_rest_of_the_batch(tmp_path: Path):
    """Regression test: when papers run in parallel (ThreadPoolExecutor), one
    paper raising an exception (e.g. malformed LLM JSON response) must not
    crash the whole batch and must not prevent other papers' results from
    being collected and written to the reports.
    """
    papers_root = tmp_path / "papers"
    output_root = tmp_path / "outputs"
    for paper_id in ("P000001", "P000002"):
        paper_dir = papers_root / paper_id
        paper_dir.mkdir(parents=True)
        (paper_dir / "05_agent_input.md").write_text(
            "# Agent Input\n## Textual Tables\n| Catalyst | yield |", encoding="utf-8"
        )

    class CrashingFakeRunner(FakeRunner):
        def run_agent(self, kind: str, paper_dir: Path, project_root: Path) -> dict:
            if kind == "extraction" and paper_dir.name == "P000002":
                raise ValueError("Expecting value: line 2 column 11 (char 12)")
            return super().run_agent(kind, paper_dir, project_root)

    runner = CrashingFakeRunner()

    result = run_pipeline(
        project_root=tmp_path,
        papers_root=papers_root,
        output_root=output_root,
        paper_ids=["P000001", "P000002"],
        max_repair_loops=2,
        runner=runner,
    )

    assert result.accepted == ["P000001"]
    assert result.incomplete == ["P000002"]
    crashed = next(r for r in result.paper_results if r.paper_id == "P000002")
    assert "pipeline_error" in crashed.notes


def test_screening_is_skip_helper(tmp_path: Path):
    paper_dir = tmp_path / "P000001"
    paper_dir.mkdir()
    assert _screening_is_skip(paper_dir) is False  # no file -> not skip
    (paper_dir / "06_screening.json").write_text(json.dumps({"decision": "keep"}), encoding="utf-8")
    assert _screening_is_skip(paper_dir) is False
    (paper_dir / "06_screening.json").write_text(json.dumps({"decision": "skip"}), encoding="utf-8")
    assert _screening_is_skip(paper_dir) is True



def test_run_pipeline_skips_repair_when_judge_issues_are_all_unrecoverable(tmp_path: Path):
    """Judge fail with only unrecoverable issue types (out_of_scope_catalyst,
    scope_violation) must result in immediate rejected without calling repair.
    Repair cannot fix "this catalyst is homogeneous and should never have been
    extracted" — the underlying fact in the paper has not changed.
    """
    papers_root = tmp_path / "papers"
    output_root = tmp_path / "outputs"
    paper_dir = papers_root / "P000001"
    paper_dir.mkdir(parents=True)
    (paper_dir / "05_agent_input.md").write_text(
        "# Agent Input\n## Textual Tables\n| Catalyst | yield |", encoding="utf-8"
    )

    class UnrecoverableJudgeFakeRunner(FakeRunner):
        def __init__(self):
            super().__init__()
            # make validate always pass so we reach judge immediately
            self._validate_always_pass = True

        def validate(self, paper_dir: Path) -> dict:
            report = {"status": "pass", "issues": [], "stats": {"catalysts_count": 1, "reactions_count": 1}}
            (paper_dir / "08_validation.json").write_text(json.dumps(report), encoding="utf-8")
            return report

        def run_agent(self, kind: str, paper_dir: Path, project_root: Path) -> dict:
            if kind == "judge":
                self.calls.append((kind, paper_dir.name))
                report = {
                    "status": "fail",
                    "issues": [
                        {
                            "severity": "error",
                            "type": "out_of_scope_catalyst",
                            "path": "reactions",
                            "message": "TsOH is a homogeneous catalyst and must not be extracted.",
                        }
                    ],
                    "summary": "unrecoverable: homogeneous catalyst",
                }
                (paper_dir / "09_judge.json").write_text(json.dumps(report), encoding="utf-8")
                return report
            return super().run_agent(kind, paper_dir, project_root)

    runner = UnrecoverableJudgeFakeRunner()

    result = run_pipeline(
        project_root=tmp_path,
        papers_root=papers_root,
        output_root=output_root,
        paper_ids=["P000001"],
        max_repair_loops=2,
        runner=runner,
    )

    assert result.rejected == ["P000001"]
    # repair must NOT have been called
    assert ("repair", "P000001") not in runner.calls
    # judge must have been called exactly once
    assert runner.calls.count(("judge", "P000001")) == 1


def test_run_pipeline_writes_ml_core_agent_input_from_text_package(tmp_path: Path):
    papers_root = tmp_path / "papers"
    output_root = tmp_path / "outputs"
    paper_dir = papers_root / "P000001"
    paper_dir.mkdir(parents=True)
    package = {
        "paper_id": "P000001",
        "metadata": {"doi": "10.0000/mlcore"},
        "title": "Catalytic conversion",
        "abstract": "Glucose was converted.",
        "sections": [{"title": "Results", "text": "Reaction performance."}],
        "tables": [
            {
                "caption": "Table 1 Catalytic conversion.",
                "rows": [["Entry", "Catalyst", "HMF yield (%)"], ["1", "CCC", "41.2"]],
            },
            {
                "caption": "Table 2 BET surface areas.",
                "rows": [["Sample", "BET surface"], ["CCC", "24.9"]],
            },
        ],
        "figure_captions": [{"caption": "Figure 1 HMF yield."}],
    }
    (paper_dir / "03_text_package.json").write_text(json.dumps(package), encoding="utf-8")
    runner = FakeRunner()

    run_pipeline(
        project_root=tmp_path,
        papers_root=papers_root,
        output_root=output_root,
        paper_ids=["P000001"],
        extraction_mode="ml_core",
        runner=runner,
    )

    agent_input = (paper_dir / "05_agent_input.md").read_text(encoding="utf-8")
    assert "Catalytic conversion" in agent_input
    assert "BET surface" in agent_input
    assert "Figure 1" not in agent_input
