from multicat.text_acquisition.agent_input_builder import (
    build_agent_input_markdown,
    _strip_footnote_marker,
)


def test_build_agent_input_markdown_keeps_high_value_sections_and_tables():
    package = {
        "paper_id": "P000001",
        "metadata": {"doi": "10.0000/example"},
        "title": "Glucose conversion to HMF",
        "abstract": "A heterogeneous catalyst converted glucose to HMF.",
        "sections": [
            {"title": "Introduction", "text": "General biomass background."},
            {"title": "Experimental", "text": "Glucose, catalyst, and DMSO were loaded."},
            {"title": "Results and Discussion", "text": "HMF yield reached 59%."},
            {"title": "Conclusions", "text": "The catalyst was effective."},
        ],
        "tables": [
            {
                "caption": "Table 1 Reaction results.",
                "rows": [["Substrate", "Catalyst", "Yield"], ["Glucose", "Nb-Al", "59%"]],
            }
        ],
        "figure_captions": [{"id": "f6", "caption": "Figure 6 HMF yield from glucose."}],
    }

    markdown = build_agent_input_markdown(package)

    assert "# Agent Input: Glucose conversion to HMF" in markdown
    assert "DOI: 10.0000/example" in markdown
    assert "A heterogeneous catalyst converted glucose" in markdown
    assert "## Experimental" in markdown
    assert "## Results and Discussion" in markdown
    assert "General biomass background" not in markdown
    assert "The catalyst was effective" not in markdown
    assert "| Glucose | Nb-Al | 59% |" in markdown
    assert "Figure 6 HMF yield from glucose." in markdown


def test_build_agent_input_markdown_falls_back_when_no_high_value_sections():
    package = {
        "paper_id": "P000002",
        "metadata": {"doi": "10.0000/fallback"},
        "title": "Cellulose conversion",
        "abstract": "Cellulose was converted.",
        "sections": [{"title": "Main text", "text": "Levulinic acid yield was reported."}],
        "tables": [],
        "figure_captions": [],
    }

    markdown = build_agent_input_markdown(package)

    assert "## Main text" in markdown
    assert "Levulinic acid yield was reported." in markdown


def test_build_agent_input_markdown_ml_core_keeps_only_reaction_tables():
    package = {
        "paper_id": "P000003",
        "metadata": {"doi": "10.0000/mlcore"},
        "title": "Catalytic conversion",
        "abstract": "Glucose was converted to HMF.",
        "sections": [
            {"title": "Results and discussion", "text": "Table 1 shows catalyst performance."},
            {"title": "Recycling of catalyst", "text": "The catalyst was reused."},
        ],
        "tables": [
            {
                "caption": "Table 1 Catalytic conversion of glucose into HMF.",
                "rows": [["Entry", "Catalyst", "HMF yield (%)"], ["1", "CCC", "41.2"]],
            },
            {
                "caption": "Table 2 Successive use of CCC.",
                "rows": [["Run", "HMF yield (%)"], ["1", "46.4"]],
            },
            {
                "caption": "Table 3 BET surface areas.",
                "rows": [["Sample", "BET surface"], ["CCC", "24.9"]],
            },
        ],
        "figure_captions": [{"caption": "Figure 3 HMF yield from sugars."}],
    }

    markdown = build_agent_input_markdown(package, mode="ml_core")

    assert "Table 1 Catalytic conversion" in markdown
    assert "| 1 | CCC | 41.2 |" in markdown
    assert "Successive use" not in markdown
    assert "BET surface" in markdown  # characterization tables are kept in ml_core
    assert "Figure 3" not in markdown
    assert "## Figure Captions" not in markdown


def test_build_agent_input_markdown_ml_core_truncates_long_sections():
    package = {
        "paper_id": "P000004",
        "metadata": {"doi": "10.0000/long"},
        "title": "Long paper",
        "abstract": "Abstract.",
        "sections": [
            {"title": "Results and discussion", "text": "A" * 5000},
            {"title": "Experimental", "text": "B" * 5000},
        ],
        "tables": [
            {
                "caption": "Table 1 Catalytic conversion.",
                "rows": [["Catalyst", "Yield"], ["CCC", "41.2"]],
            }
        ],
        "figure_captions": [],
    }

    markdown = build_agent_input_markdown(package, mode="ml_core")

    assert "A" * 2500 not in markdown
    assert "B" * 2500 not in markdown
    assert "[truncated for ml_core]" in markdown


def test_strip_footnote_marker_removes_isolated_trailing_letter():
    """Regression test: P000003's table cell "ICC d" (where 'd' is a footnote
    marker, not part of the catalyst name) caused the extraction LLM to
    repeatedly produce malformed/runaway JSON (observed twice, both failing
    around the same ~50000-character output position). Footnote letters must
    be stripped from table cells before they reach the LLM.
    """
    assert _strip_footnote_marker("ICC d") == "ICC"
    assert _strip_footnote_marker("HCl b") == "HCl"
    assert _strip_footnote_marker("H 2 SO 4 c") == "H 2 SO 4"


def test_strip_footnote_marker_does_not_touch_chemical_formulas():
    assert _strip_footnote_marker("Cs 2.5 H 0.5 PW 12 O 40") == "Cs 2.5 H 0.5 PW 12 O 40"
    assert _strip_footnote_marker("CCC") == "CCC"
    assert _strip_footnote_marker("No catalyst") == "No catalyst"
    assert _strip_footnote_marker("SO 4 2 - /ZrO 2") == "SO 4 2 - /ZrO 2"


def test_build_agent_input_markdown_strips_footnote_markers_from_table_cells():
    package = {
        "paper_id": "P000003",
        "metadata": {"doi": "10.0000/footnote"},
        "title": "Footnote table test",
        "abstract": "Abstract.",
        "sections": [],
        "tables": [
            {
                "caption": "Table 1 Conversion results.",
                "rows": [
                    ["Entry", "Catalyst", "HMF yield (%)"],
                    ["10", "HCl b", "18.3"],
                    ["12", "ICC d", "2.1"],
                ],
            }
        ],
        "figure_captions": [],
    }

    markdown = build_agent_input_markdown(package)

    assert "| HCl |" in markdown
    assert "| ICC |" in markdown
    assert "HCl b" not in markdown
    assert "ICC d" not in markdown
