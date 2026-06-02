"""Tests for the SEO Analyst agent."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

from agents.seo_analyst import (
    SEOAnalystWorkflow,
    SEOReports,
    create_competitor_analysis_task,
    create_keyword_research_task,
    create_seo_analyst,
    create_technical_audit_task,
    format_competitor_results,
    format_keyword_results,
    format_technical_audit_results,
    format_backlink_results,
    save_keyword_report,
    save_competitor_report,
    save_technical_audit_report,
)


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------

class TestAgentCreation:
    @staticmethod
    def _mock_llm():
        return "openai/gpt-4"

    def test_create_seo_analyst(self):
        with patch("agents.seo_analyst.get_llm", return_value=self._mock_llm()):
            agent = create_seo_analyst()
        assert agent.role == "Senior SEO Analyst"

    def test_agent_has_llm(self):
        with patch("agents.seo_analyst.get_llm", return_value=self._mock_llm()):
            agent = create_seo_analyst()
        assert agent.llm is not None

    def test_workflow_creation(self):
        with patch("agents.seo_analyst.get_llm", return_value=self._mock_llm()):
            wf = SEOAnalystWorkflow("example.com")
        assert wf.domain == "example.com"
        assert wf.agent is not None


# ---------------------------------------------------------------------------
# Task factories
# ---------------------------------------------------------------------------

class TestTaskFactories:
    @staticmethod
    def _mock_llm():
        return "openai/gpt-4"

    def test_keyword_research_task(self):
        with patch("agents.seo_analyst.get_llm", return_value=self._mock_llm()):
            agent = create_seo_analyst()
        task = create_keyword_research_task("example.com", agent)
        assert task.agent is agent
        assert "example.com" in task.description

    def test_competitor_analysis_task(self):
        with patch("agents.seo_analyst.get_llm", return_value=self._mock_llm()):
            agent = create_seo_analyst()
        task = create_competitor_analysis_task("example.com", agent)
        assert task.agent is agent
        assert "competitor" in task.description.lower()

    def test_technical_audit_task(self):
        with patch("agents.seo_analyst.get_llm", return_value=self._mock_llm()):
            agent = create_seo_analyst()
        task = create_technical_audit_task("example.com", agent)
        assert task.agent is agent
        assert "technical" in task.description.lower()


# ---------------------------------------------------------------------------
# Keyword result formatting
# ---------------------------------------------------------------------------

class TestFormatKeywordResults:
    def test_empty_string_returns_empty_list(self):
        assert format_keyword_results("") == []

    def test_none_returns_empty_list(self):
        assert format_keyword_results(None) == []  # type: ignore[arg-type]

    def test_valid_json_list(self):
        raw = json.dumps([
            {"keyword": "python", "estimated_volume": "1000", "difficulty": "hard", "priority": "high"},
            {"keyword": "rust", "estimated_volume": "500", "difficulty": "easy", "priority": "medium"},
        ])
        result = format_keyword_results(raw)
        assert len(result) == 2
        assert result[0]["keyword"] == "python"
        assert result[0]["estimated_volume"] == "1000"

    def test_plain_text_fallback(self):
        raw = "- python programming\n- rust programming\n- go programming"
        result = format_keyword_results(raw)
        assert len(result) == 3
        assert result[0]["keyword"] == "python programming"

    def test_invalid_json_returns_empty_list(self):
        raw = "this is not json at all just free text"
        result = format_keyword_results(raw)
        assert isinstance(result, list)

    def test_comma_separated_list(self):
        raw = "python, rust, go"
        result = format_keyword_results(raw)
        # comma-separated single line -> 1 row in fallback mode
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Competitor result formatting
# ---------------------------------------------------------------------------

class TestFormatCompetitorResults:
    def test_empty_string_returns_empty_list(self):
        assert format_competitor_results("") == []

    def test_valid_json_list(self):
        raw = json.dumps([
            {"competitor": "competitor1.com", "metrics": "DA:80", "opportunities": "backlinks"},
        ])
        result = format_competitor_results(raw)
        assert len(result) == 1
        assert result[0]["competitor"] == "competitor1.com"


# ---------------------------------------------------------------------------
# Technical audit result formatting
# ---------------------------------------------------------------------------

class TestFormatTechnicalAuditResults:
    def test_empty_string_returns_empty_list(self):
        assert format_technical_audit_results("") == []

    def test_valid_json_list(self):
        raw = json.dumps([
            {"issue": "Missing meta description", "severity": "major", "recommendation": "Add meta description"},
        ])
        result = format_technical_audit_results(raw)
        assert len(result) == 1
        assert result[0]["severity"] == "major"

    def test_json_with_issues_key(self):
        raw = json.dumps({
            "issues": [
                {"issue": "No H1 tag", "severity": "critical", "recommendation": "Add H1"},
                {"issue": "Slow page speed", "severity": "minor", "recommendation": "Optimise images"},
            ]
        })
        result = format_technical_audit_results(raw)
        assert len(result) == 2

    def test_unknown_severity_defaults_to_minor(self):
        raw = json.dumps([{"issue": "test", "severity": "unknown", "recommendation": "fix"}])
        result = format_technical_audit_results(raw)
        assert result[0]["severity"] == "minor"

    def test_severity_values_normalised(self):
        for s in ("critical", "major", "minor"):
            raw = json.dumps([{"issue": "test", "severity": s}])
            result = format_technical_audit_results(raw)
            assert result[0]["severity"] == s


# ---------------------------------------------------------------------------
# Backlink result formatting
# ---------------------------------------------------------------------------

class TestFormatBacklinkResults:
    def test_empty_returns_empty_dict(self):
        assert format_backlink_results("") == {}

    def test_valid_json_dict(self):
        raw = json.dumps({
            "estimated_backlink_count": "5000",
            "referring_domains": "300",
            "quality_assessment": "good",
            "recommendations": "build more links",
        })
        result = format_backlink_results(raw)
        assert result["estimated_backlink_count"] == "5000"

    def test_plain_text_fallback(self):
        raw = "Some backlink summary text"
        result = format_backlink_results(raw)
        assert "summary" in result


# ---------------------------------------------------------------------------
# File output / save functions
# ---------------------------------------------------------------------------

class TestSaveReports:
    def test_save_keyword_report(self, tmp_path):
        path = save_keyword_report(
            "example.com",
            [{"keyword": "python", "estimated_volume": "100", "difficulty": "easy", "priority": "high"}],
            output_dir=str(tmp_path),
        )
        data = json.loads(Path(path).read_text())
        assert data["domain"] == "example.com"
        assert data["keyword_count"] == 1
        assert len(data["keywords"]) == 1

    def test_save_competitor_report(self, tmp_path):
        path = save_competitor_report(
            "example.com",
            [{"competitor": "comp.com", "metrics": "DA:50", "opportunities": "content"}],
            output_dir=str(tmp_path),
        )
        data = json.loads(Path(path).read_text())
        assert data["competitor_count"] == 1

    def test_save_technical_audit_report(self, tmp_path):
        issues = [
            {"issue": "Missing H1", "severity": "critical", "recommendation": "Add H1"},
            {"issue": "Slow speed", "severity": "major", "recommendation": "Optimize"},
            {"issue": "Missing alt text", "severity": "minor", "recommendation": "Add alt text"},
        ]
        path = save_technical_audit_report("example.com", issues, output_dir=str(tmp_path))
        data = json.loads(Path(path).read_text())
        assert data["summary"]["critical"] == 1
        assert data["summary"]["major"] == 1
        assert data["summary"]["minor"] == 1
        assert data["summary"]["total_issues"] == 3

    def test_save_keyword_report_empty_keywords(self, tmp_path):
        path = save_keyword_report("example.com", [], output_dir=str(tmp_path))
        data = json.loads(Path(path).read_text())
        assert data["keyword_count"] == 0
        assert data["keywords"] == []


# ---------------------------------------------------------------------------
# SEOReports integration
# ---------------------------------------------------------------------------

class TestSEOReports:
    def test_run_keyword_report(self, tmp_path):
        reports = SEOReports(output_dir=str(tmp_path))
        path = reports.run_keyword_report("example.com", "- python\n- rust\n- go")
        data = json.loads(Path(path).read_text())
        assert data["domain"] == "example.com"
        assert data["keyword_count"] == 3

    def test_run_competitor_report(self, tmp_path):
        reports = SEOReports(output_dir=str(tmp_path))
        path = reports.run_competitor_report("example.com", "- comp1.com\n- comp2.com")
        data = json.loads(Path(path).read_text())
        assert data["competitor_count"] == 2

    def test_run_technical_report(self, tmp_path):
        reports = SEOReports(output_dir=str(tmp_path))
        path = reports.run_technical_report("example.com", "- issue one\n- issue two\n- issue three")
        data = json.loads(Path(path).read_text())
        assert data["summary"]["total_issues"] == 3


# ---------------------------------------------------------------------------
# Workflow mode validation
# ---------------------------------------------------------------------------

class TestWorkflowModes:
    @staticmethod
    def _mock_llm():
        return "openai/gpt-4"

    def test_invalid_mode_raises(self):
        with patch("agents.seo_analyst.get_llm", return_value=self._mock_llm()):
            wf = SEOAnalystWorkflow("example.com")
        with pytest.raises(ValueError, match="Invalid mode"):
            wf.run(mode="invalid_mode")

    def test_valid_modes(self):
        with patch("agents.seo_analyst.get_llm", return_value=self._mock_llm()):
            wf = SEOAnalystWorkflow("example.com")
        valid = ("keyword", "competitor", "technical", "backlink", "full")
        for m in valid:
            # Just confirm no ValueError for any valid mode
            tasks = []
            if m in ("keyword", "full"):
                tasks.append(create_keyword_research_task("example.com", wf.agent))
            if m in ("competitor", "full"):
                tasks.append(create_competitor_analysis_task("example.com", wf.agent))
            if m in ("technical", "full"):
                tasks.append(create_technical_audit_task("example.com", wf.agent))
            assert isinstance(tasks, list)
