"""Tests for core infrastructure."""

import pytest
from core.config import Settings


def test_settings_defaults():
    """Settings should have sensible defaults."""
    s = Settings()
    assert s.agent_model == "gpt-4o"
    assert s.agent_temperature == 0.7
    assert s.agent_max_tokens == 4096


def test_settings_custom():
    """Settings should accept custom values."""
    import os
    os.environ["AGENT_MODEL"] = "claude-sonnet-4"
    os.environ["AGENT_TEMPERATURE"] = "0.5"
    s = Settings()
    assert s.agent_model == "claude-sonnet-4"
    assert s.agent_temperature == 0.5
    # Clean up
    del os.environ["AGENT_MODEL"]
    del os.environ["AGENT_TEMPERATURE"]


@pytest.mark.asyncio
async def test_web_search():
    """Web search should return results."""
    from core.tools.web_search import web_search
    result = await web_search("Python programming", limit=3)
    assert "results" in result
    assert isinstance(result["results"], list)


def test_file_tools():
    """File tools should read/write JSON and CSV."""
    import tempfile
    import os
    from core.tools.file_tools import write_json, read_json, write_csv, read_csv

    with tempfile.TemporaryDirectory() as tmpdir:
        # JSON
        json_path = os.path.join(tmpdir, "test.json")
        write_json(json_path, {"key": "value", "num": 42})
        data = read_json(json_path)
        assert data["key"] == "value"
        assert data["num"] == 42

        # CSV
        csv_path = os.path.join(tmpdir, "test.csv")
        rows = [{"name": "Alice", "score": "95"}, {"name": "Bob", "score": "87"}]
        write_csv(csv_path, rows)
        result = read_csv(csv_path)
        assert len(result) == 2
        assert result[0]["name"] == "Alice"
