"""GUARDRAIL (spec 3.2): no engine SDK imported outside backend/mneme/llm/.

Route every model call through LLMClient. If any module elsewhere imports
openai, vllm, or ollama directly, that is a defect and this test fails.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = [ROOT / "backend", ROOT / "ingestion"]
LLM_PACKAGE = ROOT / "backend" / "mneme" / "llm"
FORBIDDEN = re.compile(r"^\s*(?:import|from)\s+(openai|vllm|ollama)\b", re.MULTILINE)


def test_no_engine_sdk_imported_outside_llm_package():
    offenders = []
    for source_dir in SOURCE_DIRS:
        for path in source_dir.rglob("*.py"):
            if LLM_PACKAGE in path.parents:
                continue
            if FORBIDDEN.search(path.read_text()):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"engine SDK imported outside llm/: {offenders}"
