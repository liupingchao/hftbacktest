from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT / ".workflow/workflow-kit/validate_research_package_task.py"
)
SPEC = importlib.util.spec_from_file_location("workflow_validator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)
TrustKernelError = validator.TrustKernelError
TASK = ROOT / ".workflow/tasks/0820T001.md"
MATRIX = ROOT / ".workflow/contracts/0820T001-surface-matrix.json"


def _write_matrix(tmp_path: Path, mutation) -> Path:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    mutation(matrix)
    target = tmp_path / "matrix.json"
    target.write_text(
        json.dumps(matrix, indent=2) + "\n",
        encoding="ascii",
    )
    return target


def test_current_bootstrap_task_passes_gate_zero():
    result = validator.validate_task(TASK, MATRIX)
    assert result["verified"] is True
    assert result["classification"] == "research_package_infrastructure"
    assert result["surface_count"] == 7
    assert result["exit_criterion_count"] == 7


def test_missing_surface_fails_closed(tmp_path):
    target = _write_matrix(
        tmp_path,
        lambda matrix: matrix["surfaces"].pop(),
    )
    with pytest.raises(TrustKernelError) as caught:
        validator.validate_task(TASK, target)
    assert caught.value.code == "SURFACE_MATRIX_MARKDOWN_MISMATCH"


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            lambda matrix: matrix["surfaces"][0].update(
                authoritative_sources=[]
            ),
            "JSON_SCHEMA_MIN_ITEMS",
        ),
        (
            lambda matrix: matrix["surfaces"][0].update(
                negative_mutations=[]
            ),
            "JSON_SCHEMA_MIN_ITEMS",
        ),
        (
            lambda matrix: matrix["surfaces"][0].pop("identity_layer"),
            "JSON_SCHEMA_REQUIRED_MISSING",
        ),
        (
            lambda matrix: matrix["surfaces"][0].update(
                description="TBD"
            ),
            "SURFACE_MATRIX_PLACEHOLDER",
        ),
    ],
)
def test_incomplete_research_matrix_fails_closed(
    tmp_path,
    mutation,
    expected_code,
):
    target = _write_matrix(tmp_path, mutation)
    with pytest.raises(TrustKernelError) as caught:
        validator.validate_task(TASK, target)
    assert caught.value.code == expected_code


def test_general_task_passes_without_matrix(tmp_path):
    task = tmp_path / "general.md"
    task.write_text(
        """# task

任务ID：
- 0820T099

task_type：
- `general`

produces_research_package：
- `false`
""",
        encoding="utf-8",
    )
    assert validator.validate_task(task, None)["classification"] == "general"


def test_historical_task_without_classification_is_compatible():
    historical = ROOT / ".workflow/tasks/0815T003.md"
    result = validator.validate_task(historical, None)
    assert result["classification"] == "historical_compatible"


def test_dependency_cycle_fails_closed(tmp_path):
    def mutation(matrix):
        mutated = copy.deepcopy(matrix)
        mutated["surfaces"][0]["depends_on_surfaces"] = [
            "archive_and_cleanup_envelope"
        ]
        matrix.clear()
        matrix.update(mutated)

    target = _write_matrix(tmp_path, mutation)
    with pytest.raises(TrustKernelError) as caught:
        validator.validate_task(TASK, target)
    assert caught.value.code == "SURFACE_DEPENDENCY_CYCLE"
