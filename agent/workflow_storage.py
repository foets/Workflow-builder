"""
Workflow JSON Storage (Source of Truth)

- Workflows are stored as JSON files under the repo-local ./workflows/ directory (demo default)
- Workflow steps are schema-validated via Pydantic models in schemas.py
- Markdown is a rendered VIEW only
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from schemas import Workflow, WorkflowDraft, WorkflowStep, WorkflowStepDraft


# Storage directory (repo-local by default for this demo).
# You can override via env var WORKFLOW_STORAGE_DIR.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_WORKFLOWS_DIR = _PROJECT_ROOT / "workflows"
_LEGACY_WORKFLOWS_DIR = Path(os.path.expanduser("~/.workflow-builder"))
WORKFLOWS_DIR = Path(os.environ.get("WORKFLOW_STORAGE_DIR", str(_DEFAULT_WORKFLOWS_DIR))).expanduser()


def ensure_workflows_dir() -> None:
    WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)

    # One-time best-effort migration: copy legacy wf_*.json into the new dir.
    # IMPORTANT: this MUST NOT continuously re-create deleted workflows.
    # We create a marker file after the first attempt.
    marker = WORKFLOWS_DIR / ".migrated_from_home"
    try:
        if marker.exists():
            return
        if WORKFLOWS_DIR != _LEGACY_WORKFLOWS_DIR:
            legacy_files = sorted(_LEGACY_WORKFLOWS_DIR.glob("wf_*.json")) if _LEGACY_WORKFLOWS_DIR.exists() else []
            if legacy_files:
                for src in legacy_files:
                    dst = WORKFLOWS_DIR / src.name
                    if dst.exists():
                        continue
                    try:
                        dst.write_bytes(src.read_bytes())
                    except Exception:
                        # ignore individual copy failures
                        pass
        # Mark migration attempted (even if nothing to copy) so deletes won't be undone.
        try:
            marker.write_text("ok\n", encoding="utf-8")
        except Exception:
            pass
    except Exception:
        # Never fail app startup due to migration logic.
        pass


def _workflow_path(workflow_id: str) -> Path:
    ensure_workflows_dir()
    return WORKFLOWS_DIR / f"{workflow_id}.json"


def _new_workflow_id() -> str:
    # Stable, sortable, collision-resistant
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"wf_{ts}_{suffix}"


def create_workflow_from_draft(draft: WorkflowDraft) -> Workflow:
    """Create a full Workflow with IDs/orders from a draft payload."""
    now = datetime.now(timezone.utc)
    workflow_id = _new_workflow_id()

    steps: list[WorkflowStep] = []
    for i, s in enumerate(draft.steps, start=1):
        step = WorkflowStep(
            id=f"step_{i}",
            order=i,
            name=s.name,
            description=s.description,
            instructions=s.instructions,
            tool=s.tool,
            tool_params=s.tool_params,
            toolkit=s.toolkit,
            inputs=s.inputs,
            outputs=s.outputs,
            status="pending",
        )
        steps.append(step)

    return Workflow(
        id=workflow_id,
        name=draft.name,
        description=draft.description,
        created_at=now,
        updated_at=now,
        status="draft",
        steps=steps,
    )


def save_workflow(workflow: Workflow) -> Workflow:
    """Persist workflow JSON to disk (updates updated_at)."""
    ensure_workflows_dir()
    now = datetime.now(timezone.utc)
    updated = workflow.model_copy(deep=True)
    updated.updated_at = now
    # Re-validate to re-derive any computed fields (e.g., requires_connection) after mutations.
    updated = Workflow.model_validate(updated.model_dump(mode="json"))
    updated.updated_at = now

    path = _workflow_path(updated.id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(updated.model_dump(mode="json"), f, indent=2)
    return updated


def load_workflow(workflow_id: str) -> Optional[Workflow]:
    """Load workflow by ID."""
    path = _workflow_path(workflow_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Workflow.model_validate(data)
    except Exception:
        return None


def list_workflows() -> list[dict[str, Any]]:
    """List saved workflow JSON files (summary only)."""
    ensure_workflows_dir()
    results: list[dict[str, Any]] = []
    for p in WORKFLOWS_DIR.glob("wf_*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            wf = Workflow.model_validate(data)
            results.append(
                {
                    "id": wf.id,
                    "name": wf.name,
                    "description": wf.description,
                    "status": wf.status,
                    "created_at": wf.created_at.isoformat(),
                    "updated_at": wf.updated_at.isoformat(),
                    "step_count": len(wf.steps),
                    "required_toolkits": list(wf.required_toolkits),
                }
            )
        except Exception:
            # ignore invalid/partial files
            continue
    # newest first
    results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return results


def delete_workflow(workflow_id: str) -> bool:
    """Delete workflow JSON by ID."""
    path = _workflow_path(workflow_id)
    if path.exists():
        path.unlink()
        return True
    return False


def update_step_status(workflow_id: str, step_id: str, status: str) -> Optional[Workflow]:
    wf = load_workflow(workflow_id)
    if not wf:
        return None
    updated = wf.model_copy(deep=True)
    for step in updated.steps:
        if step.id == step_id:
            step.status = status  # type: ignore[assignment]
            break
    return save_workflow(updated)


def update_workflow_status(workflow_id: str, status: str) -> Optional[Workflow]:
    wf = load_workflow(workflow_id)
    if not wf:
        return None
    updated = wf.model_copy(deep=True)
    updated.status = status  # type: ignore[assignment]
    return save_workflow(updated)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "WORKFLOWS_DIR",
    "ensure_workflows_dir",
    "create_workflow_from_draft",
    "save_workflow",
    "load_workflow",
    "list_workflows",
    "delete_workflow",
    "update_step_status",
    "update_workflow_status",
]








