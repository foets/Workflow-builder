"""
Workflow API Server (JSON-first)

This server provides REST endpoints for the web UI to list and load workflows.
Workflows are stored as JSON under the repo-local workflows/ directory (demo default)
and rendered to Markdown for display purposes.

Endpoints:
- GET    /workflows            - List workflows (summary)
- GET    /workflows/{id}       - Get workflow JSON + rendered markdown
- PUT    /workflows/{id}       - Update workflow JSON (validated) + return rendered markdown
- DELETE /workflows/{id}       - Delete workflow
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any
import uvicorn

from schemas import Workflow
from workflow_storage import WORKFLOWS_DIR, delete_workflow, list_workflows, load_workflow, save_workflow


def render_workflow_markdown(workflow: Workflow) -> str:
    # Keep headings consistent with the UI's step counter (## Step N:)
    lines: list[str] = []
    lines.append(f"# {workflow.name}")
    lines.append("")
    lines.append(f"> {workflow.description}")
    lines.append("")
    lines.append("## Required Connections")
    if workflow.required_toolkits:
        for t in workflow.required_toolkits:
            lines.append(f"- {t}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("---")
    lines.append("")
    for s in workflow.steps:
        lines.append(f"## Step {s.order}: {s.name}")
        lines.append("")
        lines.append(f"- **Tool**: `{s.tool}`")
        lines.append(f"- **Toolkit**: `{s.toolkit}`")
        lines.append(f"- **Status**: `{s.status}`")
        lines.append("")
        lines.append("### Description")
        lines.append(s.description)
        lines.append("")
        lines.append("### Instructions")
        lines.append("```")
        lines.append(s.instructions)
        lines.append("```")
        lines.append("")
        lines.append("### Parameters")
        if s.tool_params:
            import json as _json
            lines.append("```json")
            lines.append(_json.dumps(s.tool_params, indent=2))
            lines.append("```")
        else:
            lines.append("_None_")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).strip() + "\n"

app = FastAPI(
    title="Workflow Files API",
    description="REST API for managing workflow Markdown files",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Models
# ============================================================================

class WorkflowSummary(BaseModel):
    id: str
    name: str
    description: str
    step_count: int
    updated_at: str
    status: str
    required_toolkits: List[str] = []


class WorkflowContent(BaseModel):
    id: str
    name: str
    description: str
    workflow: dict[str, Any]
    markdown: str


class WorkflowUpdateRequest(BaseModel):
    workflow: dict[str, Any]


# ============================================================================
# Routes
# ============================================================================

@app.get("/")
async def root():
    """Health check and info."""
    return {
        "service": "Workflow Files API",
        "storage": WORKFLOWS_DIR,
        "status": "running"
    }


@app.get("/workflows", response_model=List[WorkflowSummary])
async def get_all_workflows():
    """List all saved workflows (summary)."""
    return list_workflows()

@app.delete("/workflows")
async def delete_all_workflows(dry_run: bool = False):
    """
    Delete ALL saved workflows (JSON files) from storage.

    Query params:
      - dry_run: if True, returns what would be deleted without deleting.
    """
    # We only manage JSON-first workflows here.
    paths = sorted(WORKFLOWS_DIR.glob("wf_*.json"))
    workflow_ids = [p.stem for p in paths]

    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "storage": str(WORKFLOWS_DIR),
            "count": len(workflow_ids),
            "workflow_ids": workflow_ids[:200],
        }

    deleted = 0
    errors: list[str] = []
    for p in paths:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            errors.append(f"{p.name}:{type(e).__name__}")

    return {
        "success": True,
        "dry_run": False,
        "storage": str(WORKFLOWS_DIR),
        "deleted": deleted,
        "errors": errors[:200],
    }


@app.get("/workflows/{workflow_id}", response_model=WorkflowContent)
async def get_workflow_by_id(workflow_id: str):
    """Get a workflow by ID (JSON + rendered markdown)."""
    wf = load_workflow(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    return {
        "id": wf.id,
        "name": wf.name,
        "description": wf.description,
        "workflow": wf.model_dump(mode="json"),
        "markdown": render_workflow_markdown(wf),
    }


@app.put("/workflows/{workflow_id}", response_model=WorkflowContent)
async def update_workflow_by_id(workflow_id: str, payload: WorkflowUpdateRequest):
    """
    Replace a workflow JSON by ID (validated by schema), persist it, and return updated JSON + markdown.

    Note: This is a full replace (not a patch). The UI edits the JSON source of truth.
    """
    try:
        wf = Workflow.model_validate(payload.workflow)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid workflow JSON: {type(e).__name__}")

    if wf.id != workflow_id:
        raise HTTPException(status_code=400, detail="Workflow id mismatch (body.id must match URL id)")

    wf = save_workflow(wf)
    return {
        "id": wf.id,
        "name": wf.name,
        "description": wf.description,
        "workflow": wf.model_dump(mode="json"),
        "markdown": render_workflow_markdown(wf),
    }


@app.delete("/workflows/{workflow_id}")
async def delete_workflow_by_id(workflow_id: str):
    """Delete a workflow JSON by ID."""
    if delete_workflow(workflow_id):
        return {"success": True, "message": f"Workflow '{workflow_id}' deleted"}
    raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"Starting Workflow Files API...")
    print(f"Workflows stored in: {WORKFLOWS_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=2026)








