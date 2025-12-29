"""
LangChain tools for the Workflow Builder.

Important separation:
- BUILD tools: create/update/get/list/delete workflows (JSON-first)
- Execution tools (Composio meta-tools) are provided separately per mode in agent.py
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from langchain_core.tools import tool

from schemas import Workflow, WorkflowDraft, WorkflowStep, WorkflowStepDraft
from workflow_storage import (
    create_workflow_from_draft,
    delete_workflow as storage_delete_workflow,
    list_workflows as storage_list_workflows,
    load_workflow,
    save_workflow,
)


def _render_workflow_markdown(workflow: Workflow) -> str:
    """Render workflow to human-friendly Markdown for UI display (view-only)."""
    lines: list[str] = []
    lines.append(f"# {workflow.name}")
    lines.append("")
    lines.append(f"> {workflow.description}")
    lines.append("")

    # Toolkits / connections (stable order already derived in schema)
    lines.append("## Required Connections")
    if workflow.required_toolkits:
        for t in workflow.required_toolkits:
            lines.append(f"- {t}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("---")
    lines.append("")

    for step in workflow.steps:
        lines.append(f"## Step {step.order}: {step.name}")
        lines.append("")
        lines.append(f"- **Tool**: `{step.tool}`")
        lines.append(f"- **Toolkit**: `{step.toolkit}`")
        lines.append(f"- **Status**: `{step.status}`")
        lines.append("")
        lines.append("### Description")
        lines.append(step.description)
        lines.append("")
        lines.append("### Instructions")
        lines.append("```")
        lines.append(step.instructions)
        lines.append("```")
        lines.append("")
        lines.append("### Parameters")
        if step.tool_params:
            lines.append("```json")
            lines.append(json.dumps(step.tool_params, indent=2))
            lines.append("```")
        else:
            lines.append("_None_")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Trim trailing separators
    while lines and lines[-1] == "":
        lines.pop()
    if lines and lines[-1] == "---":
        lines.pop()
    return "\n".join(lines).strip() + "\n"


def _workflow_tool_payload(workflow: Workflow) -> str:
    """Tool payload returned to the agent graph for deterministic state hydration."""
    payload = {
        "type": "workflow",
        "workflow_id": workflow.id,
        "workflow": workflow.model_dump(mode="json"),
        "workflow_markdown": _render_workflow_markdown(workflow),
    }
    return json.dumps(payload)


_OUTPUT_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_STEP_REF_RE = re.compile(r"\{\{\s*steps\.(\d+)\.([A-Za-z_][A-Za-z0-9_]*)", flags=re.IGNORECASE)


def _iter_strings(obj: Any):
    """Yield all string leaf values from a JSON-ish structure."""
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, list):
        for it in obj:
            yield from _iter_strings(it)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)


def _slugify_key(s: str) -> str:
    """Best-effort make a short snake_case identifier key."""
    s = (s or "").strip()
    if not s:
        return "output"
    # Replace non-alnum with underscores and collapse.
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "output"
    # Ensure doesn't start with a digit.
    if s[0].isdigit():
        s = f"out_{s}"
    # Keep reasonably short
    return s[:80]


def _is_valid_output_key(v: Any) -> bool:
    return isinstance(v, str) and (1 <= len(v) <= 80) and bool(_OUTPUT_KEY_RE.fullmatch(v))


def _infer_outputs_key_from_refs(steps: list[dict], step_index: int) -> Optional[str]:
    """Infer an outputs key by scanning later steps' templates like {{steps.<i>.<key>...}}."""
    if step_index < 0:
        return None
    for j in range(step_index + 1, len(steps)):
        step = steps[j]
        if not isinstance(step, dict):
            continue
        tool_params = step.get("tool_params")
        for s in _iter_strings(tool_params):
            for m in _STEP_REF_RE.finditer(s):
                try:
                    idx = int(m.group(1))
                except Exception:
                    continue
                if idx != step_index:
                    continue
                key = m.group(2)
                if _is_valid_output_key(key):
                    return key
    return None


def _normalize_step_outputs(steps: list[dict]) -> list[dict]:
    """
    Normalize each step's `outputs` to be a short identifier key (<=80 chars).

    This prevents schema validation failures and makes templating stable.
    """
    out: list[dict] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        step = dict(step)

        raw = step.get("outputs")
        if _is_valid_output_key(raw):
            out.append(step)
            continue

        inferred = _infer_outputs_key_from_refs(steps, i)
        if _is_valid_output_key(inferred):
            step["outputs"] = inferred
            out.append(step)
            continue

        # Fall back to a slug based on tool/name.
        base = step.get("outputs") if isinstance(step.get("outputs"), str) else ""
        if not base:
            base = step.get("tool") if isinstance(step.get("tool"), str) else ""
        if not base:
            base = step.get("name") if isinstance(step.get("name"), str) else ""

        candidate = _slugify_key(str(base))
        if not _is_valid_output_key(candidate):
            candidate = "output"
        step["outputs"] = candidate
        out.append(step)
    return out


@tool
def discover_tools(query: str = "") -> str:
    """
    Explain available tool categories and how they map to workflow steps.

    Use this in BUILD mode when you need to:
    - Explain what tools exist
    - Help the user pick the right tool(s)
    - Provide parameter examples

    NOTE: This tool is informational only. It does NOT call external services.
    """
    # Keep this concise: it's for chat, and the UI will render Markdown.
    return (
        "## Tool Catalog\n\n"
        "Use these abstract tools in workflow steps (resolved at execution time):\n\n"
        "### Email (Gmail)\n"
        "- `GMAIL_GET_MESSAGES` — fetch messages (use `toolkit: gmail`)\n"
        "- `GMAIL_SEND_EMAIL` — send an email (use `toolkit: gmail`)\n\n"
        "### Documents (Google Docs)\n"
        "- `GOOGLEDOCS_CREATE_DOCUMENT` — create a document (use `toolkit: googledocs`)\n"
        "- `GOOGLEDOCS_UPDATE_DOCUMENT` — write content (use `toolkit: googledocs`)\n\n"
        "### Files (Google Drive)\n"
        "- `DRIVE_*` tools — folders/files/watch/upload (use `toolkit: googledrive`)\n\n"
        "### Messaging\n"
        "- `SLACK_SEND_MESSAGE` — post a message (use `toolkit: slack`)\n\n"
        "### AI Processing (No Auth)\n"
        "- `AI_PROCESS` — summarize/analyze/generate (use `toolkit: internal`)\n\n"
        f"Search hint: {query or 'None'}\n"
    )


@tool
def create_workflow(name: str, description: str, steps: list[dict]) -> str:
    """
    Create a new workflow (JSON-first) and persist it.

    Args:
        name: Workflow name (human readable)
        description: 1-2 sentence summary
        steps: Array of step objects matching the workflow schema. Each step MUST include:
            - name, description, instructions
            - tool (UPPERCASE_WITH_UNDERSCORES)
            - tool_params (dict)
            - toolkit (lowercase, or internal)
            - inputs (list[str])
            - outputs (str)

    Returns:
        JSON string payload containing:
        - workflow_id
        - workflow (object)
        - workflow_markdown (string)
    """
    steps = _normalize_step_outputs(steps if isinstance(steps, list) else [])
    draft = WorkflowDraft.model_validate(
        {
            "name": name,
            "description": description,
            "steps": steps,
        }
    )
    workflow = create_workflow_from_draft(draft)
    workflow = save_workflow(workflow)
    return _workflow_tool_payload(workflow)


@tool
def get_workflow(workflow_id: str) -> str:
    """
    Load a workflow by ID.

    Returns the workflow JSON + rendered Markdown for the UI.
    """
    wf = load_workflow(workflow_id)
    if not wf:
        return json.dumps({"type": "error", "error": f"Workflow '{workflow_id}' not found"})
    return _workflow_tool_payload(wf)


@tool
def list_workflows() -> str:
    """
    List saved workflows (summary).

    Returns JSON string list for deterministic parsing in the graph/UI.
    """
    return json.dumps({"type": "workflow_list", "workflows": storage_list_workflows()})


@tool
def delete_workflow(workflow_id: str) -> str:
    """Delete a workflow by ID."""
    ok = storage_delete_workflow(workflow_id)
    if ok:
        return json.dumps({"type": "deleted", "workflow_id": workflow_id})
    return json.dumps({"type": "error", "error": f"Workflow '{workflow_id}' not found"})


@tool
def update_workflow(
    workflow_id: str,
    action: str,
    step_index: Optional[int] = None,
    step_data: Optional[dict] = None,
    field: Optional[str] = None,
    value: Optional[Any] = None,
) -> str:
    """
    Modify an existing workflow.

    Args:
        workflow_id: The workflow ID to update.
        action: One of:
            - update_step
            - add_step
            - remove_step
            - update_metadata
        step_index: 0-based step index (required for update_step/remove_step)
        step_data: full step object (required for add_step)
        field: field name for update_step (name, description, instructions, tool, tool_params, toolkit, inputs, outputs)
        value: new value for update_step/update_metadata
    """
    wf = load_workflow(workflow_id)
    if not wf:
        return json.dumps({"type": "error", "error": f"Workflow '{workflow_id}' not found"})

    updated = wf.model_copy(deep=True)

    if action == "update_metadata":
        if field not in {"name", "description"}:
            return json.dumps({"type": "error", "error": "update_metadata requires field in {name, description}"})
        if field == "name":
            updated.name = str(value or "")
        if field == "description":
            updated.description = str(value or "")

    elif action == "update_step":
        if step_index is None or field is None:
            return json.dumps({"type": "error", "error": "update_step requires step_index and field"})
        if step_index < 0 or step_index >= len(updated.steps):
            return json.dumps({"type": "error", "error": f"Invalid step_index {step_index}"})

        step = updated.steps[step_index]
        if field == "name":
            step.name = str(value or "")
        elif field == "description":
            step.description = str(value or "")
        elif field == "instructions":
            step.instructions = str(value or "")
        elif field == "tool":
            step.tool = str(value or "")
        elif field == "tool_params":
            step.tool_params = value if isinstance(value, dict) else {}
        elif field == "toolkit":
            step.toolkit = str(value or "")
        elif field == "inputs":
            step.inputs = value if isinstance(value, list) else []
        elif field == "outputs":
            step.outputs = str(value or "")
        else:
            return json.dumps({"type": "error", "error": f"Unknown step field '{field}'"})

    elif action == "add_step":
        if not step_data:
            return json.dumps({"type": "error", "error": "add_step requires step_data"})
        # Validate step draft and append
        normalized = _normalize_step_outputs([step_data] if isinstance(step_data, dict) else [])
        if not normalized:
            return json.dumps({"type": "error", "error": "Invalid step_data"})
        draft = WorkflowStepDraft.model_validate(normalized[0])
        next_order = len(updated.steps) + 1
        updated.steps.append(
            WorkflowStep(
                id=f"step_{next_order}",
                order=next_order,
                name=draft.name,
                description=draft.description,
                instructions=draft.instructions,
                tool=draft.tool,
                tool_params=draft.tool_params,
                toolkit=draft.toolkit,
                inputs=draft.inputs,
                outputs=draft.outputs,
                status="pending",
            )
        )

    elif action == "remove_step":
        if step_index is None:
            return json.dumps({"type": "error", "error": "remove_step requires step_index"})
        if step_index < 0 or step_index >= len(updated.steps):
            return json.dumps({"type": "error", "error": f"Invalid step_index {step_index}"})
        updated.steps.pop(step_index)
        # Re-number steps
        for i, s in enumerate(updated.steps, start=1):
            s.order = i
            s.id = f"step_{i}"

    else:
        return json.dumps({"type": "error", "error": f"Unknown action '{action}'"})

    updated = save_workflow(updated)
    return _workflow_tool_payload(updated)


__all__ = [
    "discover_tools",
    "create_workflow",
    "get_workflow",
    "list_workflows",
    "delete_workflow",
    "update_workflow",
]


