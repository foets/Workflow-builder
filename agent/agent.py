"""
Workflow Builder Agent (v2)

Three explicit modes, selected by the frontend:
- build: conversational, schema-first workflow construction (NO Composio tools)
- preflight: check required connections via Tool Router (COMPOSIO_MANAGE_CONNECTIONS)
- execute: execute workflow step-by-step via Tool Router (search + multi_execute) and AI_PROCESS

Notes:
- Workflow source-of-truth is JSON stored under ~/.workflow-builder/*.json
- Markdown is rendered for the UI as a view-only representation
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Annotated, Any, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from composio_integration import COMPOSIO_AVAILABLE, create_session, get_tools as get_composio_tools, is_configured
from prompts import (
    BUILD_SYSTEM_PROMPT,
    CONFIGURE_SYSTEM_PROMPT_TEMPLATE,
    CONNECTIONS_SYSTEM_PROMPT_TEMPLATE,
    EXECUTE_SYSTEM_PROMPT_TEMPLATE,
)
from schemas import Workflow
from tools import (
    create_workflow,
    delete_workflow,
    discover_tools,
    get_workflow,
    list_workflows,
    update_workflow,
)
from workflow_storage import load_workflow, save_workflow


# ---------------------------------------------------------------------------
# Debug logging (NDJSON) - DEBUG MODE instrumentation
# ---------------------------------------------------------------------------

_DEBUG_LOG_PATH = "/Users/kostyantinkolisnyk/Desktop/workflow-builder-demo/.cursor/debug.log"
_DEBUG_SESSION_ID = "debug-session"
_DEBUG_RUN_ID = "run1"


def _dbg_log(hypothesis_id: str, location: str, message: str, data: Optional[dict[str, Any]] = None) -> None:
    try:
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "runId": _DEBUG_RUN_ID,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Never fail agent execution due to debug logging.
        pass


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    mode: Literal["build", "preflight", "configure", "execute"]

    # Current workflow context
    current_workflow_id: Optional[str]
    workflow: Optional[dict]  # Workflow JSON object
    workflow_markdown: Optional[str]  # Rendered view for UI

    # Connections (auth check)
    auth_status: Optional[dict]  # { toolkit: { connected, connect_url, label } , ... }

    # Configure (build-time auth + resource selection)
    tool_router_session_id: Optional[str]
    missing_config_keys: Optional[list[str]]

    # Execute
    execution_cursor: int
    step_results: dict  # { step_id: any }

    error: Optional[str]


def _ensure_defaults(state: AgentState) -> AgentState:
    if "mode" not in state or state["mode"] not in {"build", "preflight", "configure", "execute"}:
        state["mode"] = "build"
    state.setdefault("execution_cursor", 0)
    state.setdefault("step_results", {})
    return state


# ---------------------------------------------------------------------------
# Helpers (parsing tool outputs, rendering, etc.)
# ---------------------------------------------------------------------------


def _tool_content_to_text(content: Any) -> str:
    # Composio tools often return a list of {"type": "text", "text": "...json..."} parts
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content)
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if "text" in p and isinstance(p["text"], str):
                    parts.append(p["text"])
                elif "content" in p and isinstance(p["content"], str):
                    parts.append(p["content"])
                else:
                    parts.append(json.dumps(p))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(content)


def _last_tool_message(messages: list[AnyMessage]) -> Optional[ToolMessage]:
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            return m
    return None


def _render_workflow_markdown(workflow: Workflow) -> str:
    # Keep markdown stable and easy to count steps.
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
            lines.append("```json")
            lines.append(json.dumps(s.tool_params, indent=2))
            lines.append("```")
        else:
            lines.append("_None_")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _toolkit_label(toolkit: str) -> str:
    labels = {
        "gmail": "Gmail",
        "googledocs": "Google Docs",
        "googlesheets": "Google Sheets",
        "googledrive": "Google Drive",
        "drive": "Google Drive",
        "calendar": "Google Calendar",
        "slack": "Slack",
        "notion": "Notion",
        "github": "GitHub",
        "linear": "Linear",
        "jira": "Jira",
        "asana": "Asana",
        "hubspot": "HubSpot",
        "salesforce": "Salesforce",
        "internal": "Internal",
    }
    return labels.get(toolkit, toolkit)


def _extract_connect_urls(text: str) -> list[str]:
    # Composio connect links usually look like https://connect.composio.dev/link/...
    urls = re.findall(r"https?://connect\.composio\.dev/link/[^\s\"')]+", text)
    # de-dup in order
    seen = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _redact_connect_urls(text: str) -> str:
    """
    Remove/neutralize raw OAuth/connect URLs from assistant chat content.

    The UI renders connect buttons from `auth_status.toolkits[*].connect_url`, so we should not
    show raw URLs inside chat messages.
    """
    if not isinstance(text, str) or not text:
        return text
    # Replace markdown links pointing to Composio connect with just the link text.
    text = re.sub(
        r"\[([^\]]+)\]\((https?://connect\.composio\.dev/link/[^)]+)\)",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    # Remove any remaining bare connect URLs.
    text = re.sub(r"https?://connect\.composio\.dev/link/[^\s\"')]+", "", text, flags=re.IGNORECASE)
    return text


def _extract_auth_status_from_tool_output(required_toolkits: list[str], tool_text: str) -> dict:
    """
    Best-effort parsing into:
    {
      "toolkits": {
         "gmail": {"connected": True, "connect_url": None, "label": "Gmail"},
         "googledocs": {"connected": False, "connect_url": "...", "label": "Google Docs"}
      },
      "all_connected": bool
    }
    """
    # Defaults: unknown toolkits assumed disconnected until proven otherwise
    toolkits_map: dict[str, dict[str, Any]] = {
        t: {"connected": False, "connect_url": None, "label": _toolkit_label(t)} for t in required_toolkits
    }

    # Try JSON parsing first
    try:
        data = json.loads(tool_text)
    except Exception:
        data = None

    # Common structures:
    # - COMPOSIO_MANAGE_CONNECTIONS: {"successful":true,"data":{"results":{ "<toolkit>": {...} } } }
    # - COMPOSIO_SEARCH_TOOLS: data.toolkit_connection_statuses = [{toolkit, has_active_connection, ...}]
    if isinstance(data, dict):
        # COMPOSIO_MANAGE_CONNECTIONS-style: data.results.<toolkit>
        results_obj = None
        if isinstance(data.get("data"), dict) and isinstance(data["data"].get("results"), dict):
            results_obj = data["data"]["results"]
        if isinstance(results_obj, dict):
            for tk, entry in results_obj.items():
                if not isinstance(tk, str) or tk not in toolkits_map:
                    continue
                if not isinstance(entry, dict):
                    continue
                connected = bool(entry.get("has_active_connection")) or entry.get("status") == "active"
                toolkits_map[tk]["connected"] = connected
                # Surface scope / permission issues as a warning
                cui = entry.get("current_user_info")
                if isinstance(cui, dict) and isinstance(cui.get("error"), dict):
                    toolkits_map[tk]["error"] = cui["error"].get("message") or "Connection has an error"
                    # Treat scope/permission errors as NOT ready for execution.
                    toolkits_map[tk]["connected"] = False
                # Connect URL (if present)
                for k in ("redirect_url", "connect_url", "connectUrl", "connect_link", "connectLink", "url", "link"):
                    if isinstance(entry.get(k), str) and "connect.composio.dev/link/" in entry.get(k, ""):
                        toolkits_map[tk]["connect_url"] = entry.get(k)
                        break

        maybe_statuses = None
        if isinstance(data.get("data"), dict) and isinstance(data["data"].get("toolkit_connection_statuses"), list):
            maybe_statuses = data["data"]["toolkit_connection_statuses"]
        elif isinstance(data.get("toolkit_connection_statuses"), list):
            maybe_statuses = data.get("toolkit_connection_statuses")

        if isinstance(maybe_statuses, list):
            for entry in maybe_statuses:
                if not isinstance(entry, dict):
                    continue
                tk = entry.get("toolkit")
                if not isinstance(tk, str) or tk not in toolkits_map:
                    continue
                connected = bool(entry.get("has_active_connection") or entry.get("connected"))
                toolkits_map[tk]["connected"] = connected
                # Treat explicit errors as not ready
                if entry.get("error"):
                    toolkits_map[tk]["error"] = str(entry.get("error"))
                    toolkits_map[tk]["connected"] = False

    # If any connect URLs exist, assign them to any disconnected toolkits (best-effort)
    urls = _extract_connect_urls(tool_text)
    if urls:
        # If only one missing toolkit, map it directly
        missing = [t for t, s in toolkits_map.items() if not s["connected"]]
        if len(missing) == 1:
            toolkits_map[missing[0]]["connect_url"] = urls[0]
        else:
            # Otherwise, attach URLs to missing toolkits in order (best-effort).
            for i, t in enumerate(missing):
                if i < len(urls):
                    toolkits_map[t]["connect_url"] = urls[i]

    # "Ready to execute" means connected AND no recorded error.
    all_connected = all(bool(v.get("connected")) and not v.get("error") for v in toolkits_map.values()) if toolkits_map else True
    return {"all_connected": all_connected, "toolkits": toolkits_map}


# ---------------------------------------------------------------------------
# Configure helpers (workflow.config + placeholders)
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")


def _collect_placeholder_exprs(obj: Any) -> list[str]:
    """Recursively collect `{{ ... }}` placeholder expressions from a JSON-ish object."""
    out: list[str] = []
    if obj is None:
        return out
    if isinstance(obj, str):
        return [m.strip() for m in _PLACEHOLDER_RE.findall(obj)]
    if isinstance(obj, list):
        for it in obj:
            out.extend(_collect_placeholder_exprs(it))
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_collect_placeholder_exprs(v))
        return out
    return out


def _split_outputs(outputs: Any) -> set[str]:
    """Parse `step.outputs` into a set of output keys (best-effort)."""
    if not isinstance(outputs, str):
        return set()
    parts = [p.strip() for p in outputs.split(",")]
    return {p for p in parts if p}


def _is_simple_identifier(s: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", s or ""))


def _missing_config_keys(wf: dict) -> list[str]:
    """
    Determine which workflow-level config keys are required but missing, by scanning
    `steps[*].tool_params` for placeholders.

    Rules:
    - `{{config.key}}` is ALWAYS treated as workflow config key `key`.
    - `{{key}}` is treated as workflow config key IFF `key` is NOT produced by any previous step output.
    """
    if not isinstance(wf, dict):
        return []

    cfg = wf.get("config") if isinstance(wf.get("config"), dict) else {}
    steps = wf.get("steps") if isinstance(wf.get("steps"), list) else []

    # Stable order: by step.order if present, else list order.
    def _step_sort_key(s: Any, idx: int) -> tuple[int, int]:
        if isinstance(s, dict) and isinstance(s.get("order"), int):
            return (s["order"], idx)
        return (idx + 1, idx)

    required: list[str] = []
    prev_outputs: set[str] = set()

    indexed_steps = list(enumerate(steps))
    for idx, step in sorted(indexed_steps, key=lambda t: _step_sort_key(t[1], t[0])):
        if not isinstance(step, dict):
            continue

        exprs = _collect_placeholder_exprs(step.get("tool_params"))
        for expr in exprs:
            expr = (expr or "").strip()
            if expr.startswith("config."):
                key = expr[len("config.") :].strip()
                if _is_simple_identifier(key) and key not in required:
                    required.append(key)
                continue

            # Only treat simple identifiers as candidates; ignore expressions like `a || b`.
            if not _is_simple_identifier(expr):
                continue

            # If produced by previous steps, it's runtime input (not config).
            if expr in prev_outputs:
                continue

            if expr not in required:
                required.append(expr)

        prev_outputs |= _split_outputs(step.get("outputs"))

    missing = [k for k in required if k not in cfg or cfg.get(k) in (None, "")]
    return missing


def _resolve_config_placeholders(obj: Any, cfg: dict[str, Any]) -> Any:
    """
    Replace config placeholders in a JSON-ish object.

    Supported placeholder forms:
    - `{{config.key}}`
    - `{{key}}` (only if `key` exists in workflow.config)
    """
    if obj is None:
        return None

    if isinstance(obj, str):
        s = obj
        # If the entire string is exactly one placeholder, preserve original type.
        m = re.fullmatch(r"\{\{\s*([^}]+?)\s*\}\}", s)
        if m:
            expr = (m.group(1) or "").strip()
            key = None
            if expr.startswith("config."):
                key = expr[len("config.") :].strip()
            elif _is_simple_identifier(expr) and expr in cfg:
                key = expr
            if key and key in cfg and cfg.get(key) not in (None, ""):
                return cfg.get(key)
            return s

        def _repl(match: re.Match[str]) -> str:
            expr = (match.group(1) or "").strip()
            key = None
            if expr.startswith("config."):
                key = expr[len("config.") :].strip()
            elif _is_simple_identifier(expr) and expr in cfg:
                key = expr
            if not key or key not in cfg or cfg.get(key) in (None, ""):
                return match.group(0)
            v = cfg.get(key)
            if isinstance(v, (dict, list)):
                return json.dumps(v)
            return str(v)

        return _PLACEHOLDER_RE.sub(_repl, s)

    if isinstance(obj, list):
        return [_resolve_config_placeholders(v, cfg) for v in obj]

    if isinstance(obj, dict):
        return {k: _resolve_config_placeholders(v, cfg) for k, v in obj.items()}

    return obj


# ---------------------------------------------------------------------------
# Composio tool initialization (meta tools)
# ---------------------------------------------------------------------------


def _init_composio_tools_sync() -> list[BaseTool]:
    if not (COMPOSIO_AVAILABLE and is_configured()):
        return []
    try:
        prev_loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            prev_loop = asyncio.get_event_loop()
        except Exception:
            prev_loop = None

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(create_session())
        tools = loop.run_until_complete(get_composio_tools())
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        finally:
            loop.close()
            # IMPORTANT: restore previous loop (or clear) so we don't leave a CLOSED loop as current.
            try:
                if prev_loop is not None and not prev_loop.is_closed():
                    asyncio.set_event_loop(prev_loop)
                else:
                    asyncio.set_event_loop(None)
            except Exception:
                pass

        return tools or []
    except Exception as e:
        print(f"⚠ Composio init error: {e}")
        return []


_composio_tools: list[BaseTool] = _init_composio_tools_sync()
_composio_by_name = {t.name: t for t in _composio_tools if getattr(t, "name", None)}

PREFLIGHT_TOOLS: list[BaseTool] = [t for n, t in _composio_by_name.items() if n == "COMPOSIO_MANAGE_CONNECTIONS"]
EXECUTE_TOOLS: list[BaseTool] = [
    t
    for n, t in _composio_by_name.items()
    if n in {"COMPOSIO_SEARCH_TOOLS", "COMPOSIO_MULTI_EXECUTE_TOOL", "COMPOSIO_GET_TOOL_SCHEMAS"}
]

# region agent log
_dbg_loop_state = "unknown"
try:
    _dbg_loop = asyncio.get_event_loop()
    _dbg_loop_state = "closed" if _dbg_loop.is_closed() else "open"
except Exception as _e:
    _dbg_loop_state = f"no_loop:{type(_e).__name__}"
_dbg_log(
    "H1",
    "agent.py:composio_init",
    "tools_initialized",
    {
        "COMPOSIO_AVAILABLE": bool(COMPOSIO_AVAILABLE),
        "PREFLIGHT_TOOLS_count": len(PREFLIGHT_TOOLS),
        "EXECUTE_TOOLS_count": len(EXECUTE_TOOLS),
        "event_loop_state_after_init": _dbg_loop_state,
    },
)
# endregion agent log


# ---------------------------------------------------------------------------
# LLMs (separate per mode)
# ---------------------------------------------------------------------------

def _env_model(name: str, default: str) -> str:
    v = os.getenv(name)
    v = v.strip() if isinstance(v, str) else ""
    return v or default


# Defaults: keep behavior identical unless env vars are set.
_DEFAULT_MODEL = _env_model("DEFAULT_LLM_MODEL", "gpt-5.2")
_BUILD_MODEL = _env_model("BUILD_LLM_MODEL", "gpt-5.2")
_PREFLIGHT_MODEL = _env_model("PREFLIGHT_LLM_MODEL", "gpt-5-nano")
_CONFIGURE_MODEL = _env_model("CONFIGURE_LLM_MODEL", _PREFLIGHT_MODEL)
_EXECUTE_TOOL_MODEL = _env_model("EXECUTE_TOOL_LLM_MODEL", "gpt-5-mini")
_EXECUTE_INTERNAL_MODEL = _env_model("EXECUTE_INTERNAL_LLM_MODEL", "gpt-5-mini")


build_llm = ChatOpenAI(model=_BUILD_MODEL)
preflight_llm = ChatOpenAI(model=_PREFLIGHT_MODEL)
configure_llm = ChatOpenAI(model=_CONFIGURE_MODEL)
execute_llm = ChatOpenAI(model=_EXECUTE_TOOL_MODEL)
execute_llm_internal = ChatOpenAI(model=_EXECUTE_INTERNAL_MODEL)

_dbg_log(
    "H1",
    "agent.py:llm_init",
    "models",
    {
        "DEFAULT_LLM_MODEL": _DEFAULT_MODEL,
        "BUILD_LLM_MODEL": _BUILD_MODEL,
        "PREFLIGHT_LLM_MODEL": _PREFLIGHT_MODEL,
        "CONFIGURE_LLM_MODEL": _CONFIGURE_MODEL,
        "EXECUTE_TOOL_LLM_MODEL": _EXECUTE_TOOL_MODEL,
        "EXECUTE_INTERNAL_LLM_MODEL": _EXECUTE_INTERNAL_MODEL,
    },
)


BUILD_TOOLS: list[BaseTool] = [
    discover_tools,
    create_workflow,
    update_workflow,
    get_workflow,
    list_workflows,
    delete_workflow,
]


@tool
def set_workflow_config(workflow_id: str, key: str, value: Any) -> str:
    """
    Persist a workflow configuration value to `workflow.config`.

    Use this ONLY in Configure mode after the user has selected/provided a value.
    Returns a structured payload for deterministic state hydration.
    """
    wf = load_workflow(workflow_id)
    if not wf:
        return json.dumps({"type": "error", "error": f"Workflow '{workflow_id}' not found"})

    try:
        if not isinstance(wf.config, dict):
            wf.config = {}  # type: ignore[assignment]
        wf.config[str(key)] = value
        saved = save_workflow(wf)
        payload = {
            "type": "workflow",
            "workflow_id": saved.id,
            "workflow": saved.model_dump(mode="json"),
            "workflow_markdown": _render_workflow_markdown(saved),
        }
        return json.dumps(payload)
    except Exception as e:
        return json.dumps({"type": "error", "error": f"Failed to save config: {type(e).__name__}"})


CONFIGURE_TOOLS: list[BaseTool] = [set_workflow_config]
for _n in (
    "COMPOSIO_MANAGE_CONNECTIONS",
    "COMPOSIO_SEARCH_TOOLS",
    "COMPOSIO_GET_TOOL_SCHEMAS",
    "COMPOSIO_MULTI_EXECUTE_TOOL",
):
    _t = _composio_by_name.get(_n)
    if _t is not None:
        CONFIGURE_TOOLS.append(_t)

build_llm_with_tools = build_llm.bind_tools(BUILD_TOOLS)
preflight_llm_with_tools = preflight_llm.bind_tools(PREFLIGHT_TOOLS) if PREFLIGHT_TOOLS else preflight_llm
configure_llm_with_tools = configure_llm.bind_tools(CONFIGURE_TOOLS) if CONFIGURE_TOOLS else configure_llm
execute_llm_with_tools = execute_llm.bind_tools(EXECUTE_TOOLS) if EXECUTE_TOOLS else execute_llm
 # execute_llm_internal intentionally has NO tool binding (hard guarantee: cannot call Tool Router meta-tools)


# ---------------------------------------------------------------------------
# Router / shared nodes
# ---------------------------------------------------------------------------

def _messages_for_llm(messages: list[AnyMessage]) -> list[AnyMessage]:
    """
    OpenAI (and others) require that any assistant message with tool_calls is followed by
    tool messages responding to each tool_call_id. If a previous run crashed mid-tool-call,
    the persisted thread can contain dangling tool_calls which will cause hard 400 errors.

    We defensively drop any incomplete tool-call clusters before sending history to the LLM.
    """
    cleaned: list[AnyMessage] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            tool_calls = getattr(m, "tool_calls", None) or []
            call_ids = []
            for tc in tool_calls:
                try:
                    cid = tc.get("id") if isinstance(tc, dict) else None
                except Exception:
                    cid = None
                if cid:
                    call_ids.append(cid)

            # Collect following ToolMessages (if any)
            j = i + 1
            tool_msgs: list[AnyMessage] = []
            seen_tool_ids: set[str] = set()
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                tm = messages[j]
                tool_msgs.append(tm)
                try:
                    if getattr(tm, "tool_call_id", None):
                        seen_tool_ids.add(str(getattr(tm, "tool_call_id")))
                except Exception:
                    pass
                j += 1

            # If we have call_ids and all of them are satisfied, keep the cluster; else drop it.
            if not call_ids or all(cid in seen_tool_ids for cid in call_ids):
                cleaned.append(m)
                cleaned.extend(tool_msgs)
            else:
                _dbg_log(
                    "H0",
                    "agent.py:_messages_for_llm",
                    "dropped_incomplete_tool_calls",
                    {"missing_tool_call_ids": [cid for cid in call_ids if cid not in seen_tool_ids][:50]},
                )
            i = j
            continue

        cleaned.append(m)
        i += 1

    return cleaned


def route_mode(state: AgentState) -> dict:
    _ensure_defaults(state)
    return {}


def choose_mode(state: AgentState) -> Literal["build_assistant", "connections_load", "configure_load", "execute_load"]:
    mode = state.get("mode", "build")
    nxt: Literal["build_assistant", "connections_load", "configure_load", "execute_load"] = "build_assistant"
    if mode == "preflight":
        nxt = "connections_load"
    elif mode == "configure":
        nxt = "configure_load"
    elif mode == "execute":
        nxt = "execute_load"
    # region agent log
    _dbg_log("H2", "agent.py:choose_mode", "route", {"mode": mode, "next": nxt})
    # endregion agent log
    return nxt


def load_workflow_into_state(state: AgentState) -> dict:
    """If current_workflow_id is set, load workflow JSON into state.workflow for preflight/execute."""
    workflow_id = state.get("current_workflow_id")
    # Always reload from disk to ensure UI edits (or any external changes) are reflected.
    # Workflows are small JSON files; correctness > micro-optimizations.
    cache_hit = False
    found: Optional[bool] = None

    out: dict[str, Any] = {}
    if not workflow_id:
        out = {}
    else:
        wf = load_workflow(workflow_id)
        found = bool(wf)
        if not wf:
            out = {"error": f"Workflow '{workflow_id}' not found", "workflow": None, "workflow_markdown": None}
        else:
            wf_json = wf.model_dump(mode="json")
            out = {
                "workflow": wf_json,
                "workflow_markdown": _render_workflow_markdown(wf),
                "missing_config_keys": _missing_config_keys(wf_json),
                "error": None,
            }

    # region agent log
    _dbg_log(
        "H3",
        "agent.py:load_workflow_into_state",
        "load",
        {"workflow_id": workflow_id, "cache_hit": cache_hit, "found": found, "returned_keys": sorted(list(out.keys()))},
    )
    # endregion agent log
    return out


# ---------------------------------------------------------------------------
# BUILD loop
# ---------------------------------------------------------------------------


async def build_assistant(state: AgentState) -> dict:
    _ensure_defaults(state)
    current_id = state.get("current_workflow_id")
    wf = state.get("workflow") or {}
    wf_summary = ""
    if current_id and isinstance(wf, dict) and wf.get("id") == current_id:
        # Keep summary short to avoid bloating prompts.
        steps = wf.get("steps") or []
        if isinstance(steps, list):
            step_lines = []
            for i, s in enumerate(steps[:10], start=1):
                if isinstance(s, dict):
                    step_lines.append(f"{i}. {s.get('name', '')} ({s.get('tool', '')})")
            wf_summary = "\n".join(step_lines)

    sys_parts = [BUILD_SYSTEM_PROMPT]
    if current_id:
        sys_parts.append("\n## Current Workflow Context\n")
        sys_parts.append(f"- current_workflow_id: {current_id}\n")
        if wf_summary:
            sys_parts.append("### Current Steps (summary)\n" + wf_summary + "\n")
        sys_parts.append(
            "If the user asks to modify the current workflow, call update_workflow with workflow_id=current_workflow_id.\n"
        )

    sys_msg = SystemMessage(content="\n".join(sys_parts).strip())
    response = await build_llm_with_tools.ainvoke([sys_msg] + _messages_for_llm(state.get("messages", [])))
    return {"messages": [response]}


def build_should_continue(state: AgentState) -> Literal["build_tools", END]:
    messages = state.get("messages", [])
    if not messages:
        return END
    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "build_tools"
    return END


def build_hydrate(state: AgentState) -> dict:
    tm = _last_tool_message(state.get("messages", []))
    dbg: dict[str, Any] = {"has_tool_message": bool(tm)}
    out: dict[str, Any] = {}

    text = _tool_content_to_text(tm.content) if tm else ""
    dbg["tool_text_len"] = len(text)

    payload: Any = None
    parse_ok = False
    payload_type: Optional[str] = None
    try:
        payload = json.loads(text) if text else None
        parse_ok = True if text else False
    except Exception as e:
        dbg["json_error"] = type(e).__name__

    if isinstance(payload, dict):
        payload_type = str(payload.get("type")) if payload.get("type") is not None else None
        if payload_type == "workflow":
            wf_obj = payload.get("workflow")
            missing_cfg = _missing_config_keys(wf_obj) if isinstance(wf_obj, dict) else []
            out = {
                "current_workflow_id": payload.get("workflow_id"),
                "workflow": wf_obj,
                "workflow_markdown": payload.get("workflow_markdown"),
                "missing_config_keys": missing_cfg,
                "error": None,
            }
        elif payload_type == "workflow_list":
            out = {"workflows": payload.get("workflows", [])}

    dbg["json_parse_ok"] = parse_ok
    dbg["payload_type"] = payload_type
    dbg["returned_keys"] = sorted(list(out.keys()))

    # region agent log
    _dbg_log("H3", "agent.py:build_hydrate", "hydrate", dbg)
    # endregion agent log
    return out


# ---------------------------------------------------------------------------
# CONNECTIONS loop
# ---------------------------------------------------------------------------


async def connections_assistant(state: AgentState) -> dict:
    _ensure_defaults(state)
    wf = state.get("workflow")
    if not isinstance(wf, dict):
        msg = AIMessage(
            content=(
                "## Connections\n\n"
                "No workflow is currently selected. Please select a workflow and try again."
            )
        )
        return {"messages": [msg]}

    required_toolkits = wf.get("required_toolkits") or []
    if not isinstance(required_toolkits, list):
        required_toolkits = []

    sys_msg = SystemMessage(
        content=CONNECTIONS_SYSTEM_PROMPT_TEMPLATE.format(
            workflow_name=wf.get("name", "Untitled"),
            required_toolkits=", ".join(required_toolkits) if required_toolkits else "None",
        )
    )
    response = await preflight_llm_with_tools.ainvoke([sys_msg] + _messages_for_llm(state.get("messages", [])))
    # Never surface raw connect URLs in chat; UI renders connect buttons from auth_status.
    try:
        if isinstance(response, AIMessage) and not getattr(response, "tool_calls", None):
            c = getattr(response, "content", None)
            if isinstance(c, str):
                redacted = _redact_connect_urls(c)
                if redacted != c:
                    response = AIMessage(content=redacted)
    except Exception:
        pass
    return {"messages": [response]}


def connections_should_continue(state: AgentState) -> Literal["connections_tools", END]:
    messages = state.get("messages", [])
    if not messages:
        return END
    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "connections_tools"
    return END


def connections_hydrate(state: AgentState) -> dict:
    tm = _last_tool_message(state.get("messages", []))
    dbg: dict[str, Any] = {"has_tool_message": bool(tm)}
    out: dict[str, Any] = {}

    if tm:
        wf = state.get("workflow") or {}
        required_toolkits = []
        if isinstance(wf, dict) and isinstance(wf.get("required_toolkits"), list):
            required_toolkits = wf["required_toolkits"]
        tool_text = _tool_content_to_text(tm.content)
        auth_status = _extract_auth_status_from_tool_output(required_toolkits, tool_text)
        out = {"auth_status": auth_status}

        disconnected: list[str] = []
        try:
            toolkits_map = auth_status.get("toolkits", {}) if isinstance(auth_status, dict) else {}
            if isinstance(toolkits_map, dict):
                disconnected = [
                    k for k, v in toolkits_map.items() if isinstance(v, dict) and not bool(v.get("connected"))
                ]
        except Exception:
            disconnected = []

        json_ok = False
        try:
            json.loads(tool_text)
            json_ok = True
        except Exception:
            json_ok = False

        dbg.update(
            {
                "required_toolkits_count": len(required_toolkits),
                "tool_text_len": len(tool_text),
                "tool_text_json": json_ok,
                "connect_urls_count": len(_extract_connect_urls(tool_text)),
                "all_connected": bool(auth_status.get("all_connected")) if isinstance(auth_status, dict) else None,
                "disconnected_toolkits": disconnected[:20],
            }
        )

    # region agent log
    _dbg_log("H4", "agent.py:connections_hydrate", "hydrate", dbg)
    # endregion agent log
    return out


# ---------------------------------------------------------------------------
# CONFIGURE loop (build-time auth + resource selection)
# ---------------------------------------------------------------------------


def _last_human_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            c = getattr(m, "content", "")
            if isinstance(c, str):
                return c
            try:
                return json.dumps(c)
            except Exception:
                return str(c)
    return ""


# ---------------------------------------------------------------------------
# CONFIGURE loop (LLM-driven, prompt-regulated)
# ---------------------------------------------------------------------------


async def configure_assistant(state: AgentState) -> dict:
    """LLM-driven Configure stage that can handle many toolkits via Tool Router meta-tools."""
    _ensure_defaults(state)
    wf = state.get("workflow")
    if not isinstance(wf, dict):
        return {
            "messages": [
                AIMessage(
                    content="## Configure\n\nNo workflow is currently selected. Please select a workflow and try again."
                )
            ]
        }

    required_toolkits = wf.get("required_toolkits") or []
    if not isinstance(required_toolkits, list):
        required_toolkits = []

    missing = state.get("missing_config_keys")
    if not isinstance(missing, list):
        missing = _missing_config_keys(wf)
    missing = [k for k in missing if isinstance(k, str) and k.strip()]

    # Fast-path: if nothing is missing, don't spend an LLM call.
    if not missing:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "## Configure\n\n"
                        "✅ Configuration is complete.\n\n"
                        "You can keep refining the workflow in chat, or click **Run** when you’re ready."
                    )
                )
            ],
        }

    bullets = "\n".join([f"- **{k}**" for k in missing[:20]]) if missing else "- None"
    cfg = wf.get("config") if isinstance(wf.get("config"), dict) else {}
    workflow_id = wf.get("id") if isinstance(wf.get("id"), str) else state.get("current_workflow_id")
    if not isinstance(workflow_id, str) or not workflow_id:
        workflow_id = "unknown"
    tool_router_session_id = state.get("tool_router_session_id")
    if not isinstance(tool_router_session_id, str) or not tool_router_session_id:
        tool_router_session_id = ""

    sys_msg = SystemMessage(
        content=CONFIGURE_SYSTEM_PROMPT_TEMPLATE.format(
            workflow_name=wf.get("name", "Untitled"),
            workflow_description=wf.get("description", ""),
            workflow_id=workflow_id,
            required_toolkits=", ".join([str(t) for t in required_toolkits if isinstance(t, str)]) if required_toolkits else "None",
            missing_config_bullets=bullets,
            workflow_config_json=json.dumps(cfg, indent=2),
            tool_router_session_id=tool_router_session_id,
        )
    )

    response = await configure_llm_with_tools.ainvoke([sys_msg] + _messages_for_llm(state.get("messages", [])))
    # Never surface raw connect URLs in chat; UI renders connect buttons from auth_status.
    try:
        if isinstance(response, AIMessage) and not getattr(response, "tool_calls", None):
            c = getattr(response, "content", None)
            if isinstance(c, str):
                redacted = _redact_connect_urls(c)
                if redacted != c:
                    response = AIMessage(content=redacted)
    except Exception:
        pass
    return {"messages": [response]}


def configure_should_continue(state: AgentState) -> Literal["configure_tools", END]:
    messages = state.get("messages", [])
    if not messages:
        return END
    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "configure_tools"
        return END
    

def configure_hydrate(state: AgentState) -> dict:
    """
    Hydrate Configure state from tool outputs.

    - `set_workflow_config` updates workflow + missing_config_keys
    - `COMPOSIO_MANAGE_CONNECTIONS` updates auth_status (for AuthStatus UI panel)
    - `COMPOSIO_SEARCH_TOOLS` updates tool_router_session_id (best-effort)
    """
    messages = state.get("messages", [])

    # Process all trailing tool messages (ToolNode can emit multiple ToolMessages per AI tool_calls)
    trailing: list[ToolMessage] = []
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            trailing.append(m)
        else:
            break
    trailing = list(reversed(trailing))

    dbg: dict[str, Any] = {"tool_messages_count": len(trailing)}
    out: dict[str, Any] = {}

    if not trailing:
        _dbg_log("H4", "agent.py:configure_hydrate", "hydrate", dbg)
        return out

    for tm in trailing:
        tool_name = str(getattr(tm, "name", "") or "")
        text = _tool_content_to_text(tm.content)
        payload: Any = None
        try:
            payload = json.loads(text) if text else None
        except Exception:
            payload = None

        # Internal persistence tool returns a structured workflow payload.
        if isinstance(payload, dict) and payload.get("type") == "workflow":
            wf_obj = payload.get("workflow")
            if isinstance(wf_obj, dict):
                out.update(
                    {
                        "current_workflow_id": payload.get("workflow_id"),
                        "workflow": wf_obj,
                        "workflow_markdown": payload.get("workflow_markdown"),
                        "missing_config_keys": _missing_config_keys(wf_obj),
                        "error": None,
                    }
                )

        # Tool Router session id (from search results)
        if tool_name == "COMPOSIO_SEARCH_TOOLS":
            try:
                sid = None
                if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
                    sid = payload["data"].get("session", {}).get("id")
                if isinstance(sid, str) and sid:
                    out["tool_router_session_id"] = sid
            except Exception:
                pass

        # Auth status for Required Connections panel (connect buttons)
        if tool_name in {"COMPOSIO_MANAGE_CONNECTIONS", "COMPOSIO_SEARCH_TOOLS"}:
            wf_cur = state.get("workflow") or {}
            required = []
            if isinstance(wf_cur, dict) and isinstance(wf_cur.get("required_toolkits"), list):
                required = wf_cur["required_toolkits"]
            out["auth_status"] = _extract_auth_status_from_tool_output(required, text)

        dbg.setdefault("tools_seen", [])
        if isinstance(dbg.get("tools_seen"), list):
            dbg["tools_seen"].append(tool_name)

    _dbg_log("H4", "agent.py:configure_hydrate", "hydrate", dbg)
    return out

# ---------------------------------------------------------------------------
# EXECUTE loop
# ---------------------------------------------------------------------------


async def execute_assistant(state: AgentState) -> dict:
    _ensure_defaults(state)
    wf = state.get("workflow")
    if not isinstance(wf, dict):
        return {
            "messages": [
                AIMessage(
                    content="## Execution\n\nNo workflow is selected. Please select a workflow and try again."
                )
            ]
        }

    steps = wf.get("steps") or []
    if not isinstance(steps, list) or not steps:
        return {"messages": [AIMessage(content="## Execution\n\nWorkflow has no steps.")]}

    # Hard guard: never start execution for workflows that require connections
    # unless we have an auth_status marking all required toolkits as connected.
    required_toolkits = wf.get("required_toolkits") or []
    if not isinstance(required_toolkits, list):
        required_toolkits = []
    if required_toolkits:
        auth = state.get("auth_status")
        ok = bool(isinstance(auth, dict) and auth.get("all_connected"))
        toolkits_map = auth.get("toolkits", {}) if isinstance(auth, dict) else {}
        if ok and isinstance(toolkits_map, dict):
            for tk in required_toolkits:
                entry = toolkits_map.get(tk)
                if not (isinstance(entry, dict) and bool(entry.get("connected")) and not entry.get("error")):
                    ok = False
                    break
        else:
            ok = False

        if not ok:
        return {
                "messages": [
                    AIMessage(
                        content=(
                            "## Execution blocked\n\n"
                            "This workflow requires external connections. Please run **Connections** first, "
                            "complete OAuth if needed, then click **Run** again."
                        )
                    )
                ]
            }

    # Hard guard: never start execution if config placeholders are still unresolved.
    missing_cfg = _missing_config_keys(wf)
    if missing_cfg:
        bullets = "\n".join([f"- **{k}**" for k in missing_cfg[:20]])
        return {
            "messages": [
                AIMessage(
                    content=(
                        "## Execution blocked\n\n"
                        "This workflow still needs configuration before it can run:\n\n"
                        f"{bullets}\n\n"
                        "Please complete **Configure** first, then try again."
                    )
                )
            ]
        }

    cursor = int(state.get("execution_cursor", 0))
    if cursor < 0:
        cursor = 0
    if cursor >= len(steps):
        return {"messages": [AIMessage(content="## Execution\n\nWorkflow is already complete.")]}

    step = steps[cursor]
    if not isinstance(step, dict):
        return {"messages": [AIMessage(content="## Execution\n\nInvalid step format.")]}

    # Determine whether this step is internal-only (no Tool Router).
    step_tool = step.get("tool", "")
    step_toolkit = step.get("toolkit", "")
    is_internal = bool(step.get("execution_method") == "internal" or step_toolkit == "internal" or step_tool == "AI_PROCESS")

    cfg = wf.get("config") if isinstance(wf.get("config"), dict) else {}
    raw_params = step.get("tool_params", {}) or {}
    resolved_params = _resolve_config_placeholders(raw_params, cfg) if isinstance(cfg, dict) else raw_params

    sys_msg = SystemMessage(
        content=EXECUTE_SYSTEM_PROMPT_TEMPLATE.format(
            workflow_name=wf.get("name", "Untitled"),
            workflow_description=wf.get("description", ""),
            step_order=step.get("order", cursor + 1),
            total_steps=len(steps),
            step_name=step.get("name", ""),
            step_tool=step_tool,
            step_toolkit=step_toolkit,
            step_instructions=step.get("instructions", ""),
            step_params_json=json.dumps(resolved_params, indent=2),
            step_results_json=json.dumps(state.get("step_results", {}), indent=2),
        )
    )

    # Internal steps: do NOT bind Tool Router tools (hard guarantee).
    llm = execute_llm_internal if is_internal else execute_llm_with_tools
    response = await llm.ainvoke([sys_msg] + _messages_for_llm(state.get("messages", [])))
    return {"messages": [response]}


def execute_should_continue(state: AgentState) -> Literal["execute_tools", "execute_advance", END]:
    messages = state.get("messages", [])
    decision: Literal["execute_tools", "execute_advance", END] = END
    last_type: Optional[str] = None
    has_tool_calls = False
    cursor = int(state.get("execution_cursor", 0))
    steps_len: Optional[int] = None

    if not messages:
        decision = END
    else:
        last = messages[-1]
        last_type = last.__class__.__name__
        has_tool_calls = bool(isinstance(last, AIMessage) and getattr(last, "tool_calls", None))

        if has_tool_calls:
            decision = "execute_tools"
        else:
            # If tool output or assistant output indicates OAuth/connect is required, STOP execution.
            # This prevents the graph from advancing steps while the user is completing OAuth.
            try:
                text = ""
                if isinstance(last, (AIMessage, ToolMessage)):
                    text = _tool_content_to_text(last.content)
                if _extract_connect_urls(text) or (
                    isinstance(last, ToolMessage) and getattr(last, "status", None) == "error"
                ):
                    decision = END
                    _dbg_log(
                        "H5",
                        "agent.py:execute_should_continue",
                        "blocked_by_auth",
                        {"cursor": cursor},
                    )
            except Exception:
                # best-effort only
                pass

            wf = state.get("workflow")
            if isinstance(wf, dict) and isinstance(wf.get("steps"), list):
                steps_len = len(wf["steps"])
                if cursor >= steps_len:
                    decision = END
                elif state.get("mode") == "execute" and decision != END:
                    decision = "execute_advance"
                else:
                    decision = END
            elif state.get("mode") == "execute" and decision != END:
                decision = "execute_advance"
            else:
                decision = END

    # region agent log
    _dbg_log(
        "H5",
        "agent.py:execute_should_continue",
        "decide",
        {
            "mode": state.get("mode"),
            "last_message_type": last_type,
            "has_tool_calls": has_tool_calls,
            "cursor": cursor,
            "steps_len": steps_len,
            "decision": decision if decision != END else "END",
        },
    )
    # endregion agent log
    return decision


def execute_hydrate(state: AgentState) -> dict:
    """Store the latest relevant tool output into step_results for the current step."""
    dbg: dict[str, Any] = {"reason": None}
    out: dict[str, Any] = {}

    wf = state.get("workflow")
    if not isinstance(wf, dict):
        dbg["reason"] = "no_workflow"
    else:
        steps = wf.get("steps") or []
        if not isinstance(steps, list) or not steps:
            dbg["reason"] = "no_steps"
        else:
            cursor = int(state.get("execution_cursor", 0))
            if cursor < 0 or cursor >= len(steps):
                dbg["reason"] = "cursor_out_of_range"
                dbg["cursor"] = cursor
                dbg["steps_len"] = len(steps)
            else:
                step = steps[cursor]
                if not isinstance(step, dict):
                    dbg["reason"] = "invalid_step"
                    dbg["cursor"] = cursor
                else:
                    step_id = step.get("id") or f"step_{cursor+1}"
                    tm = _last_tool_message(state.get("messages", []))
                    if not tm:
                        dbg["reason"] = "no_tool_message"
                        dbg["cursor"] = cursor
                        dbg["step_id"] = step_id
                    else:
                        tool_name_raw = getattr(tm, "name", None) or ""
                        tool_name = tool_name_raw
                        if tool_name not in {
                            "COMPOSIO_MULTI_EXECUTE_TOOL",
                            "COMPOSIO_SEARCH_TOOLS",
                            "COMPOSIO_GET_TOOL_SCHEMAS",
                        }:
                            tool_name = tool_name or "tool"

                        tool_text = _tool_content_to_text(tm.content)
                        updated_results = dict(state.get("step_results", {}) or {})

                        # Normalize shape:
                        # step_results[step_id] = {
                        #   "tool_outputs": { ... },
                        #   "assistant_output": "...",
                        #   "outputs": { "<outputs_name>": value }
                        # }
                        entry = updated_results.get(step_id)
                        if not isinstance(entry, dict):
                            entry = {}
                        tool_outputs = entry.get("tool_outputs")
                        if not isinstance(tool_outputs, dict):
                            tool_outputs = {}
                        tool_outputs[tool_name] = tool_text
                        entry["tool_outputs"] = tool_outputs
                        updated_results[step_id] = entry
                        out = {"step_results": updated_results}
                        dbg.update(
                            {
                                "reason": "stored",
                                "cursor": cursor,
                                "step_id": step_id,
                                "tool_name_raw": tool_name_raw,
                                "stored_key": tool_name,
                                "tool_text_len": len(tool_text),
                            }
                        )

    # region agent log
    _dbg_log("H5", "agent.py:execute_hydrate", "hydrate", dbg)
    # endregion agent log
    return out


def execute_advance(state: AgentState) -> dict:
    """Advance execution cursor after the assistant completes a step summary."""
    wf = state.get("workflow")
    cursor_before = int(state.get("execution_cursor", 0))
    cursor_after: Optional[int] = None
    done: Optional[bool] = None
    validate_ok: Optional[bool] = None
    validate_error: Optional[str] = None
    stored_output: Optional[bool] = None
    step_id: Optional[str] = None
    outputs_name: Optional[str] = None
    is_internal: Optional[bool] = None
    step_results_update: Optional[dict] = None

    try:
        if not isinstance(wf, dict):
            return {}
        steps = wf.get("steps") or []
        if not isinstance(steps, list) or not steps:
            return {}
        cursor = cursor_before
        if cursor < 0:
            cursor = 0

        # Capture assistant output (this is the last AI message when no tool calls happened).
        last_msg = state.get("messages", [])[-1] if state.get("messages") else None
        assistant_output = None
        if isinstance(last_msg, AIMessage):
            assistant_output = _tool_content_to_text(last_msg.content)

        # Identify step metadata
        if cursor < len(steps) and isinstance(steps[cursor], dict):
            step_id = steps[cursor].get("id") or f"step_{cursor+1}"
            outputs_name = steps[cursor].get("outputs") or "output"
            tool = steps[cursor].get("tool") or ""
            toolkit = steps[cursor].get("toolkit") or ""
            is_internal = bool(steps[cursor].get("execution_method") == "internal" or toolkit == "internal" or tool == "AI_PROCESS")

            # Store a deterministic step output in step_results:
            # - internal: assistant_output
            # - external: COMPOSIO_MULTI_EXECUTE_TOOL output if present
            step_results = dict(state.get("step_results", {}) or {})
            entry = step_results.get(step_id)
            if not isinstance(entry, dict):
                entry = {}

            if isinstance(assistant_output, str) and assistant_output.strip():
                entry["assistant_output"] = assistant_output.strip()

            tool_outputs = entry.get("tool_outputs")
            if not isinstance(tool_outputs, dict):
                tool_outputs = {}

            output_value = None
            if is_internal:
                output_value = entry.get("assistant_output")
            else:
                output_value = tool_outputs.get("COMPOSIO_MULTI_EXECUTE_TOOL") or tool_outputs.get("tool")

            outputs_map = entry.get("outputs")
            if not isinstance(outputs_map, dict):
                outputs_map = {}
            if output_value is not None and outputs_name:
                outputs_map[str(outputs_name)] = output_value
            entry["outputs"] = outputs_map
            entry["tool_outputs"] = tool_outputs

            step_results[step_id] = entry
            step_results_update = step_results
            stored_output = True
        else:
            stored_output = False

        # Mark current step completed in the workflow view (best-effort)
        if cursor < len(steps) and isinstance(steps[cursor], dict):
            steps[cursor]["status"] = "completed"

        cursor += 1
        cursor_after = cursor
        done = cursor >= len(steps)

        # Persist workflow status changes (best-effort)
        if done and isinstance(wf.get("id"), str):
            try:
                model = Workflow.model_validate(wf)
                model.status = "completed"  # type: ignore[assignment]
                model.steps = model.steps  # keep
                saved = save_workflow(model)
                wf = saved.model_dump(mode="json")
            except Exception:
                pass

        validated = Workflow.model_validate(wf)
        validate_ok = True
        out = {
            "execution_cursor": cursor,
            "workflow": wf,
            "workflow_markdown": _render_workflow_markdown(validated),
        }
        if step_results_update is not None:
            out["step_results"] = step_results_update
        return out
    except Exception as e:
        validate_ok = False
        validate_error = type(e).__name__
        raise
    finally:
        # region agent log
        _dbg_log(
            "H5",
            "agent.py:execute_advance",
            "advance",
            {
                "cursor_before": cursor_before,
                "cursor_after": cursor_after,
                "done": done,
                "validate_ok": validate_ok,
                "validate_error": validate_error,
                "step_id": step_id,
                "outputs_name": outputs_name,
                "is_internal": is_internal,
                "stored_output": stored_output,
            },
        )
        # endregion agent log


def execute_done(state: AgentState) -> Literal["execute_assistant", END]:
    wf = state.get("workflow")
    if not isinstance(wf, dict):
        return END
    steps = wf.get("steps") or []
    if not isinstance(steps, list) or not steps:
        return END
    cursor = int(state.get("execution_cursor", 0))
    if cursor >= len(steps):
        return END
    return "execute_assistant"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


builder = StateGraph(AgentState)

# Router
builder.add_node("route_mode", route_mode)
builder.add_conditional_edges("route_mode", choose_mode, ["build_assistant", "connections_load", "configure_load", "execute_load"])

# Build loop
builder.add_node("build_assistant", build_assistant)
builder.add_node("build_tools", ToolNode(BUILD_TOOLS, handle_tool_errors=True))
builder.add_node("build_hydrate", build_hydrate)

builder.add_edge("build_tools", "build_hydrate")
builder.add_edge("build_hydrate", "build_assistant")
builder.add_conditional_edges("build_assistant", build_should_continue, ["build_tools", END])

# Connections loop
builder.add_node("connections_load", load_workflow_into_state)
builder.add_node("connections_assistant", connections_assistant)
builder.add_node("connections_tools", ToolNode(PREFLIGHT_TOOLS, handle_tool_errors=True))
builder.add_node("connections_hydrate", connections_hydrate)

builder.add_edge("connections_load", "connections_assistant")
builder.add_edge("connections_tools", "connections_hydrate")
builder.add_edge("connections_hydrate", "connections_assistant")
builder.add_conditional_edges("connections_assistant", connections_should_continue, ["connections_tools", END])

# Configure (build-time auth + config)
builder.add_node("configure_load", load_workflow_into_state)
builder.add_node("configure_assistant", configure_assistant)
builder.add_node("configure_tools", ToolNode(CONFIGURE_TOOLS, handle_tool_errors=True))
builder.add_node("configure_hydrate", configure_hydrate)

builder.add_edge("configure_load", "configure_assistant")
builder.add_edge("configure_tools", "configure_hydrate")
builder.add_edge("configure_hydrate", "configure_assistant")
builder.add_conditional_edges("configure_assistant", configure_should_continue, ["configure_tools", END])

# Execute loop
builder.add_node("execute_load", load_workflow_into_state)
builder.add_node("execute_assistant", execute_assistant)
builder.add_node("execute_tools", ToolNode(EXECUTE_TOOLS, handle_tool_errors=True))
builder.add_node("execute_hydrate", execute_hydrate)
builder.add_node("execute_advance", execute_advance)

builder.add_edge("execute_load", "execute_assistant")
builder.add_edge("execute_tools", "execute_hydrate")
builder.add_edge("execute_hydrate", "execute_assistant")
builder.add_conditional_edges("execute_assistant", execute_should_continue, ["execute_tools", "execute_advance", END])
builder.add_conditional_edges("execute_advance", execute_done, ["execute_assistant", END])

# Start
builder.add_edge(START, "route_mode")

# Compile graph
graph = builder.compile()


