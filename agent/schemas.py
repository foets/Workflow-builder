"""
Schemas for the workflow builder.

We keep Workflows JSON-first (source of truth). Markdown is only a rendered view.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


StepStatus = Literal["pending", "running", "completed", "failed"]
WorkflowStatus = Literal["draft", "ready", "executing", "completed", "failed"]
AgentMode = Literal["build", "preflight", "configure", "execute"]
ExecutionMethod = Literal["internal", "tool_router"]


TOOL_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
TOOLKIT_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class WorkflowStepDraft(BaseModel):
    """Step shape expected from the BUILD agent (before IDs/orders are assigned)."""

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=500)
    instructions: str = Field(min_length=1, max_length=10_000)
    tool: str = Field(min_length=1, max_length=80)
    tool_params: dict[str, Any] = Field(default_factory=dict)
    toolkit: str = Field(min_length=1, max_length=40)
    inputs: list[str] = Field(default_factory=list)
    outputs: str = Field(min_length=1, max_length=80)
    execution_method: ExecutionMethod = "tool_router"
    requires_connection: bool = True

    @field_validator("tool")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        if not TOOL_NAME_RE.match(v):
            raise ValueError("tool must be UPPERCASE_WITH_UNDERSCORES (e.g., GMAIL_GET_MESSAGES, AI_PROCESS)")
        return v

    @field_validator("toolkit")
    @classmethod
    def validate_toolkit(cls, v: str) -> str:
        # Allow "internal" for AI_PROCESS steps.
        if v == "internal":
            return v
        if not TOOLKIT_RE.match(v):
            raise ValueError("toolkit must be lowercase (e.g., gmail, googledocs, slack, internal)")
        return v

    @model_validator(mode="after")
    def derive_execution_flags(self) -> "WorkflowStepDraft":
        # Canonical: AI_PROCESS is always internal
        if self.tool == "AI_PROCESS" and self.toolkit != "internal":
            raise ValueError("AI_PROCESS steps must use toolkit 'internal'")

        internal = (self.toolkit == "internal") or (self.tool == "AI_PROCESS")
        self.execution_method = "internal" if internal else "tool_router"
        self.requires_connection = False if internal else True
        return self


class WorkflowDraft(BaseModel):
    """Workflow shape expected from the BUILD agent (before IDs are assigned)."""

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=600)
    steps: list[WorkflowStepDraft] = Field(min_length=1)


class WorkflowStep(BaseModel):
    id: str
    order: int = Field(ge=1)
    name: str
    description: str
    instructions: str
    tool: str
    tool_params: dict[str, Any] = Field(default_factory=dict)
    toolkit: str
    inputs: list[str] = Field(default_factory=list)
    outputs: str
    status: StepStatus = "pending"
    execution_method: ExecutionMethod = "tool_router"
    requires_connection: bool = True

    @model_validator(mode="after")
    def derive_execution_flags(self) -> "WorkflowStep":
        if self.tool == "AI_PROCESS" and self.toolkit != "internal":
            raise ValueError("AI_PROCESS steps must use toolkit 'internal'")
        internal = (self.toolkit == "internal") or (self.tool == "AI_PROCESS")
        self.execution_method = "internal" if internal else "tool_router"
        self.requires_connection = False if internal else True
        return self


class Workflow(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    status: WorkflowStatus = "draft"
    config: dict[str, Any] = Field(default_factory=dict)
    required_toolkits: list[str] = Field(default_factory=list)
    steps: list[WorkflowStep]

    @model_validator(mode="after")
    def derive_required_toolkits(self) -> "Workflow":
        toolkits = []
        for s in self.steps:
            if s.toolkit and s.toolkit != "internal":
                toolkits.append(s.toolkit)
        # stable order (first appearance)
        seen = set()
        ordered = []
        for t in toolkits:
            if t not in seen:
                seen.add(t)
                ordered.append(t)
        self.required_toolkits = ordered
        return self
