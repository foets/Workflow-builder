"""
Workflow Files Manager

Stores workflows as Markdown files in ~/.workflow-builder/
Each workflow is a single .md file containing:
- Workflow name and description
- List of tools used
- Sequential steps with instructions and parameters

This provides a simple, reliable, file-based persistence layer.
"""

import os
import re
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path


# ============================================================================
# Configuration
# ============================================================================

# Storage directory in user's home
WORKFLOWS_DIR = os.path.expanduser("~/.workflow-builder")

def ensure_workflows_dir():
    """Ensure the workflows directory exists."""
    os.makedirs(WORKFLOWS_DIR, exist_ok=True)


# ============================================================================
# Markdown Schema
# ============================================================================

WORKFLOW_TEMPLATE = """# {name}

> {description}

### 🔧 Tools Used
{tools}

---

{steps}

---

<sub>📅 Created: {created_at} | ✏️ Modified: {updated_at}</sub>
"""

STEP_TEMPLATE = """## 📌 Step {order}: {name}

| Property | Value |
|----------|-------|
| **Tool** | `{tool}` |

### 📝 Description
{description}

### 📋 Instructions
```
{instructions}
```

### ⚙️ Parameters
{parameters}
"""


def create_workflow_markdown(
    name: str,
    description: str,
    steps: List[Dict[str, Any]],
    tools: Optional[List[str]] = None
) -> str:
    """
    Create a Markdown string for a workflow.
    
    Args:
        name: Workflow name
        description: Two-sentence summary
        steps: List of step dictionaries
        tools: Optional list of tools (auto-extracted from steps if not provided)
    
    Returns:
        Formatted Markdown string
    """
    # Extract tools from steps if not provided
    if not tools:
        tools = list(set(
            step.get("tool", "").split("_")[0] 
            for step in steps 
            if step.get("tool")
        ))
    
    tools_md = "\n".join(f"- {tool.upper()}" for tool in tools) if tools else "- None specified"
    
    # Build steps markdown
    steps_md_parts = []
    for i, step in enumerate(steps, 1):
        # Format parameters
        params = step.get("tool_params", step.get("parameters", {}))
        if isinstance(params, dict):
            if params:
                params_md = "\n".join(f"- {k}: {v}" for k, v in params.items())
            else:
                params_md = "- None"
        else:
            params_md = f"- {params}"
        
        step_md = STEP_TEMPLATE.format(
            order=i,
            name=step.get("name", f"Step {i}"),
            tool=step.get("tool", "UNKNOWN"),
            description=step.get("description", ""),
            instructions=step.get("instructions", "No instructions provided."),
            parameters=params_md
        )
        steps_md_parts.append(step_md)
    
    steps_md = "\n---\n\n".join(steps_md_parts)
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    return WORKFLOW_TEMPLATE.format(
        name=name,
        description=description,
        tools=tools_md,
        steps=steps_md,
        created_at=now,
        updated_at=now
    )


def parse_workflow_markdown(content: str) -> Dict[str, Any]:
    """
    Parse a Markdown workflow file into a structured dictionary.
    
    Args:
        content: Markdown content
    
    Returns:
        Dictionary with name, description, tools, steps
    """
    result = {
        "name": "",
        "description": "",
        "tools": [],
        "steps": [],
        "created_at": "",
        "updated_at": ""
    }
    
    # Parse name (first H1)
    name_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if name_match:
        result["name"] = name_match.group(1).strip()
    
    # Parse description (blockquote after name)
    desc_match = re.search(r'^>\s*(.+)$', content, re.MULTILINE)
    if desc_match:
        result["description"] = desc_match.group(1).strip()
    
    # Parse tools section
    tools_match = re.search(r'## Tools\n((?:- .+\n?)+)', content)
    if tools_match:
        tools_text = tools_match.group(1)
        result["tools"] = [
            line.strip("- ").strip().upper()
            for line in tools_text.strip().split("\n")
            if line.strip().startswith("-")
        ]
    
    # Parse steps
    step_pattern = r'## Step (\d+):\s*(.+?)\n\n\*\*Tool:\*\*\s*`(.+?)`\n\n(.+?)\n\n\*\*Instructions:\*\*\n(.+?)\n\n\*\*Parameters:\*\*\n((?:- .+\n?)+)'
    step_matches = re.finditer(step_pattern, content, re.DOTALL)
    
    for match in step_matches:
        order, name, tool, description, instructions, params_text = match.groups()
        
        # Parse parameters
        params = {}
        for line in params_text.strip().split("\n"):
            if line.strip().startswith("-") and ":" in line:
                key, value = line.strip("- ").split(":", 1)
                params[key.strip()] = value.strip()
        
        result["steps"].append({
            "id": f"step_{order}",
            "order": int(order),
            "name": name.strip(),
            "tool": tool.strip(),
            "description": description.strip(),
            "instructions": instructions.strip(),
            "tool_params": params,
            "status": "ready"
        })
    
    # Parse timestamps
    time_match = re.search(r'\*Created:\s*(.+?)\s*\|\s*Last Modified:\s*(.+?)\*', content)
    if time_match:
        result["created_at"] = time_match.group(1).strip()
        result["updated_at"] = time_match.group(2).strip()
    
    return result


# ============================================================================
# File Operations
# ============================================================================

def sanitize_filename(name: str) -> str:
    """Convert a workflow name to a safe filename."""
    # Remove/replace invalid characters
    safe = re.sub(r'[^\w\s-]', '', name)
    safe = re.sub(r'[-\s]+', '-', safe).strip('-')
    return safe.lower()[:50]  # Limit length


def get_workflow_path(name: str) -> str:
    """Get the full path for a workflow file."""
    ensure_workflows_dir()
    filename = sanitize_filename(name)
    return os.path.join(WORKFLOWS_DIR, f"{filename}.md")


def save_workflow_file(name: str, content: str) -> str:
    """
    Save workflow content to a Markdown file.
    
    Args:
        name: Workflow name (used as filename)
        content: Markdown content
    
    Returns:
        Path to the saved file
    """
    ensure_workflows_dir()
    filepath = get_workflow_path(name)
    
    # Update the "Last Modified" timestamp
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    content = re.sub(
        r'Last Modified:\s*[^*]+\*',
        f'Last Modified: {now}*',
        content
    )
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    return filepath


def get_workflow_file(name: str) -> Optional[str]:
    """
    Read a workflow file by name.
    
    Args:
        name: Workflow name (matches filename)
    
    Returns:
        Markdown content, or None if not found
    """
    filepath = get_workflow_path(name)
    if not os.path.exists(filepath):
        # Try exact filename match
        for filename in os.listdir(WORKFLOWS_DIR):
            if filename.endswith(".md"):
                check_path = os.path.join(WORKFLOWS_DIR, filename)
                try:
                    with open(check_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Check if this file's name matches
                        name_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                        if name_match and name_match.group(1).strip().lower() == name.lower():
                            return content
                except Exception:
                    pass
        return None
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def get_workflow_by_filename(filename: str) -> Optional[str]:
    """
    Read a workflow file by its filename (without extension).
    
    Args:
        filename: The filename (without .md extension)
    
    Returns:
        Markdown content, or None if not found
    """
    ensure_workflows_dir()
    filepath = os.path.join(WORKFLOWS_DIR, f"{filename}.md")
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def list_workflow_files() -> List[Dict[str, str]]:
    """
    List all workflow files.
    
    Returns:
        List of dicts with filename, name, description, modified
    """
    ensure_workflows_dir()
    workflows = []
    
    for filename in os.listdir(WORKFLOWS_DIR):
        if filename.endswith(".md"):
            filepath = os.path.join(WORKFLOWS_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Parse basic info
                name_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                desc_match = re.search(r'^>\s*(.+)$', content, re.MULTILINE)
                
                # Count steps
                step_count = len(re.findall(r'^## Step \d+:', content, re.MULTILINE))
                
                workflows.append({
                    "filename": filename[:-3],  # Remove .md extension
                    "name": name_match.group(1).strip() if name_match else filename[:-3],
                    "description": desc_match.group(1).strip() if desc_match else "",
                    "step_count": step_count,
                    "modified": datetime.fromtimestamp(
                        os.path.getmtime(filepath)
                    ).strftime("%Y-%m-%d %H:%M")
                })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    # Sort by modification time (newest first)
    workflows.sort(key=lambda x: x["modified"], reverse=True)
    return workflows


def delete_workflow_file(name: str) -> bool:
    """
    Delete a workflow file.
    
    Args:
        name: Workflow name or filename
    
    Returns:
        True if deleted, False if not found
    """
    filepath = get_workflow_path(name)
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    
    # Try direct filename
    direct_path = os.path.join(WORKFLOWS_DIR, f"{name}.md")
    if os.path.exists(direct_path):
        os.remove(direct_path)
        return True
    
    return False


# ============================================================================
# High-Level Operations
# ============================================================================

def create_and_save_workflow(
    name: str,
    description: str,
    steps: List[Dict[str, Any]],
    tools: Optional[List[str]] = None
) -> str:
    """
    Create a new workflow and save it to disk.
    
    Args:
        name: Workflow name
        description: Two-sentence summary
        steps: List of step dictionaries
        tools: Optional list of tools
    
    Returns:
        Path to the saved file
    """
    content = create_workflow_markdown(name, description, steps, tools)
    return save_workflow_file(name, content)


def update_workflow_from_markdown(filename: str, new_content: str) -> Optional[str]:
    """
    Update a workflow file with new Markdown content.
    
    Args:
        filename: The filename (without extension)
        new_content: New Markdown content
    
    Returns:
        Path to the saved file, or None if failed
    """
    ensure_workflows_dir()
    filepath = os.path.join(WORKFLOWS_DIR, f"{filename}.md")
    
    try:
        # Update modification timestamp
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        new_content = re.sub(
            r'Last Modified:\s*[^*]+\*',
            f'Last Modified: {now}*',
            new_content
        )
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        return filepath
    except Exception as e:
        print(f"Error updating workflow: {e}")
        return None


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'WORKFLOWS_DIR',
    'ensure_workflows_dir',
    'create_workflow_markdown',
    'parse_workflow_markdown',
    'save_workflow_file',
    'get_workflow_file',
    'get_workflow_by_filename',
    'list_workflow_files',
    'delete_workflow_file',
    'create_and_save_workflow',
    'update_workflow_from_markdown',
]

