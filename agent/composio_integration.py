"""
Composio Tool Router Integration for Workflow Builder

Uses the Tool Router + MCP pattern for proper session management,
authentication handling, and toolkit scoping.

Tool Router handles:
- Tool discovery across integrations
- Authentication flows (prompts user with OAuth link if needed)
- Tool execution with proper context
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Any
from dotenv import load_dotenv

# Load .env from the agent directory first (works whether cwd is repo root or agent/)
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)
# Fallback to default discovery (cwd-based)
load_dotenv(override=False)

# ============================================================================
# Package Availability
# ============================================================================

COMPOSIO_AVAILABLE = False
Composio = None
MultiServerMCPClient = None

try:
    from composio import Composio
    COMPOSIO_AVAILABLE = True
except ImportError:
    pass

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None
    if COMPOSIO_AVAILABLE:
        print("Warning: langchain-mcp-adapters not installed. Run: pip install langchain-mcp-adapters")
        COMPOSIO_AVAILABLE = False


# ============================================================================
# TOOLKITS CONFIGURATION
# ============================================================================
# TODO: Add your toolkit slugs here
# Find available toolkits at: https://app.composio.dev/apps
# 
# Format: lowercase slug names as strings
# Example: ["gmail", "googledocs", "slack", "notion", "zohocrm"]

WORKFLOW_TOOLKITS = [
    "gmail",
    "googledrive",
    "googledocs",
    "googlesheets",
    "zoho",
    "zoho_mail",
    "zoho_bigin",
    "slack",
    "notion"
]


# ============================================================================
# Global Session State
# ============================================================================

_composio_client: Optional[Any] = None
_composio_session: Optional[Any] = None
_mcp_client: Optional[Any] = None
_cached_tools: Optional[List] = None


# ============================================================================
# Core Functions
# ============================================================================

def get_api_key() -> Optional[str]:
    """Get Composio API key from environment."""
    return os.getenv("COMPOSIO_API_KEY")


def is_configured() -> bool:
    """Check if Composio is properly configured."""
    return COMPOSIO_AVAILABLE and get_api_key() is not None


async def create_session(user_id: str = "workflow_builder_user") -> Tuple[Optional[Any], Optional[Any]]:
    """
    Create a Composio Tool Router session with MCP protocol.
    
    Sessions are user-scoped and limited to configured toolkits.
    Tool Router handles authentication automatically - when a tool
    needs auth, it returns a link for the user to connect their account.
    
    Args:
        user_id: Unique identifier for the user session
        
    Returns:
        Tuple of (session, mcp_client) or (None, None) if unavailable
    """
    global _composio_client, _composio_session, _mcp_client, _cached_tools
    
    if not COMPOSIO_AVAILABLE:
        print("❌ Composio not available. Install: pip install composio langchain-mcp-adapters")
        return None, None
    
    api_key = get_api_key()
    if not api_key:
        print("❌ COMPOSIO_API_KEY not set in environment")
        return None, None
    
    try:
        # Initialize Composio client
        _composio_client = Composio(api_key=api_key)
        
        # Create Tool Router session with toolkit limitations
        # This scopes the session to ONLY the specified toolkits
        _composio_session = _composio_client.create(
            user_id=user_id,
            toolkits=WORKFLOW_TOOLKITS
        )
        
        # Get MCP URL from the session
        mcp_url = _composio_session.mcp.url
        
        # Create MCP client to connect to Composio's MCP server
        _mcp_client = MultiServerMCPClient({
            "composio": {
                "transport": "http",
                "url": mcp_url,
                "headers": {"x-api-key": api_key}
            }
        })
        
        # Clear cached tools for fresh session
        _cached_tools = None
        
        print(f"✓ Composio Tool Router session created")
        print(f"  User: {user_id}")
        print(f"  Toolkits: {', '.join(WORKFLOW_TOOLKITS)}")
        
        return _composio_session, _mcp_client
        
    except Exception as e:
        print(f"❌ Error creating Composio session: {e}")
        _composio_client = None
        _composio_session = None
        _mcp_client = None
        return None, None


async def get_tools() -> List:
    """
    Get LangChain-compatible tools from Composio Tool Router.
    
    Tools are fetched from the MCP server and automatically handle:
    - Authentication (returns OAuth link if needed)
    - Execution with proper context
    - Response formatting
    
    Returns:
        List of LangChain-compatible tools
    """
    global _mcp_client, _cached_tools
    
    if _cached_tools is not None:
        return _cached_tools
    
    if _mcp_client is None:
        await create_session()
    
    if _mcp_client is None:
        return []
    
    try:
        tools = await _mcp_client.get_tools()
        _cached_tools = tools
        print(f"✓ Loaded {len(tools)} tools from Composio")
        return tools
    except Exception as e:
        print(f"❌ Error getting tools: {e}")
        return []


async def close_session():
    """Close the Composio session and clean up."""
    global _composio_client, _composio_session, _mcp_client, _cached_tools
    
    _composio_client = None
    _composio_session = None
    _mcp_client = None
    _cached_tools = None


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'create_session',
    'get_tools',
    'close_session',
    'is_configured',
    'COMPOSIO_AVAILABLE',
    'WORKFLOW_TOOLKITS',
]
