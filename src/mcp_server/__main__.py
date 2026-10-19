"""
src/mcp_server/__main__.py
──────────────────────────
Entry point for:  python -m src.mcp_server
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.mcp_server.server import mcp

mcp.run(transport="stdio")