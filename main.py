"""
main.py
───────
Entry point for the SimuScan Config Agent.

Usage:
    python main.py

Reads OPENAI_API_KEY and OPENAI_MODEL from the .env file in the project root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    # python-dotenv not installed — fall back to OS environment
    pass

# ── Validate required env vars ────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o").strip()

if not OPENAI_API_KEY:
    print(
        "ERROR: OPENAI_API_KEY not found.\n"
        "Set it in the .env file at the project root:\n\n"
        "  OPENAI_API_KEY=sk-...\n"
        "  OPENAI_MODEL=gpt-4o\n",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Launch agent ──────────────────────────────────────────────────────────────
from src.agent import run

if __name__ == "__main__":
    debug = "--debug" in sys.argv
    run(openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL, debug=debug)