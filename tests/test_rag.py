"""
tests/test_rag.py
──────────────────
Manual test of the RAG system: indexes docs/ and runs searches.
Run from the project root:
    uv run python tests/test_rag.py
"""

import sys
from pathlib import Path

# Add src/ to the path to import the modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from mcp_server.rag.indexer import index_docs_folder, list_indexed_documents
from mcp_server.rag.retriever import retrieve

# ── ANSI colors ───────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def section(title: str):
    print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'-'*60}{RESET}")

# ── 1. Index docs/ ──────────────────────────────────────────────────────────
section("STEP 1 — Index docs/")
result = index_docs_folder(force_reindex=True)
print(f"Status : {result['status']}")
print(f"Message: {result['message']}")

if result["status"] == "error":
    print(f"\n{YELLOW}Indexing error. Check that OPENAI_API_KEY is set in .env{RESET}")
    sys.exit(1)

# ── 2. List indexed documents ───────────────────────────────────────────
section("STEP 2 — Documents in the index")
docs = list_indexed_documents()
print(f"Total chunks: {docs['total_chunks']}")
for d in docs["documents"]:
    print(f"  • {d}")

# ── 3. Test searches ────────────────────────────────────────────────────
QUERIES = [
    "What does tip_radius_nm control?",
    "¿Cómo genero un dataset de prueba rápido?",
    "How to configure the AFM automation scan grid?",
    "What are the common AFM imaging artifacts?",
    "How does the triplet encoder training work?",
]

section("STEP 3 — Semantic searches")
for query in QUERIES:
    print(f"\n{BOLD}Question:{RESET} {query}")
    result = retrieve(query, n_results=2)
    if result["status"] == "error":
        print(f"  {YELLOW}Error: {result['message']}{RESET}")
        continue
    for i, r in enumerate(result["results"], 1):
        score_color = GREEN if r["score"] > 0.7 else YELLOW
        print(f"  [{i}] {score_color}score={r['score']:.3f}{RESET}  "
              f"src={r['source']}  heading={r['heading'] or '(no heading)'}")
        # Show only the first 2 lines of the chunk
        preview = r["text"].replace("\n", " ")[:120]
        print(f"       {CYAN}{preview}…{RESET}")

print(f"\n{GREEN}{BOLD}OK - Test RAG completado{RESET}\n")
