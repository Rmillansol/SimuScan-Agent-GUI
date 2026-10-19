import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
client = OpenAI()

response = client.responses.create(
    model=MODEL,
    input="cuanto es 4 mas 5",
    instructions="""
You are a math assistant.

Rules:
- ALWAYS use tools for every arithmetic operation.
- NEVER compute any arithmetic result yourself.
- Only give the final answer after the MCP tool is used.
""",
    tools=[
        {
            "type": "mcp",
            "server_label": "math",
            "server_url": "http://127.0.0.1:8000/mcp",
            "require_approval": "never",
            "allowed_tools": ["sumar"],
        }
    ],
)

print(response.output_text)