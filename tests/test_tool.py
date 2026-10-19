import os
from openai import OpenAI
from dotenv import load_dotenv

## Load CONSTANTs
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL   = os.getenv("OPENAI_MODEL")

from openai import OpenAI
import json

client = OpenAI()

def sumar(a, b):
    return a + b

tools = [{
    "type": "function",
    "name": "sumar",
    "description": "Adds two numbers",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"],
        "additionalProperties": False
    }
}]

response = client.responses.create(
    model=MODEL,
    input="cuanto es 4 mas 5",
    tools=tools,
    instructions = """
You are a math assistant.

Rules:
- ALWAYS use tools for every arithmetic operation.
- NEVER compute any arithmetic result yourself.
- If a task needs multiple operations, use multiple function calls.
- Only give the final answer after all required tool calls are completed.
"""
)

print("First output:", response.output)

for item in response.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)
        result = sumar(**args)

        response = client.responses.create(
            model=MODEL,
            previous_response_id=response.id,
            input=[{
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": str(result)
            }]
        )

print("Respuesta final:", response.output_text)