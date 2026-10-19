import os
from openai import OpenAI
from dotenv import load_dotenv
import json

## Load CONSTANTs
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL   = os.getenv("OPENAI_MODEL")

client = OpenAI()

from openai import OpenAI
import json

client = OpenAI()

# =========================
# 🧰 TOOLS (real functions)
# =========================

def sumar(a, b):
    return a + b

def multiplicar(a, b):
    return a * b

def dividir(a, b):
    if b == 0:
        return "Error: division by zero"
    return a / b

# =========================
# 🧠 TOOL DEFINITIONS
# =========================

tools = [
    {
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
    },
    {
        "type": "function",
        "name": "multiplicar",
        "description": "Multiplies two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "dividir",
        "description": "Divides two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"],
            "additionalProperties": False
        }
    }
]

# =========================
# 💬 CHAT LOOP
# =========================

while True:
    user_input = input("\nUser: ")

    response = client.responses.create(
        model=MODEL,
        input=user_input,
        tools=tools,
        instructions="""
        You are a math assistant.
        Use tools when needed.
        Explain briefly what you are doing.
        """
    )

    steps = 0
    max_steps = 5  # avoid infinite loops

    # =========================
    # 🔁 INNER LOOP (chaining)
    # =========================

    while True:
        steps += 1

        if steps > max_steps:
            print("Assistant: Too many steps, stopping.")
            break

        tool_called = False

        for item in response.output:

            # 🧰 If the model wants to use a tool
            if item.type == "function_call":
                tool_called = True

                tool_name = item.name
                args = json.loads(item.arguments)

                print(f"[DEBUG] Calling {tool_name} with {args}")

                # Run the tool
                if tool_name == "sumar":
                    result = sumar(**args)
                elif tool_name == "multiplicar":
                    result = multiplicar(**args)
                elif tool_name == "dividir":
                    result = dividir(**args)

                # Second call → return the result to the model
                response = client.responses.create(
                    model=MODEL,
                    previous_response_id=response.id,
                    input=[{
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": str(result)
                    }]
                )

                break  # important → process step by step

        # 🗣️ If there is no tool call → final answer
        if not tool_called:
            print("Assistant:", response.output_text)
            break