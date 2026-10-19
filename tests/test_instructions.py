import os
from openai import OpenAI
from dotenv import load_dotenv

## Load CONSTANTs
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL   = os.getenv("OPENAI_MODEL")

# Conect Openai
client = OpenAI(api_key=API_KEY)

response = client.responses.create(
    model=MODEL,
    reasoning={"effort": "low"},
    instructions="Talk like a pirate.",
    input="Are semicolons optional in JavaScript?",
)

print(response.output_text)