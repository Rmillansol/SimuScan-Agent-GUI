from pydantic import BaseModel
import os
from openai import OpenAI
from dotenv import load_dotenv

## Load CONSTANTs
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL   = os.getenv("OPENAI_MODEL")


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


# Conect Openai
client = OpenAI(api_key=API_KEY)

response = client.responses.parse(
    model="gpt-4o-2024-08-06",
    input=[
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ],
    text_format=CalendarEvent,
)

event = response.output_parsed

print(event)
print(response.output_text)