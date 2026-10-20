import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API KEY NOT FOUND")

client = Groq(api_key=my_api_key)
model="openai/gpt-oss-120b"


prompt = """
explain to me the working of Internet in 500 words.
"""

message={
    "role": "user",
    "content": prompt
}

messages=[message]

# response  = client.chat.completions.create(model=model, messages=messages)
# answer = response.choices[0].message.content
# print(answer)
#REMEMBER THAT STREAMING IS "FALSE" BY DEFAULT





print("SEE HOW RESPONSE IS RECEIVED IN STREAMING------------------------------------------------")
stream = client.chat.completions.create(model=model, messages=messages, stream=True)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content :
        print(content , end="", flush=True)
