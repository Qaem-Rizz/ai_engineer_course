import os 
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

my_api_key= os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("Api key missing")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-20b"
role="user"
prompt="suggest me a name for my brand"
#giving  a system role to the LLM for more role specific responses
message_system={
    "role": "system",
    "content":"You are a brand manager who suggests name for my food company. Response should be in one word."
}

message={
    "role":role,
    "content": prompt
}

messages=[message_system , message]

#we can have the temperature range between 0-2 for most models out there.
response = client.chat.completions.create(model=model,messages=messages, temperature=2)

print(response)
print("_______________________________")
answer = response.choices[0].message.content
print(answer)