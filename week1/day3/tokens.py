import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("Api key missing")

client=Groq(api_key=my_api_key)

model="openai/gpt-oss-20b"
role="user"
prompt1="hey there"
prompt2="tell me in detail about history of computers"
prompt3="write me an essay on the topic of Ai engineer in 500 words"

prompts=[prompt1,prompt2,prompt3]

for prompt in prompts:
    message={
        "role":role,
        "content": prompt
    }
    messages=[message]
    response = client.chat.completions.create(model=model,messages=messages,max_tokens=100)
    usage= response.usage
    print(f"Prompt: {prompt} --> your tokens: {usage.prompt_tokens} completion tokens: {usage.completion_tokens} total tokens: {usage.total_tokens} Finish reason : {response.choices[0].finish_reason}")

#if the response is naturally completed then the finish reason will be "STOP" , but if the tokens length is exceeded, the reason is "LENGTH"