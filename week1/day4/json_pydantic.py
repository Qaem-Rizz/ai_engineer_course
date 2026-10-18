import os
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")


if not my_api_key:
    raise ValueError("API KEY MISSING")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-20b"
role="user"

#structuring the output recieved from LLM using system prompt role and pydantic , to get JSON structured response.
from pydantic import BaseModel
class Ticket(BaseModel):
    name:str
    email:str
    issue:str

schema=Ticket.model_json_schema()
response_format={
    "type" : "json_object"
}
system_prompt=f"""
extract the personal information from this Ticket strictly based on this schema and give a json output. {schema}
"""
message_system={
    "role":"system",
    "content":system_prompt
}



text=" Hey there ! my name is lucifer and i hate you bitch my ex dumped me last week and uspe se sala badluck itna kharab ke recently purchased Iphone stopped working too, wo bhi the last week. i'm from delhi and i can't find any emi claims here , my email is xyz@gmail.com and my phone number is 7761863459. kindly help me idk what to do nigga or imma sue ya'll."
prompt =f"""
this is a customer ticket. please extract the information from this . {text}
"""
message={
    "role": role,
    "content": prompt
}

messages = [message_system,message]

response = client.chat.completions.create(model=model,messages=messages,response_format=response_format)


print("______________________________________________")

answer = response.choices[0].message.content

#ab structured data ko pass krne ke liye json format me parse krenge, so  that further code me data pass kiya ja sake...
import json
raw_json=answer
data_file=json.loads(raw_json)
ticket=Ticket(**data_file)

print(ticket.name)
print(ticket.email)
print(ticket.issue)

