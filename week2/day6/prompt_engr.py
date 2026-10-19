import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API KEY MISSING")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-20b"

#making a function to call the LLM again and again and only change the prompt part to understand the depth and quality of responses
def llm_response(prompt):

    message={
        "role":"user",
        "content":prompt
    }

    messages=[message]
    response=client.chat.completions.create(model=model , messages=messages)
    answer = response.choices[0].message.content
    return answer

bad_prompt="""
This is a user complaint:
My laptop is not working.
handle this."""

print(llm_response(bad_prompt))


good_prompt="""
#ROLE: you are a customer support assistant at a mobile/laptop company.
#TASK: You have to classify the issue in a category.
#CONSTRAINT: You have to classify in one of the 3 categories namely billing, technical and return.
#OUTPUT FORMAT: your answer should be in one word only. The one word should be in the three categories given to you in constraints.
#EXAMPLE: For instance if a user says he wants a refund, then the category is "Return".
#FALLBACK: if the issue provided is unrelated to any of the three categories provided in constraints, then the response should be "OTHER".





this is a customer issue:
My girlfriend dumped me.
"""

print("_______________________________________________________________________")
print("NOW LOOK AT THE DIFFERENCE IN RESPONSE FOR A GOOD PROMPT:")
print("_______________________________________________________________________")

print(llm_response(good_prompt))