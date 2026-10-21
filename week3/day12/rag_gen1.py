import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("API KEY MISSING")

client = Groq(api_key=my_api_key)
model ="openai/gpt-oss-120b"

#Implementing RAG

#step 1: making a knowledge base: using a DICT here for now
context ={
    "age" :"the age of qaem is 22 years.",
    "hobbies": "the hobbies of qaem include reading islamic history and politics.",
    "skintone":"the skin tone of qaem is fair"
}


#step 2: doing the retrieval - replicating a simple coding if-else to tell the llm where to look into from
def retrieve(question):
    question = question.lower()
    
    if "age" in question:
        return context["age"]
    elif "hobbies" in question:
        return context["hobbies"]
    elif "skintone" in question:
        return context["skintone"]
    else:
        return None



# function to call the llm
def call_llm(question):

    retrieved_info = retrieve(question)
    
    sys_prompt = f"""answer in one line only and strictly based on this context, do not hallucinate. Context = {retrieved_info}"""
    
    message_system={
        "role":"system",
        "content": sys_prompt
    }
    message={
        "role": "user",
        "content": question
    }

    messages=[message_system,message]
    response = client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message.content
    return answer

user_prompt = "Do you know what is qaem's age?"

print(call_llm(user_prompt))


#the issue with this gen of RAG were a lot..to see one of them, check this prompt.... it won't show the result though we oly used a synonym for the word "age" :
prompt2 = "do you know how old is qaem?"

print(call_llm(prompt2))