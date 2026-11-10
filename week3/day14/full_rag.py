import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer
import sys



model = SentenceTransformer('all-MiniLM-L6-v2')


load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API KEY MISSING")

client  = Groq(api_key=my_api_key)
groq_model= "openai/gpt-oss-120b"


documents = [
    "Employees receive 24 days of paid leave per year.",
   
    "Employees work from the office on Tuesday, Wednesday and Thursday. "
    "Monday and Friday are optional work-from-home days.",
   
    "Employees receive Rs 3000 per month for gym reimbursement.",
   
    "Employees can claim Rs 2000 per month for home internet.",
   
    "Employees have a 90 day notice period."
]

document_embeddings = model.encode(documents) #every line in the document variable is now converted using embedding

print(sys.getsizeof(document_embeddings)) #just to see how much size this small document context has


def cosine_similarity(a,b):
    cosine_sim= np.dot(a,b)/(
        np.linalg.norm(a) * np.linalg.norm(b)
    )
    return cosine_sim


def retrieve(q_embedding):
    scores =[]
    for i , document in enumerate(document_embeddings):
        score = cosine_similarity(q_embedding,document)
        scores.append((score,documents[i]))   #we are storing 2 things - one is the score using cosine similarity, second is the document vector whose score we are saving

    scores.sort(reverse=True)
    return scores[0] #so it will return 2 things


query =  input("Enter your question: ")
q_embedding= model.encode(query)  #embedding our query - so that comparison can happen in the vector DB


score,context = retrieve(q_embedding)
print(score)
print(context)


print("Calling the LLM now to reply as per the context:----------------------------------------\n")
print("\n")


def call_llm(query,context):
    
    sys_prompt = f"""answer in one line only and strictly based on this context, do not hallucinate. Context = {context}"""
    
    message_system={
        "role":"system",
        "content": sys_prompt
    }
    message={
        "role": "user",
        "content": query
    }

    messages=[message_system,message]
    response = client.chat.completions.create(model=groq_model, messages=messages)
    answer = response.choices[0].message.content
    return answer

answer = call_llm(query, context)

print(answer)


#limitations with this local  context embedding:
"""
firstly, this documents variable had only 6 lines of different sentences and still this takes around 128 bytes..
now imagine what happens if these 6 lines become 60 Lakh lines? the memory gets upto 122 MB

but what if, in practical RAG situations, the number of lines and embeddings gets to Crores? lets say 60 crore lines? the size gets huge , upto 11.92 GB

but sadly, our pc has only 8 GB of Total RAM ! is it scalable?? NO. thus came vector DBs... """