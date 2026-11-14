import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


load_dotenv()

# importing all the API keys:

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


#connecting to qdrant:

client = QdrantClient(
    url = QDRANT_URL,
    api_key = QDRANT_API_KEY
)

print("Connected to Qdrant Cloud")


# creating a Qdrant collection-----------------------------------------------

COLLECTION_NAME = "knowledge"
EMBEDDING_SIZE = 384

#step 1: delete collection if it already exists:

if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection {COLLECTION_NAME}")

    client.delete_collection(COLLECTION_NAME)

#step 2: create collection:

client.create_collection(
    collection_name = COLLECTION_NAME,
    vectors_config = VectorParams(
        size = EMBEDDING_SIZE,
        distance= Distance.COSINE,
    ),
)


print(f"created collection {COLLECTION_NAME}")
print(f"size : {EMBEDDING_SIZE}")
print("distance:  COSINE")



#step 3: load our knowledge base from text file to the vector DB:

with open("knowledge.txt" , "r" , encoding = "utf-8") as f:
    documents =[
        line.strip()
        for line in f
        if line.strip()
        ]
print(f"loaded {len(documents)} docuemnts")




#step 4: making their embeddings in the vector DB:

print("Loading embedding model....")

model = SentenceTransformer('all-MiniLM-L6-v2')

print("Embedding model ready....")

embeddings = model.encode(documents)

print(f"Generated {len(embeddings)} embeddings...")

print(f"embedding size is {len(embeddings[0])}")

#step 5: creating QDRANT points:

points= []

for i , embedding in enumerate(embeddings):
    point = PointStruct(
        id = i +1,
        vector = embedding.tolist(),
        payload = {
            "text" : documents[i]
        }
    )

    points.append(point)

#step 6: upload to QDRANT:

client.upsert(
    collection_name = COLLECTION_NAME,
    points= points
) #upload +insert


print(f"uploaded {len(points)} documents to qdrant")


#step 7: search from QDRANT


def search(query, top_k=3):

    #converting our query to vector embedding :
    query_vector = model.encode(query).tolist()


    #search Qdrant for similar vectors
    results = client.query_points(
        collection_name = COLLECTION_NAME,
        query = query_vector,
        limit = top_k,
        with_payload = True,
    ).points

    return results

#step 8: Test search

query = input("Enter your query here for the embedding answer: ")

results = search(query, top_k=3)


print("\nSearch results:")

for result in results:
    print(f"result score :  {result.score:.3f}")
    print(result.payload ["text"])
    print()


#step 9: Connect to GROQ:
groq_model= "openai/gpt-oss-120b"

groq_client = Groq(
    api_key = GROQ_API_KEY

)


#step 10: ask the llm

def ask_llm(question,context):
    
    sys_prompt = f"""answer in one line only and strictly based on this context, do not hallucinate. Context = {context}"""
    
    message_system={
        "role":"system",
        "content": sys_prompt
    }
    message={
        "role": "user",
        "content": question
    }

    messages=[message_system,message]
    response = groq_client.chat.completions.create(model=groq_model, messages=messages)
    answer = response.choices[0].message.content
    return answer



# FULL RAG PIPELINE-------------------------------------
question = input("enter you question for the LLM:  ")

results = search (question, top_k =3)


#extract text from the search results in vector DB for LLM to read:
context = "\n".join(
    result.payload["text"]
    for result in results
)







answer = ask_llm(question, context)

print("answer from the LLM:")

print(answer)







