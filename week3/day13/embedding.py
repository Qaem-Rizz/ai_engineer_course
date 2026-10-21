import os
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



model = SentenceTransformer("all-miniLM-L6-v2")
text = "A demo sentence to see whether the sentences are being converted into vectors or not."

embedding= model.encode(text) #to embedd the sentence

print(embedding.shape) #to show dimensions of array
print(embedding) #to show the array made

print("_______________________________________________________")

# now just to see how 2 different sentences are converted into vector and what the cosine similarity looks between them, 2 ways are there:
# 1. using numpy and writing a function:

sentence1 = "I love you riya."
sentence2 = "Mars is a planet."

v1=model.encode(sentence1)
v2=model.encode(sentence2)


# def cosine_similarity(a,b):
#     cosine_sim= np.dot(a,b)/(
#         np.linalg.norm(a) * np.linalg.norm(b)
#     )
#     return cosine_sim

# print(cosine_similarity(v1,v2))






# 2. using the Scikit-learn inbuilt module of cosine similariy:

similariy_of_s1_s2= cosine_similarity([v1],[v2])

print("the cosine similarity is: " ,similariy_of_s1_s2)

