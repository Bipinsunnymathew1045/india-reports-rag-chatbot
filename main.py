import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import numpy as np
load_dotenv(".chatenv")
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed some sentences
sentences = [
    "The cat sat on the mat",
    "A feline rested on the rug",   # similar meaning to above
    "I love eating pasta",           # unrelated
]

vectors = embedder.embed_documents(sentences)

# Look at what you got
print(f"Number of vectors: {len(vectors)}")
print(f"Each vector has {len(vectors[0])} dimensions")
print(f"First 5 numbers of sentence 1: {vectors[0][:5]}")



def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

sim_1_2 = cosine_similarity(vectors[0], vectors[1])
sim_1_3 = cosine_similarity(vectors[0], vectors[2])

print(f"\nSimilarity between 'cat on mat' and 'feline on rug': {sim_1_2:.4f}")
print(f"Similarity between 'cat on mat' and 'I love pasta': {sim_1_3:.4f}")
print(f"\nSentence 1&2 are {'MORE' if sim_1_2 > sim_1_3 else 'LESS'} similar than 1&3")