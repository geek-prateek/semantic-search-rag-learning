from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Python is used for machine learning.",
    "Angular is a frontend framework.",
    "RAG retrieves relevant external knowledge.",
    "SQL Server is a relational database."
]

query = "How does RAG find information?"

document_vectors = model.encode(documents)
query_vector = model.encode([query])

scores = cosine_similarity(
    query_vector, document_vectors
)[0]

best_match = scores.argmax()

print("Best match:", documents[best_match])
print("Similarity:", scores[best_match])
