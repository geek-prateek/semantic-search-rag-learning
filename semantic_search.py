
from sentence_transformers import SentenceTransformer
import numpy as np

# Step 1: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Our knowledge base
documents = [
    "Employees can request annual leave through the HR portal.",
    "Python is commonly used for AI and machine learning.",
    "Payroll is processed on the last working day.",
    "Employees can reset passwords through the login page.",
    "Angular is a frontend framework."
]

# Step 3: Convert documents into embeddings
document_embeddings = model.encode(documents)

# Step 4: Get user's question
question = input("Ask your question: ")

# Step 5: Convert question into embedding
question_embedding = model.encode(question)

# Step 6: Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

scores = []

for embedding in document_embeddings:
    score = cosine_similarity(question_embedding, embedding)
    scores.append(score)

# Step 7: Find the best matching document
best_index = np.argmax(scores)

print("\nMost Relevant Document:")
print(documents[best_index])
print("Similarity Score:", round(float(scores[best_index]), 4))
