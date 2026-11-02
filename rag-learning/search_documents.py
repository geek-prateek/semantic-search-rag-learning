
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Use the SAME embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Open existing persistent database
client = chromadb.PersistentClient(path="./chroma_data")

# 3. Load existing collection
collection = client.get_collection("employee_knowledge")

# 4. Ask question
question = input("Ask your question: ")

# 5. Generate ONLY query embedding
query_embedding = model.encode(question).tolist()

# 6. Search existing vectors
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["documents", "distances", "metadatas"]
)

# 7. Display retrieved documents
print("\nTop matching documents:\n")

for document, distance, metadata in zip(
    results["documents"][0],
    results["distances"][0],
    results["metadatas"][0]
):
    print("Document:", document)
    print("Category:", metadata["category"])
    print("Cosine Distance:", round(distance, 4))
    print("Cosine Similarity:", round(1 - distance, 4))
    print("-" * 40)
