
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Create a persistent vector database
client = chromadb.PersistentClient(path="./chroma_data")

# 3. Create or load collection
collection = client.get_or_create_collection(
    name="employee_knowledge",
    configuration={"hnsw": {"space": "cosine"}}
)

# 4. Our sample documents
documents = [
    "Employees can request annual leave through the HR portal.",
    "Payroll is processed on the last working day of the month.",
    "Employees can reset passwords through the login page.",
    "Python is commonly used for machine learning.",
    "Angular is a frontend web development framework."
]

ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

metadata = [
    {"category": "HR"},
    {"category": "Payroll"},
    {"category": "Security"},
    {"category": "AI"},
    {"category": "Development"}
]

# 5. Generate embeddings
embeddings = model.encode(documents).tolist()

# 6. Save to ChromaDB
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadata
)

print("Documents stored successfully!")
print("Total documents:", collection.count())
