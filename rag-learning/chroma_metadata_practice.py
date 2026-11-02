
import chromadb
from sentence_transformers import SentenceTransformer

# Step 1: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Persistent ChromaDB
client = chromadb.PersistentClient(path="./chroma_data")

# Separate collection for this experiment
collection = client.get_or_create_collection(
    name="metadata_practice",
    configuration={"hnsw": {"space": "cosine"}}
)

# Step 3: Prepare documents
documents = [
    "Employees can apply for annual leave through the HR portal.",
    "Monthly payroll is processed on the last working day.",
    "Employee tax is calculated during payroll processing."
]

ids = ["doc1", "doc2", "doc3"]

metadatas = [
    {"category": "HR", "year": 2026},
    {"category": "Payroll", "year": 2026},
    {"category": "Payroll", "year": 2025}
]

# Step 4: Generate embeddings
embeddings = model.encode(documents).tolist()

# Step 5: Insert or update records
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("Total records:", collection.count())

# Step 6: Update one existing document
updated_text = (
    "Employees can apply for annual leave "
    "through the employee self-service portal."
)

collection.upsert(
    ids=["doc1"],
    documents=[updated_text],
    embeddings=[model.encode(updated_text).tolist()],
    metadatas=[{"category": "HR", "year": 2026}]
)

# Step 7: Query only Payroll documents
question = "How is employee payroll processed?"

query_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,
    where={"category": "Payroll"},
    include=["documents", "metadatas", "distances"]
)

for doc, meta, distance in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print("\nDocument:", doc)
    print("Metadata:", meta)
    print("Cosine similarity:", round(1 - distance, 4))
