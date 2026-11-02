
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

# Step 1: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Documents
documents = [
    "Python is a programming language used for AI.",
    "JavaScript is used for web development.",
    "Machine learning systems can be built using Python.",
    "The weather is very hot today."
]

# Step 3: User question
query = "Which language helps create machine learning apps?"

# Step 4: Generate embeddings
document_embeddings = model.encode(
    documents,
    convert_to_tensor=True
)

query_embedding = model.encode(
    [query],
    convert_to_tensor=True
)

# Step 5: Find top 3 relevant documents
results = semantic_search(
    query_embedding,
    document_embeddings,
    top_k=3
)

# Step 6: Display results
for result in results[0]:
    document_id = result["corpus_id"]
    score = result["score"]

    print("Document:", documents[document_id])
    print("Score:", round(score, 4))
    print("------------------------")
