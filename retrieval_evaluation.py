from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = {
    "doc1": "Employees can request annual leave through the HR portal.",
    "doc2": "Payroll is processed on the last working day.",
    "doc3": "Employees can reset passwords through the login page.",
    "doc4": "Annual leave requests require manager approval.",
    "doc5": "Python is used for machine learning."
}

# Each query has all known relevant document IDs
test_cases = [
    {
        "question": "How do employees apply for vacation?",
        "relevant_ids": ["doc1", "doc4"]
    },
    {
        "question": "When do employees get paid?",
        "relevant_ids": ["doc2"]
    },
    {
        "question": "How can I recover my password?",
        "relevant_ids": ["doc3"]
    }
]

doc_ids = list(documents.keys())
doc_texts = list(documents.values())

# Generate document embeddings once
doc_vectors = model.encode(
    doc_texts,
    normalize_embeddings=True
)

def evaluate_retrieval(ranked_ids, relevant_ids, k):
    top_k = ranked_ids[:k]
    relevant = set(relevant_ids)

    hits = len(set(top_k) & relevant)

    recall = hits / len(relevant) if relevant else 0
    precision = hits / k

    reciprocal_rank = 0

    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            reciprocal_rank = 1 / rank
            break

    return recall, precision, reciprocal_rank


k = 3
all_results = []

for test in test_cases:
    query_vector = model.encode(
        test["question"],
        normalize_embeddings=True
    )

    scores = np.dot(doc_vectors, query_vector)

    ranked_indices = np.argsort(scores)[::-1]
    ranked_ids = [doc_ids[i] for i in ranked_indices]

    recall, precision, rr = evaluate_retrieval(
        ranked_ids,
        test["relevant_ids"],
        k
    )

    all_results.append((recall, precision, rr))

    print("\nQuestion:", test["question"])
    print("Top-K:", ranked_ids[:k])
    print("Recall@3:", round(recall, 3))
    print("Precision@3:", round(precision, 3))
    print("Reciprocal Rank:", round(rr, 3))


average = np.mean(all_results, axis=0)

print("\n--- Final Evaluation ---")
print("Mean Recall@3:", round(average[0], 3))
print("Mean Precision@3:", round(average[1], 3))
print("MRR:", round(average[2], 3))
