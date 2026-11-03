from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Our knowledge base
documents = {
    "doc1": "Employees can request annual leave through HR portal.",
    "doc2": "Payroll is processed on the last working day.",
    "doc3": "Employees can reset passwords using the login page."
}

# Our evaluation dataset
test_cases = [
    {
        "question": "How can I apply for vacation?",
        "expected_id": "doc1"
    },
    {
        "question": "When will employees get paid?",
        "expected_id": "doc2"
    },
    {
        "question": "How do I recover my password?",
        "expected_id": "doc3"
    }
]

document_ids = list(documents.keys())
document_texts = list(documents.values())

# Generate document embeddings once
document_vectors = model.encode(
    document_texts,
    normalize_embeddings=True
)

correct = 0

for test in test_cases:
    question_vector = model.encode(
        test["question"],
        normalize_embeddings=True
    )

    # Dot product equals cosine similarity
    # for normalized non-zero vectors
    scores = np.dot(document_vectors, question_vector)

    best_index = int(np.argmax(scores))
    predicted_id = document_ids[best_index]

    passed = predicted_id == test["expected_id"]

    if passed:
        correct += 1

    print("\nQuestion:", test["question"])
    print("Expected:", test["expected_id"])
    print("Actual:", predicted_id)
    print("Passed:", passed)

total = len(test_cases)
pass_rate = correct / total

print("\nCorrect:", correct, "/", total)
print("Top-1 Accuracy:", round(pass_rate * 100, 2), "%")
