import numpy as np

def cosine_similarity(a, b):
    # Step 1: Dot product
    dot_product = np.dot(a, b)

    # Step 2: Magnitude
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    # Step 3: Handle zero vectors
    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors")

    # Step 4: Calculate similarity
    similarity = dot_product / (magnitude_a * magnitude_b)

    return similarity


vector_a = np.array([1, 0])

test_vectors = {
    "Same direction": np.array([5, 0]),
    "Perpendicular": np.array([0, 5]),
    "Opposite direction": np.array([-5, 0]),
    "Diagonal": np.array([1, 1])
}

for name, vector_b in test_vectors.items():
    score = cosine_similarity(vector_a, vector_b)
    print(f"{name}: {score:.4f}")
