import numpy as np

# Two vectors pointing in the same direction
vector_a = np.array([1, 0])
vector_b = np.array([2, 0])

# Cosine similarity
similarity = np.dot(vector_a, vector_b) / (
    np.linalg.norm(vector_a) *
    np.linalg.norm(vector_b)
)

print("Cosine Similarity:", similarity)
