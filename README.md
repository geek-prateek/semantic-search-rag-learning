
# 🚀 Semantic Search & RAG Learning

A hands-on Python project to understand **Embeddings, Semantic Search, Cosine Similarity, Vector Databases, and Retrieval-Augmented Generation (RAG)** from scratch.

The goal is to understand how each component works internally and gradually build a complete document-based RAG application.

---

## 📌 1. What is RAG?

**RAG (Retrieval-Augmented Generation)** is an approach that combines information retrieval with Large Language Models (LLMs).

Instead of relying only on the LLM's existing knowledge, RAG retrieves relevant information from external documents and provides it as context for generating answers.

### Complete RAG Flow

```text
                  DOCUMENT PROCESSING
                          |
                          v
                 Upload Documents
                          |
                          v
                   Extract Text
                          |
                          v
                  Split into Chunks
                          |
                          v
                  Generate Embeddings
                          |
                          v
                    Vector Database
                          |
                          |
                    USER QUESTION
                          |
                          v
                  Generate Embedding
                          |
                          v
                 Similarity Search
                          |
                          v
                Retrieve Top-K Chunks
                          |
                          v
                   Context + Query
                          |
                          v
                          LLM
                          |
                          v
                   Final Answer
```

**Important:** RAG has two main processes:

1. **Indexing:** Process documents and store their embeddings.
2. **Retrieval & Generation:** Search relevant information and generate an answer.

---

## 🧠 2. What are Embeddings?

An embedding is a numerical vector representation of text.

Embedding models convert text into numbers so that computers can compare semantic relationships between sentences.

### Example

```text
Sentence A: "I love programming."
Sentence B: "I enjoy coding."
Sentence C: "I like eating pizza."
```

Sentence A and Sentence B have similar meanings even though they use different words.

An embedding model can represent them as semantically similar vectors.

### Example Embeddings

```text
"I love programming"
        |
        v
[0.82, 0.14, -0.31, 0.67, ...]

"I enjoy coding"
        |
        v
[0.79, 0.18, -0.29, 0.71, ...]
```

These values are illustrative, not actual model outputs.

### Python Example

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence = "I love Python programming"

embedding = model.encode(sentence)

print("Dimensions:", len(embedding))
print("Embedding:", embedding[:10])
```

The `all-MiniLM-L6-v2` model generates **384-dimensional embeddings**.

### Key Points

- Embeddings represent semantic information numerically.
- Similar sentences generally produce similar vector representations.
- Embeddings are used for semantic search.
- Different embedding models can produce different vector dimensions.
- Documents and queries should use compatible embedding models.

---

## 🔎 3. What is Semantic Search?

Semantic Search retrieves information based on **meaning rather than exact keyword matching**.

### Example

**Stored document:**

```text
Employees can request annual leave through the HR portal.
```

**User question:**

```text
How can staff apply for vacation?
```

Even though the exact words are different, both sentences describe a similar concept.

```text
Annual Leave ≈ Vacation
Employees    ≈ Staff
Request      ≈ Apply
```

A semantic search engine can identify this relationship using embeddings.

### Semantic Search Flow

```text
               Documents
                   |
                   v
             Embedding Model
                   |
                   v
            Document Vectors
                   |
                   |
              User Query
                   |
                   v
             Embedding Model
                   |
                   v
              Query Vector
                   |
                   v
           Similarity Calculation
                   |
                   v
           Most Relevant Document
```

### Python Example

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Python is used for machine learning.",
    "JavaScript is used for web development.",
    "Employees can request annual leave.",
    "Payroll is processed every month."
]

query = "How can I build AI applications?"

document_embeddings = model.encode(
    documents,
    convert_to_tensor=True
)

query_embedding = model.encode(
    [query],
    convert_to_tensor=True
)

results = semantic_search(
    query_embedding,
    document_embeddings,
    top_k=2
)

for result in results[0]:
    print("Document:", documents[result["corpus_id"]])
    print("Score:", result["score"])
```

### Keyword Search vs Semantic Search

| Keyword Search | Semantic Search |
|---|---|
| Matches words | Compares semantic meaning |
| Exact terms are important | Different wording can still match |
| Useful for exact identifiers | Useful for natural-language questions |
| Example: `vacation` | Can retrieve `annual leave` |

---

## 📐 4. What is Cosine Similarity?

Cosine Similarity measures how similar two non-zero vectors are based on the **angle between them**.

It focuses on vector direction rather than magnitude.

### Formula

$$
\text{Cosine Similarity}(A,B)
=
\frac{A \cdot B}{\|A\| \times \|B\|}
$$

Where:

- `A · B` = Dot product
- `||A||` = Magnitude of vector A
- `||B||` = Magnitude of vector B

### Similarity Values

| Cosine Similarity | Mathematical Meaning |
|---|---|
| 1 | Same direction |
| 0 | Perpendicular |
| -1 | Opposite direction |

### Manual Calculation

Given:

```text
A = [3, 4]
B = [6, 8]
```

**Step 1: Dot Product**

```text
(3 × 6) + (4 × 8)
= 18 + 32
= 50
```

**Step 2: Magnitude**

```text
Magnitude A = √(3² + 4²) = 5

Magnitude B = √(6² + 8²) = 10
```

**Step 3: Similarity**

```text
50 / (5 × 10)
= 50 / 50
= 1.0
```

Both vectors point in the same direction.

### Python Implementation

```python
import numpy as np

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)

    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Zero vectors are not supported")

    return dot_product / (magnitude_a * magnitude_b)


A = np.array([3, 4])
B = np.array([6, 8])

print(cosine_similarity(A, B))
```

**Output:**

```text
1.0
```

### Why is Cosine Similarity useful in RAG?

1. Documents are converted into embeddings.
2. The user query is converted into an embedding.
3. Query and document vectors are compared.
4. The most relevant documents are retrieved.

Cosine Similarity is one common retrieval metric. Other options include dot product and Euclidean distance.

---

## 🗄️ 5. What is a Vector Database?

A vector database stores and retrieves embeddings efficiently.

Instead of generating document embeddings for every user question, we generate them during document indexing and save them.

### Without a Vector Database

```text
User Question
      |
      v
Generate All Document Embeddings Again
      |
      v
Compare All Embeddings
      |
      v
Return Results
```

This repeats unnecessary computation when the documents have not changed.

### With a Vector Database

```text
INDEXING (When Documents Are Added/Updated)

Documents
    |
    v
Generate Embeddings
    |
    v
Store in Vector Database


SEARCHING (For Every User Question)

User Question
    |
    v
Generate Query Embedding
    |
    v
Search Stored Vectors
    |
    v
Return Relevant Documents
```

### Benefits

- Avoids repeatedly generating unchanged document embeddings.
- Supports persistent storage.
- Supports semantic similarity search.
- Supports metadata filtering.
- Enables scalable retrieval architectures.

---

## 🟢 6. ChromaDB

**ChromaDB** is an open-source vector database used for storing and retrieving embeddings.

A ChromaDB record can contain:

```text
ID:
    doc1

Document:
    Employees can request annual leave.

Embedding:
    [0.12, -0.34, 0.56, ...]

Metadata:
    {
        "category": "HR",
        "year": 2026
    }
```

### Important ChromaDB Concepts

| Concept | Description |
|---|---|
| Collection | Logical group of vector records |
| ID | Unique identifier within a collection |
| Document | Original stored text |
| Embedding | Numerical vector |
| Metadata | Additional structured information |
| add() | Insert new records |
| upsert() | Insert or update records |
| query() | Search relevant vectors |
| where | Filter using metadata |

### Create a Persistent Collection

```python
import chromadb

client = chromadb.PersistentClient(
    path="./chroma_data"
)

collection = client.get_or_create_collection(
    name="employee_knowledge",
    configuration={"hnsw": {"space": "cosine"}}
)
```

The `chroma_data` directory stores the persistent database.

### Insert Documents

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Employees can request annual leave.",
    "Payroll is processed every month."
]

embeddings = model.encode(documents).tolist()

collection.upsert(
    ids=["doc1", "doc2"],
    documents=documents,
    embeddings=embeddings,
    metadatas=[
        {"category": "HR", "year": 2026},
        {"category": "Payroll", "year": 2026}
    ]
)
```

### Search Documents

```python
question = "When is employee payroll processed?"

query_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,
    include=["documents", "distances", "metadatas"]
)

print(results["documents"])
```

### Metadata Filtering

Search only Payroll documents:

```python
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,
    where={"category": "Payroll"}
)
```

Search Payroll documents from 2026:

```python
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,
    where={
        "$and": [
            {"category": "Payroll"},
            {"year": 2026}
        ]
    }
)
```

### Cosine Distance vs Cosine Similarity

When a collection uses cosine distance:

$$
\text{Cosine Distance} = 1 - \text{Cosine Similarity}
$$

| Similarity | Distance |
|---|---|
| 0.95 | 0.05 |
| 0.80 | 0.20 |
| 0.30 | 0.70 |

For cosine similarity, higher means more directionally similar.

For cosine distance, lower means more directionally similar.

---

## 🔄 7. add() vs upsert()

### add()

Used to insert new records.

```python
collection.add(
    ids=["doc1"],
    documents=["Original employee policy"],
    embeddings=[embedding]
)
```

An existing ID is not overwritten by `add()`.

### upsert()

Used to insert new records or update existing records.

```python
collection.upsert(
    ids=["doc1"],
    documents=["Updated employee policy"],
    embeddings=[updated_embedding]
)
```

**Important:** If document text changes, its embedding should also be updated.

Use stable IDs to avoid duplicate records.

---

## 📁 8. Project Structure

```text
semantic-search-rag-learning/
│
├── README.md
│
├── vector_calc.py
│   └── Basic vector calculations
│
├── day1_cosine_similarity.py
│   └── Cosine similarity implementation
│
├── day1_semantic_search.py
│   └── Semantic search using embeddings
│
├── semantic_search.py
│   └── Embedding-based text retrieval
│
├── semantic_search_engine.py
│   └── Semantic search engine practice
│
└── rag-learning/
    └── Additional RAG practice files
```

Suggested ChromaDB files to add:

```text
rag-learning/
├── index_documents.py
├── search_documents.py
└── chroma_metadata_practice.py
```

The files above represent the current learning exercises and suggested organization. The final structure will evolve as additional components are implemented.

---

## ⚙️ 9. Environment Setup

### Requirements

- Python 3.10+
- pip
- VS Code or another Python editor

### Clone Repository

```bash
git clone https://github.com/geek-prateek/semantic-search-rag-learning.git

cd semantic-search-rag-learning
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment (Windows)

```powershell
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install numpy sentence-transformers chromadb
```

### Run Examples

```bash
python vector_calc.py
```

```bash
python day1_cosine_similarity.py
```

```bash
python day1_semantic_search.py
```

For ChromaDB exercises, run the indexing script before searching.

The embedding model may be downloaded on the first execution.

---

## 🔍 10. Semantic Search vs RAG

Semantic Search and RAG are related, but not identical.

### Semantic Search

```text
User Question
      |
      v
Generate Query Embedding
      |
      v
Search Document Embeddings
      |
      v
Return Relevant Documents
```

### RAG

```text
User Question
      |
      v
Generate Query Embedding
      |
      v
Retrieve Relevant Document Chunks
      |
      v
Build Prompt Using Retrieved Context
      |
      v
Send Prompt to LLM
      |
      v
Generate Answer
```

**Semantic Search retrieves relevant information.**

**RAG uses retrieved information to support answer generation.**

---

## ⚠️ 11. Common Mistakes

### Mistake 1: Regenerating all embeddings for each query

**Problem:** Unnecessary computation and high latency.

**Solution:** Store document embeddings and regenerate only when documents change.

### Mistake 2: Using different embedding models

**Problem:** Document vectors and query vectors may be incompatible.

**Solution:** Use the same or explicitly compatible embedding models.

### Mistake 3: Confusing cosine distance and similarity

**Problem:** Ranking interpretation may become incorrect.

**Solution:** Check which distance metric the collection uses.

### Mistake 4: Incorrect upsert ID

**Problem:** A new record is inserted instead of updating the existing record.

**Solution:** Reuse the correct existing ID.

### Mistake 5: Incorrect metadata filter

**Problem:** Relevant documents can be excluded, or unwanted documents included.

**Solution:** Apply correct filters for category, year, document source, and authorization scope.

---

## 🛣️ 12. Learning Roadmap

### Phase 1 — Foundations

- [x] Understand embeddings
- [x] Understand semantic similarity
- [x] Understand cosine similarity
- [x] Implement cosine similarity using Python
- [x] Implement basic semantic search
- [x] Understand vector database concepts
- [x] Learn ChromaDB collections, IDs, and metadata
- [ ] Complete ChromaDB persistence and filtering experiments

### Phase 2 — Document Processing

- [ ] PDF text extraction
- [ ] Document chunking
- [ ] Chunk size and overlap
- [ ] Document metadata and source tracking
- [ ] Store document chunks in a vector database

### Phase 3 — Complete RAG Pipeline

- [ ] Retrieve relevant document chunks
- [ ] Build context-aware prompts
- [ ] Integrate an LLM
- [ ] Generate answers using retrieved context
- [ ] Add source references

### Phase 4 — Advanced RAG

- [ ] Hybrid search
- [ ] Reranking
- [ ] Query rewriting
- [ ] Conversational retrieval
- [ ] Multi-document retrieval

### Phase 5 — Production

- [ ] FastAPI backend
- [ ] Document upload API
- [ ] Authentication and authorization
- [ ] Background indexing
- [ ] RAG evaluation
- [ ] Docker deployment
- [ ] Performance monitoring

---

## ✅ 13. Key Takeaways

**Embeddings:** Convert text into numerical vectors.

**Semantic Similarity:** Compare the meaning of sentences.

**Cosine Similarity:** Measure directional similarity between vectors.

**Vector Database:** Store and retrieve embeddings.

**ChromaDB Collection:** Organize stored vector records.

**Metadata Filtering:** Restrict which documents can participate in retrieval.

**Semantic Search:** Find relevant information.

**RAG:** Retrieve relevant information and use it to generate grounded answers.

---

## 🎯 Project Goal

Build a complete Python-based RAG system while understanding every stage of the architecture.

```text
Python
   |
   v
Embeddings
   |
   v
Semantic Search
   |
   v
Cosine Similarity
   |
   v
Vector Database
   |
   v
Document Processing
   |
   v
Retrieval
   |
   v
LLM Integration
   |
   v
Complete RAG Application
```

The project is developed incrementally, with each component implemented, tested, and understood before introducing the next.

---

**Happy Learning! 🚀**
