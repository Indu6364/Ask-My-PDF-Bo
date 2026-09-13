from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

text = """
Legal Service Agreement

Payment Terms:
The client shall pay ₹50,000 upon signing this agreement.

Termination Clause:
Either party may terminate this agreement with 30 days written notice.

Confidentiality:
The confidentiality clause remains valid for 5 years after termination.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

print("Chunks:")

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

print(f"\nStored {index.ntotal} chunks in FAISS")