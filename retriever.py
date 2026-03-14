
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:

    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(documents)

        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query, top_k=2):
        query_embedding = self.model.encode([query])[0]
        dense_scores = np.dot(self.embeddings, query_embedding)
        sparse_scores = self.bm25.get_scores(query.split())
        combined_scores = dense_scores + sparse_scores

        ranked = sorted(
            zip(self.documents, combined_scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, score in ranked[:top_k]]
