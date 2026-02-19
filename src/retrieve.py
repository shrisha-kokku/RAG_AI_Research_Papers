import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
import logging


#VECTOR INDEX

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    logging.info("FAISS index created")
    return index


def vector_search(index, query_vector, top_k):
    distances, indices = index.search(query_vector, len(query_vector))
    return distances[0], indices[0]


#BM25

def create_bm25_index(chunks):
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    logging.info("BM25 index created")
    return bm25


def bm25_search(query, bm25):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    return scores


#HYBRID

def hybrid_search(query, query_vector, chunks, index, bm25, top_k=5, alpha=0.6):
    try:
        # Vector scores
        distances, vec_indices = index.search(query_vector, len(chunks))
        vec_scores = 1 / (1 + distances[0])

        # BM25 scores
        bm25_scores = bm25_search(query, bm25)

        # Normalize both
        scaler = MinMaxScaler()

        vec_scores_norm = scaler.fit_transform(vec_scores.reshape(-1, 1)).flatten()
        bm25_scores_norm = scaler.fit_transform(np.array(bm25_scores).reshape(-1, 1)).flatten()

        # Combine
        combined_scores = alpha * vec_scores_norm + (1 - alpha) * bm25_scores_norm

        # Rank
        ranked_indices = np.argsort(combined_scores)[::-1][:top_k]

        results = [chunks[i] for i in ranked_indices]

        logging.info("Hybrid search completed")
        return results

    except Exception as e:
        logging.error(f"Hybrid search failed: {e}")
        raise
