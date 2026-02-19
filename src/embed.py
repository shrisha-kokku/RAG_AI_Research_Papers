from sentence_transformers import SentenceTransformer
import logging

model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks):
    try:
        embeddings = model.encode(chunks)
        logging.info("Embeddings generated successfully")
        return embeddings
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        raise


def encode_query(query):
    return model.encode([query])
