import logging

from preprocess import load_dataset, clean_and_prepare, create_chunks
from embed import generate_embeddings, encode_query
from retrieve import create_faiss_index, create_bm25_index, hybrid_search
from generate import generate_response


logging.basicConfig(
    filename="../logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    logging.info("RAG system started")

    #Load & preprocess
    df = load_dataset("../data/arxiv_ai.csv")
    df = df.head(200)   #Use only first 200 papers for now
    df = clean_and_prepare(df)
    chunks = create_chunks(df)

    #Embeddings
    embeddings = generate_embeddings(chunks)

    #Indexing
    faiss_index = create_faiss_index(embeddings)
    bm25_index = create_bm25_index(chunks)

    #User Query
    query = input("Enter your query: ")
    query_vector = encode_query(query)

    #Hybrid Retrieval
    results = hybrid_search(
        query,
        query_vector,
        chunks,
        faiss_index,
        bm25_index,
        top_k=5
    )

    #Generation
    answer = generate_response(query, results)

    print(answer)

    logging.info("RAG pipeline completed successfully")


if __name__ == "__main__":
    main()
