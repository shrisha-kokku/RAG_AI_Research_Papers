import pandas as pd
import logging


def load_dataset(path):
    try:
        df = pd.read_csv(path)
        logging.info("Dataset loaded successfully")
        return df
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise


def clean_and_prepare(df):
    df = df.dropna(subset=["title", "summary"])
    df["text"] = df["title"] + " " + df["summary"]
    return df



def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks


def create_chunks(df):
    all_chunks = []

    for text in df["text"]:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    logging.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks
