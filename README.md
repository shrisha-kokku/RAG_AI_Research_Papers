# RAG System for AI Research Papers

This project implements a Retrieval-Augmented Generation (RAG) system that helps you search and get answers from AI research papers. It combines traditional keyword search with modern vector-based semantic search to find the most relevant papers and generate accurate answers.

## What This Project Does?

The system takes a question about AI research papers and:
1. Searches through the dataset using both keyword matching (BM25) and semantic similarity (vector search)
2. Combines both search results to find the best matches
3. Uses an AI model to generate a clear answer based on the retrieved papers

## Dataset

We use the **AI Research Papers Dataset** from Kaggle. This dataset contains research papers from ArXiv with titles, summaries, authors, and other metadata.

**Dataset Link:** [AI Research Papers Dataset on Kaggle](https://www.kaggle.com/datasets/yasirabdaali/arxivorg-ai-research-papers-dataset)

**Note:** Download the dataset and place the CSV file in the `data/` folder as `arxiv_ai.csv`

## Tech Stack

- **Python** - Main programming language
- **FAISS** - Vector database for fast similarity search
- **Sentence Transformers** - For generating embeddings (using `all-MiniLM-L6-v2` model)
- **BM25** - Traditional keyword-based search
- **Groq API** - LLM for generating responses (using LLaMA 3.1 8B Instant)
- **Pandas** - Data processing

## Prerequisites

Make sure you have:
- Python 3.8 or higher installed
- A Groq API key (free to get, see steps below)
- The dataset downloaded from Kaggle

## Installation Steps

### Step 1: Clone or Download This Repository

If you have the code, you're all set. If not, make sure all the project files are in one folder.

### Step 2: Create a Virtual Environment

Open your terminal or command prompt in the project folder and run:

```bash
python -m venv venv
```

Then activate it:

**On Windows:**
```bash
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
source venv/bin/activate
```

### Step 3: Install Required Packages

```bash
pip install -r requirements.txt
```

This will install all the necessary libraries:
- pandas
- numpy
- faiss-cpu
- sentence-transformers
- rank-bm25
- scikit-learn
- groq
- python-dotenv

### Step 4: Get Your Groq API Key

1. Go to [https://console.groq.com/](https://console.groq.com/)
2. Sign up for a free account (or log in if you already have one)
3. Once logged in, go to the "API Keys" section
4. Click "Create API Key"
5. Copy the API key (it will look something like: `gsk_xxxxxxxxxxxxx`)

### Step 5: Create Environment File

Create a file named `.env` in the root folder of the project (same level as `README.md`).

Add your Groq API key to this file:

```
GROQ_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with the actual API key you copied from Groq.

**Important:** The `.env` file is already in `.gitignore`, so your API key won't be uploaded to GitHub.

### Step 6: Download the Dataset

1. Go to the Kaggle dataset page: [AI Research Papers Dataset](https://www.kaggle.com/datasets/yasirabdaali/arxivorg-ai-research-papers-dataset)
2. Download the dataset
3. Extract the CSV file
4. Place it in the `data/` folder and name it `arxiv_ai.csv`

## How to Run

1. Make sure your virtual environment is activated
2. Navigate to the `src/` folder:
   ```bash
   cd src
   ```
3. Run the main script:
   ```bash
   python main.py
   ```
4. When prompted, enter your question about AI research papers
5. Wait for the system to process and generate an answer

## Project Structure

```
RAG_AI_Research_Papers/
├── data/
│   └── arxiv_ai.csv          
├── src/
│   ├── main.py               
│   ├── preprocess.py         
│   ├── embed.py              
│   ├── retrieve.py           
│   └── generate.py           
├── logs/
│   └── app.log               
├── venv/                     
├── .env                      
├── requirements.txt          
├── .gitignore               
└── README.md                
```

## How It Works?

1. **Preprocessing**: The system loads the CSV file, cleans the data, and splits papers into smaller chunks (300 words each) for better search accuracy.

2. **Embedding**: Each chunk is converted into a vector (embedding) using Sentence Transformers. This allows the system to understand the meaning of text, not just keywords.

3. **Indexing**: 
   - **FAISS Index**: Stores all embeddings for fast vector similarity search
   - **BM25 Index**: Creates a keyword-based search index

4. **Hybrid Search**: When you ask a question:
   - The system searches using both vector similarity (semantic) and BM25 (keyword)
   - Both results are combined with a weighting factor (60% vector, 40% keyword)
   - Top 5 most relevant chunks are retrieved

5. **Generation**: The retrieved chunks are sent to Groq's LLaMA 3.1 model along with your question to generate a clear, accurate answer.

## Key Features

- ✅ Hybrid search combining semantic and keyword matching
- ✅ Fast vector search using FAISS
- ✅ Comprehensive logging for debugging
- ✅ Error handling for robust operation
- ✅ Clean, modular code structure

## Logging

All important events and errors are logged to `logs/app.log`. This helps you track what the system is doing and debug any issues.

## Troubleshooting

**Problem: "GROQ_API_KEY not found"**
- Make sure you created the `.env` file in the root folder
- Check that the API key is written correctly (no extra spaces)

**Problem: "Failed to load dataset"**
- Make sure `arxiv_ai.csv` is in the `data/` folder
- Check that the file name is exactly `arxiv_ai.csv`

**Problem: Import errors**
- Make sure your virtual environment is activated
- Run `pip install -r requirements.txt` again

## About Sentence Transformers

We use the `all-MiniLM-L6-v2` model from Sentence Transformers. This is a lightweight but powerful model that converts text into 384-dimensional vectors. It's perfect for semantic search tasks.

**Learn more:** [Sentence Transformers Documentation](https://www.sbert.net/)

## License

This project is created for educational purpose.

## Author Notes

This RAG system demonstrates:
- Understanding of retrieval-augmented generation concepts
- Implementation of hybrid search strategies
- Clean code organization and best practices
- Proper error handling and logging
- Integration with modern LLM APIs

