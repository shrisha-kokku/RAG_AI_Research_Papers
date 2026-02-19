import os
import logging
from groq import Groq
from dotenv import load_dotenv

#Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please set it in the .env file.")

#Initialize Groq client
client = Groq(api_key=api_key)


def generate_response(query, retrieved_docs):
    try:
        #Limit to top 3 chunks to avoid token overflow
        retrieved_docs = retrieved_docs[:3]

        context = "\n\n".join(retrieved_docs)

        prompt = f"""
You are an AI research assistant.

Answer the question using ONLY the provided research paper context.
If the answer is not in the context, say that it is not found.

Context:
{context}

Question:
{query}

Answer:
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful and precise research assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=400
        )

        answer = response.choices[0].message.content

        logging.info("Groq LLaMA 3.1 response generated successfully")

        return answer

    except Exception as e:
        logging.error(f"Groq LLM generation failed: {e}")
        raise
