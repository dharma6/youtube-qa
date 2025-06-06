import os
import uuid
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import chromadb
from chromadb import PersistentClient

# === CONFIGURATION ===
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "youtube_captions"
DEFAULT_TOP_K = 5
RETRIEVE_K = 15  # retrieve more for re-ranking
OPENAI_API_KEY = "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"


# === INIT ===
client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

app = FastAPI()


# === MODELS ===
class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = DEFAULT_TOP_K


class QAResponse(BaseModel):
    answer: str
    sources: List[dict]  # {text, url, start, end}


# === FUNCTIONS ===
def embed_text(text: str) -> List[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return response.data[0].embedding


def re_rank_chunks(question: str, chunks: List[str]) -> List[int]:
    # Ask GPT to rank most relevant segments
    formatted_chunks = "\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(chunks)])
    ranking_prompt = f"""
You are given a question and a list of caption segments.
Rank them in order of usefulness for answering the question.

Question: {question}

Segments:
{formatted_chunks}

Return the top 5 segment numbers as a comma-separated list (like: 3,5,1,2,4):
"""

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a ranking assistant that selects the most relevant caption segments to answer a question."},
            {"role": "user", "content": ranking_prompt}
        ],
        temperature=0
    )

    top_ids = response.choices[0].message.content.strip().split(",")
    return [int(i.strip()) - 1 for i in top_ids if i.strip().isdigit()]


@app.post("/ask", response_model=QAResponse)
async def ask_question(req: AskRequest):
    question = req.question
    top_k = req.top_k or DEFAULT_TOP_K
    embedding = embed_text(question)

    # Step 1: Retrieve more than needed
    results = collection.query(
        query_embeddings=[embedding],
        n_results=RETRIEVE_K,
        include=["documents", "metadatas"]
    )

    if not results["documents"] or not results["documents"][0]:
        return QAResponse(answer="❌ Sorry, no relevant content was found in the video captions.", sources=[])

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Step 2: Re-rank with GPT
    best_indices = re_rank_chunks(question, documents)
    best_indices = best_indices[:top_k]

    chunks = [documents[i] for i in best_indices]
    selected_meta = [metadatas[i] for i in best_indices]

    # Step 3: Format context
    context_parts = []
    sources = []
    for chunk, meta in zip(chunks, selected_meta):
        ts_link = f"https://www.youtube.com/watch?v={meta['video_id']}&t={int(meta['start'])}s"
        context_parts.append(f"{chunk}\n[Watch]({ts_link})")
        sources.append({
            "text": chunk,
            "url": ts_link,
            "start": meta["start"],
            "end": meta["end"]
        })

    context = "\n\n".join(context_parts)

    # Step 4: GPT Answer
    prompt = f"""
You are a helpful assistant. Use only the following YouTube captions to answer the question. Be factual. If there's not enough information, try to provide best possible answers based on the sources.

### Captions:
{context}

### Question:
{question}

### Answer:
"""

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a factual assistant answering questions based only on YouTube caption context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content.strip()
    return QAResponse(answer=answer, sources=sources)
