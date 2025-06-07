from typing import List
from openai import OpenAI
from chromadb import PersistentClient
import os
import streamlit as st

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "youtube_captions"
TOP_K = 5

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def validate_sources(question: str, answer: str, sources: List[dict]) -> List[dict]:
  enriched_sources = []
  for source in sources:
      validation_prompt = f"""
        You are a helpful assistant. A user asked a question and was given an answer based on transcribed YouTube captions.

        Evaluate how relevant the following caption chunk is to answering the user's question and whether it supports the final answer.

        Question: {question}
        Answer: {answer}

        Caption chunk:
        \"\"\"
        {source['text']}
        \"\"\"

        Respond in the following JSON format:
        {{
          "relevance": "<High/Medium/Low>",
          "comment": "<Why it is or isn’t relevant>"
        }}"""
      response = client.chat.completions.create(
          model=GPT_MODEL,
          messages=[{"role": "user", "content": validation_prompt}],
          temperature=0.3
      )

      import json
      try:
          validation = json.loads(response.choices[0].message.content.strip())
      except Exception:
          validation = {"relevance": "Unknown", "comment": "Could not parse validation"}

      if validation["relevance"] in ["High"]:
          source["validation"] = validation
          enriched_sources.append(source)

  return enriched_sources

def embed_text(text: str) -> List[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return response.data[0].embedding

def query_youtube_qa(question: str):
    embedding = embed_text(question)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K,
        include=["documents", "metadatas"]
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []
    sources = []
    for chunk, meta in zip(chunks, metadatas):
        ts_link = f"https://www.youtube.com/watch?v={meta['video_id']}&t={int(meta['start'])}s"
        context_parts.append(f"{chunk}\n[Watch here]({ts_link})")
        sources.append({
            "text": chunk,
            "url": ts_link,
            "start": meta["start"],
            "end": meta["end"],
            "video_id": meta["video_id"]
        })

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful assistant. Answer the question based only on the following YouTube captions (transcribed from videos).
Do not make up facts. Add helpful details only from the context.

Context:
{context}

Question: {question}
Answer:
"""

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = response.choices[0].message.content.strip()
    # sources = validate_sources(question, answer, sources)
    return answer, sources
