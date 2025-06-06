import os
import json
import uuid
from tqdm import tqdm
from openai import OpenAI
import chromadb
from chromadb import PersistentClient
import tiktoken

# === CONFIG ===
DATA_FILE = "captions_output.json"
DB_DIR = "chroma_db"
COLLECTION_NAME = "youtube_captions"
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# === INIT ===
client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

tokenizer = tiktoken.get_encoding("cl100k_base")

# === FUNCTION TO GROUP CAPTIONS BY SEMANTIC BOUNDARY (token-aware) ===
def smart_group_captions(captions, max_tokens=180):
    grouped = []
    buffer = []
    current_tokens = 0

    def flush():
        if not buffer:
            return None
        combined_text = ' '.join([c['text'].strip() for c in buffer])
        return {
            "text": combined_text,
            "start": buffer[0]['start'],
            "end": buffer[-1]['end'],
            "video_id": buffer[0]['video_id'],
            "url": buffer[0]['url']
        }

    for c in captions:
        tokens = len(tokenizer.encode(c['text']))
        if current_tokens + tokens > max_tokens:
            group = flush()
            if group:
                grouped.append(group)
            buffer = [c]
            current_tokens = tokens
        else:
            buffer.append(c)
            current_tokens += tokens

    group = flush()
    if group:
        grouped.append(group)

    return grouped

# === LOAD & PROCESS CAPTIONS ===
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"📄 Loaded {len(data)} caption chunks")

# === SMART GROUPING ===
grouped_data = smart_group_captions(data)
print(f"📄 Grouped into {len(grouped_data)} semantically optimized chunks")

# === BATCHING & EMBEDDING ===
batch_size = 100
for i in tqdm(range(0, len(grouped_data), batch_size), desc="📦 Embedding & Storing"):
    batch = grouped_data[i:i + batch_size]
    texts = [entry['text'] for entry in batch]

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    embeddings = [e.embedding for e in response.data]

    metadatas = [
        {
            "video_id": entry["video_id"],
            "start": entry["start"],
            "end": entry["end"],
            "url": entry["url"]
        } for entry in batch
    ]

    ids = [str(uuid.uuid4()) for _ in batch]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

print("✅ Done embedding and storing to Chroma.")
