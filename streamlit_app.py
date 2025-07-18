import streamlit as st
from youtube_qa import query_youtube_qa
from crawl_youtube import process_playlist_and_embed
from emned_captions import embed_captions
st.set_page_config(
    page_title="YouTube Q&A",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === Custom Styles ===
st.markdown("""
<style>
.stApp {
    background-color: #f9f9f9;
    color: #222;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.title {
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.answer-box {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgb(0 0 0 / 0.1);
    margin-bottom: 2rem;
    font-size: 1.1rem;
    line-height: 1.6;
}
.source-card {
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgb(0 0 0 / 0.07);
    padding: 1rem;
    margin-bottom: 1rem;
    display: flex;
    gap: 1rem;
    align-items: center;
    transition: transform 0.15s ease-in-out;
}
.source-card:hover {
    transform: scale(1.02);
    box-shadow: 0 6px 20px rgb(0 0 0 / 0.15);
}
.thumbnail {
    border-radius: 10px;
    flex-shrink: 0;
}
.source-text {
    flex-grow: 1;
    font-size: 0.95rem;
    color: #333;
}
.source-link {
    font-weight: 600;
    color: #0078d4;
    text-decoration: none;
    margin-bottom: 0.25rem;
    display: inline-block;
}
.source-link:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# === UI Header ===
st.markdown('<h1 class="title">🎥 Ask Questions Based on a YouTube Playlist</h1>', unsafe_allow_html=True)
st.markdown("This app lets you extract captions from a playlist and ask questions about the content.")

# === Playlist Processing Section ===
with st.sidebar.expander("📂 Upload and Embed a New Playlist"):
    playlist_url = st.text_input("Paste a YouTube Playlist URL", placeholder="https://www.youtube.com/playlist?list=...")
    if st.button("📥 Process Playlist"):
        if playlist_url:
            with st.spinner("⏳ Processing playlist and embedding into ChromaDB..."):
                try:
                    process_playlist_and_embed(playlist_url)
                    st.success("✅ Playlist crawled successfully")
                    embed_captions()


                except Exception as e:
                    st.error(f"❌ Failed to process playlist: {e}")
        else:
            st.warning("⚠️ Please enter a valid playlist URL.")

# === Q&A Section ===
st.markdown("---")
question = st.text_input("🔎 Ask a question from the embedded playlists", placeholder="E.g., What did he say about entrepreneurship?")
if question:
    with st.spinner("Thinking... 💭"):
        answer, sources = query_youtube_qa(question)

    st.markdown('<div class="answer-box"><strong>🧠 Answer:</strong><br>' + answer.replace('\n', '<br>') + '</div>', unsafe_allow_html=True)

    st.markdown("### 📌 Sources")
    for source in sources:
        thumbnail_url = f"https://img.youtube.com/vi/{source['video_id']}/hqdefault.jpg"
        link_url = source['url']
        text = source['text']

        st.markdown(f"""
        <div class="source-card">
            <img src="{thumbnail_url}" width="120" class="thumbnail" alt="Thumbnail" />
            <div class="source-text">
                <a href="{link_url}" target="_blank" class="source-link">▶ Watch segment</a>
                <p>{text}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with ❤️ by Dharma Bandaru using Streamlit + OpenAI + ChromaDB")
