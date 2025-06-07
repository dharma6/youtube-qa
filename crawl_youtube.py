import yt_dlp
import os
import webvtt
import json

OUTPUT_DIR = "captions"

def time_to_seconds(t):
    parts = t.split(':')
    parts = [float(p.replace(',', '.')) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + int(s)

def get_video_urls(playlist_url):
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'dump_single_json': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        return [entry['url'] for entry in info['entries']]

def download_caption(video_url):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'skip_download': True,
        'subtitlesformat': 'vtt',
        'outtmpl': f'{OUTPUT_DIR}/%(id)s.%(ext)s',
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        video_id = info['id']

        ydl.download([video_url])

        # Check if caption file exists (either manual or auto-generated English subtitles)
        manual_path = f"{OUTPUT_DIR}/{video_id}.en.vtt"
        auto_path = f"{OUTPUT_DIR}/{video_id}.en.vtt"  # same name, downloaded whichever available

        if os.path.exists(manual_path):
            return True
        else:
            # No English subtitles file found
            return False

def parse_vtt(video_id):
    vtt_path = f"{OUTPUT_DIR}/{video_id}.en.vtt"
    data = []

    try:
        for caption in webvtt.read(vtt_path):
            start = time_to_seconds(caption.start)
            end = time_to_seconds(caption.end)
            text = caption.text.strip()

            if text:
                data.append({
                    "video_id": video_id,
                    "start": start,
                    "end": end,
                    "text": text,
                    "url": f"https://www.youtube.com/watch?v={video_id}&t={start}s"
                })
        return data
    except FileNotFoundError:
        print(f"⚠️ No captions found for {video_id}")
        return []

def process_playlist(playlist_url):
    print("🔍 Fetching video list...")
    video_urls = get_video_urls(playlist_url)
    all_data = []

    for video_url in video_urls:
        try:
          video_id = video_url.split("v=")[-1].split("&")[0]
          print(f"🎬 Processing {video_url}")
          found_captions = download_caption(video_url)
          if not found_captions:
              print(f"⚠️ No captions found for {video_id} (manual or auto-generated)")
              continue
          chunks = parse_vtt(video_id)
          all_data.extend(chunks)
        except Exception as e:
            print(f"❌ Error processing {video_url}: {e}")

    with open("captions_output.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    print(f"✅ Done! Total chunks extracted: {len(all_data)}")

def process_playlist_and_embed(playlist_url):
    process_playlist(playlist_url)
