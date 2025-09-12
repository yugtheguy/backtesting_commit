# summarizer/services.py

import requests
import concurrent.futures
from django.conf import settings
from supadata import Supadata, SupadataError

# --- Main Service Function ---

def generate_summary_from_url(url):
    """
    Orchestrates the entire summarization process.
    Takes a YouTube URL, gets the transcript, and summarizes it.
    """
    try:
        video_id = url.split("v=")[1].split("&")[0]
    except IndexError:
        return {"error": "Invalid YouTube URL format."}

    transcript_text = _get_transcript(video_id)
    
    if transcript_text:
        return _summarize_text(transcript_text)
    else:
        return {"error": "Could not retrieve transcript."}

# --- Helper Functions ---

def _get_transcript(video_id):
    """Fetches transcript using the Supadata library."""
    try:
        supadata = Supadata(api_key=settings.SUPADATA_API_KEY)
        response = supadata.youtube.transcript(video_id=video_id, text=True)
        return response.content
    except SupadataError as e:
        print(f"Supadata API Error: {e}")
        return None

def _summarize_text(text):
    """Summarizes text in parallel using Hugging Face API."""
    api_url = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6" # Using the faster model
    headers = {"Authorization": f"Bearer {settings.HF_API_TOKEN}"}

    if len(text.split()) < 40:
        return {"summary": "Transcript is too short to be summarized."}

    max_chunk_length = 3500
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]

    def summarize_chunk(chunk):
        payload = {"inputs": chunk}
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()[0]['summary_text']
        except requests.RequestException:
            return ""

    with concurrent.futures.ThreadPoolExecutor() as executor:
        chunk_summaries = list(executor.map(summarize_chunk, chunks))

    final_summary = " ".join(filter(None, chunk_summaries))
    return {"summary": final_summary} if final_summary else {"error": "Failed to generate summary."}