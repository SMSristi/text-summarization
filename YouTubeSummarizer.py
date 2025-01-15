import sys
import re
import torch
import gradio as gr
from transformers import pipeline

def install_package(package_name):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import VideoUnavailable, TranscriptsDisabled
except ModuleNotFoundError:
    print("Required package 'youtube-transcript-api' not found. Installing...")
    install_package("youtube-transcript-api")
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import VideoUnavailable, TranscriptsDisabled

# Model path and pipeline setup
model_path = "../Models/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff"
text_summary = pipeline("summarization", model=model_path, torch_dtype=torch.bfloat16)

# Function to summarize text
def summary(input_text):
    output = text_summary(input_text)
    return output[0]['summary_text']

# Function to get YouTube transcript
def get_youtube_transcript(url):
    try:
        video_id_match = re.search(r"(?:v=|youtu.be/|embed/)([\w-]{11})", url)
        if not video_id_match:
            return "Invalid YouTube URL."

        video_id = video_id_match.group(1)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = "\n".join([entry['text'] for entry in transcript])
        return transcript_text

    except VideoUnavailable:
        return "The video is unavailable or does not exist."
    except TranscriptsDisabled:
        return "Transcripts are disabled for this video."
    except Exception as e:
        return f"An error occurred: {e}"

def chunk_text(text, chunk_size=800):
        """
        Splits the text into smaller chunks of a specified size.
        """
        words = text.split()
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

def youtube_summary(url):
        transcript = get_youtube_transcript(url)
        if "An error occurred" in transcript or "Invalid" in transcript:
            return transcript, ""

        # Split the transcript into manageable chunks
        chunks = chunk_text(transcript, chunk_size=800)

        # Summarize each chunk
        summaries = [summary(chunk) for chunk in chunks]

        # Optionally, summarize the combined summaries for a concise result
        combined_summary = " ".join(summaries)
        if len(combined_summary.split()) > 800:  # If combined summary is still long
            final_summary = summary(combined_summary)
        else:
            final_summary = combined_summary

        return transcript, final_summary
        #Gradio automatically maps the outputs based on the order v


# Custom CSS for the Gradio interface
css = """
    h1 {
        font-size: 2rem;
        color: #ffffff;
        margin-bottom: 20px;
        text-align: center;
    }
    .gradio-input textarea {
    height: 80px;
    border-radius: 8px;
    border: 1px solid #ccc;
    padding: 10px;
    font-size: 1rem;
    background-color: #f9f9f9;
    }

    gradio-output textarea {
        height: 250px;
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
        font-size: 1rem;
        background-color: #f9f9f9;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 14px;
        color: #6c757d;
    }
    .footer a {
        color: #007bff;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
"""

# Gradio interface with styling
interface = gr.Interface(
    fn=youtube_summary,
    inputs=gr.Textbox(label="Enter YouTube Video URL", placeholder="Paste your YouTube URL here...", lines=1),
    outputs=[
        gr.Textbox(label="Transcript", placeholder="The full transcript will appear here...", lines=10),
        gr.Textbox(label="Summary", placeholder="The summary will appear here...", lines=5),
    ],
    title="YouTube Script Summarizer",
    description="Provide a YouTube video URL to get its transcript and summary.",
    css=css  # Apply custom CSS
)

# Launch the Gradio app
interface.launch()

