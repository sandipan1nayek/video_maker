import streamlit as st
import json
import os
import subprocess
import asyncio
import edge_tts
from mutagen.mp3 import MP3
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration Constants
DB_PATH = "master_database.json"
RAW_ASSETS_DIR = "raw_assets"
OUTPUT_FILE = "final_render.mp4"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize directories
if not os.path.exists(RAW_ASSETS_DIR):
    os.makedirs(RAW_ASSETS_DIR)

# Cache the SentenceTransformer model to prevent reloading
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Helper Functions
def time_to_seconds(time_str):
    """Converts HH:MM:SS format to float seconds."""
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s

def load_db():
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_db(data):
    with open(DB_PATH, 'w') as f:
        json.dump(data, f, indent=4)

async def generate_audio(text, output_file):
    """Generates audio using edge-tts asynchronously."""
    communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
    await communicate.save(output_file)

# UI Layout
st.title("🎬 AI Video Assembly Engine (Phase 2)")

st.sidebar.info(
    "Ensure `GEMINI_API_KEY` is set in your `.env` file.\n\n"
    "Place raw videos in the `raw_assets/` folder."
)

st.header("1. Database Append Automation")
uploaded_file = st.file_uploader("Upload batch_results.json from Phase 1", type=["json"])
if uploaded_file is not None:
    try:
        new_data = json.load(uploaded_file)
        db = load_db()
        db.extend(new_data)
        save_db(db)
        st.success(f"Successfully appended {len(new_data)} items to master_database.json!")
    except Exception as e:
        st.error(f"Error reading JSON file: {e}")

st.divider()

st.header("2. Video Assembly")
script = st.text_area("Enter Voiceover Script", height=150)

if st.button("Generate Final Video"):
    if not GEMINI_API_KEY:
        st.error("Missing GEMINI_API_KEY. Please create a `.env` file in the root directory.")
        st.stop()
        
    if not script.strip():
        st.error("Please enter a voiceover script.")
        st.stop()
        
    db = load_db()
    if not db:
        st.error("Master database is empty. Please upload a `batch_results.json` first.")
        st.stop()

    # Step 1: Contextual Script Splitting via Gemini 2.5 Flash
    with st.spinner("Splitting script using Gemini 2.5 Flash..."):
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = (
            "Deconstruct this voiceover script into distinct, sequential phrases or individual sentences "
            "where the visual context changes. If a single long sentence implies two distinct scenarios "
            "or visual transitions, split that sentence into two separate phases. "
            "Return exclusively a JSON array of strings."
        )
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt, script],
                config={'response_mime_type': 'application/json'}
            )
            phrases = json.loads(response.text)
        except Exception as e:
            st.error(f"Failed to process script with Gemini: {e}")
            st.stop()
            
    st.write(f"Found {len(phrases)} distinct phases:")
    st.json(phrases)

    # Step 2 & 3: Audio Synthesis, Vector Search, and Duration Matching
    used_clips = []
    video_segments = []
    audio_segments = []
    
    # Pre-compute embeddings for database entries
    db_texts = [item['clip_description'] for item in db]
    db_embeddings = model.encode(db_texts, convert_to_tensor=True)
    
    progress_bar = st.progress(0)
    
    for i, phrase in enumerate(phrases):
        st.write(f"**Phase {i+1}:** {phrase}")
        
        # Audio Synthesis
        audio_file = f"temp_audio_{i}.mp3"
        asyncio.run(generate_audio(phrase, audio_file))
        audio_segments.append(audio_file)
        
        # Timing Component
        audio_info = MP3(audio_file)
        target_duration = audio_info.info.length
        
        # Local Vector Semantic Search Engine
        phrase_embedding = model.encode(phrase, convert_to_tensor=True)
        similarities = cos_sim(phrase_embedding, db_embeddings)[0]
        sorted_indices = similarities.argsort(descending=True).tolist()
        
        current_video_duration = 0
        clips_for_phrase = []
        
        for idx in sorted_indices:
            clip = db[idx]
            # Create a unique identifier for the clip to avoid immediate repetition
            clip_id = f"{clip['video_source']}_{clip['start_time']}"
            
            if clip_id in used_clips:
                continue
                
            used_clips.append(clip_id)
            clip_dur = clip['duration_seconds']
            
            # Duration Matching Logic
            needed_dur = target_duration - current_video_duration
            cut_dur = min(clip_dur, needed_dur)
            cut_start = time_to_seconds(clip['start_time'])
            
            input_video = os.path.join(RAW_ASSETS_DIR, clip['video_source'])
            output_cut = f"temp_vid_p{i}_c{len(clips_for_phrase)}.mp4"
            
            if not os.path.exists(input_video):
                st.error(f"CRITICAL ERROR: Raw video {input_video} not found. FFmpeg will fail.")
                st.stop()
            
            # Non-Destructive FFmpeg Video Splitting
            cmd_cut = [
                "ffmpeg", "-y",
                "-ss", str(cut_start),
                "-t", str(cut_dur),
                "-i", input_video,
                "-an",          # Mute original audio
                "-c:v", "copy", # Fast stream copy
                output_cut
            ]
            subprocess.run(cmd_cut, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            clips_for_phrase.append(output_cut)
            current_video_duration += cut_dur
            
            # Allow a tiny float precision gap to exit the loop
            if current_video_duration >= target_duration - 0.05:
                break
                
        # Chain clips if we needed multiple for this single audio phase
        if len(clips_for_phrase) > 1:
            list_file = f"concat_list_p{i}.txt"
            with open(list_file, 'w') as f:
                for c in clips_for_phrase:
                    f.write(f"file '{c}'\n")
            
            merged_vid = f"merged_vid_p{i}.mp4"
            cmd_merge = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", merged_vid
            ]
            subprocess.run(cmd_merge, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            video_segments.append(merged_vid)
        elif len(clips_for_phrase) == 1:
            video_segments.append(clips_for_phrase[0])
        else:
            st.error(f"Could not find enough video clips to cover phase {i+1} duration.")
            st.stop()
            
        progress_bar.progress((i + 1) / len(phrases))

    # Step 4: Final Stitch Protocol
    with st.spinner("Assembling final video..."):
        # Concatenate all phrase video segments
        final_video_only = "temp_final_video.mp4"
        with open("final_vid_list.txt", "w") as f:
            for v in video_segments:
                f.write(f"file '{v}'\n")
                
        cmd_concat_vid = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "final_vid_list.txt", "-c", "copy", final_video_only]
        subprocess.run(cmd_concat_vid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Concatenate all phrase audio segments
        final_audio_only = "temp_final_audio.mp3"
        with open("final_aud_list.txt", "w") as f:
            for a in audio_segments:
                f.write(f"file '{a}'\n")
                
        cmd_concat_aud = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "final_aud_list.txt", "-c", "copy", final_audio_only]
        subprocess.run(cmd_concat_aud, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Mux final video and audio
        cmd_mux = [
            "ffmpeg", "-y",
            "-i", final_video_only,
            "-i", final_audio_only,
            "-c:v", "copy",
            "-c:a", "aac",
            OUTPUT_FILE
        ]
        subprocess.run(cmd_mux, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    st.success("✅ Assembly Complete!")
    st.video(OUTPUT_FILE)
    
    # Instructions for cleanup
    st.info("Temporary files (temp_*.mp4/mp3, *.txt) have been left on disk for debugging purposes. You may delete them manually.")
