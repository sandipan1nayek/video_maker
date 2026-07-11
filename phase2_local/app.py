import streamlit as st
import json
import os
import shutil
import subprocess
import asyncio
import edge_tts
from mutagen.mp3 import MP3
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import google.generativeai as genai
from dotenv import load_dotenv

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Video Assembly Engine - Phase 2",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Configuration Paths & Directories
DB_PATH = "master_database.json"
RAW_ASSETS_DIR = "raw_assets"
TEMP_BUILD_DIR = "temp_build"
OUTPUT_FILE = "final_render.mp4"

# Ensure essential directories exist
os.makedirs(RAW_ASSETS_DIR, exist_ok=True)
os.makedirs(TEMP_BUILD_DIR, exist_ok=True)

# ---------------------------------------------------------
# Helper Functions & Resource Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading semantic embedding model...")
def get_embedding_model():
    """Load lightweight local SentenceTransformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def load_database():
    """Load the master database containing video clip metadata."""
    if not os.path.exists(DB_PATH):
        return []
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return []

def save_database(data):
    """Save the updated metadata array to master_database.json."""
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def time_to_seconds(time_str):
    """Convert HH:MM:SS format into float seconds."""
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = map(float, parts)
        return h * 3600 + m * 60 + s
    return 0.0

def get_ffmpeg_exe():
    """Return 'ffmpeg' if in system PATH, otherwise fallback to bundled imageio_ffmpeg binary."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


async def generate_speech(text, output_path, voice="en-US-ChristopherNeural"):
    """Generate voiceover audio asynchronously via edge-tts."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def run_async(coro):
    """Safely run async coroutine inside Streamlit without event loop clashes."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

def clean_temp_build():
    """Clean all intermediate files from the temp_build directory."""
    if os.path.exists(TEMP_BUILD_DIR):
        for item in os.listdir(TEMP_BUILD_DIR):
            file_path = os.path.join(TEMP_BUILD_DIR, item)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception:
                pass

# ---------------------------------------------------------
# Sidebar - Environment & Status
# ---------------------------------------------------------
st.sidebar.title("🛠️ Phase 2 Settings")
api_key = os.environ.get("GEMINI_API_KEY", "").strip()

if api_key:
    st.sidebar.success("✅ Gemini API Key Detected")
else:
    st.sidebar.error("❌ GEMINI_API_KEY missing in .env")

voice_choice = st.sidebar.selectbox(
    "Voiceover Narrator",
    options=[
        "en-US-ChristopherNeural",
        "en-US-GuyNeural",
        "en-US-AriaNeural",
        "en-GB-RyanNeural"
    ],
    index=0
)

if st.sidebar.button("🧹 Clean Temporary Build Files"):
    clean_temp_build()
    st.sidebar.success("Temporary build folder cleaned!")

st.sidebar.divider()
st.sidebar.markdown(
    "**Local Assets Directory:**\n"
    "`raw_assets/` (Place `.mp4` source clips here)\n\n"
    "**Master Database:**\n"
    "`master_database.json`"
)

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.title("🎬 AI Video Assembly Engine")
st.caption("Phase 2 Local Assembly Pipeline — Voiceover Synthesis, Semantic B-Roll Matching & FFmpeg Stitching")

tab1, tab2 = st.tabs(["⚡ Video Assembly Pipeline", "📂 Database Management"])

# =========================================================
# TAB 1: VIDEO ASSEMBLY PIPELINE
# =========================================================
with tab1:
    script_input = st.text_area(
        "Enter Voiceover Script",
        placeholder="Type or paste your narrative script here...",
        height=180
    )

    assemble_btn = st.button("🚀 Generate Final Video", type="primary", use_container_width=True)

    if assemble_btn:
        if not api_key:
            st.error("Please add your GEMINI_API_KEY to the `.env` file before generating videos.")
            st.stop()

        if not script_input.strip():
            st.warning("Please provide a voiceover script.")
            st.stop()

        db = load_database()
        if not db:
            st.error("Your master database is empty! Please go to the 'Database Management' tab and upload your Phase 1 batch results.")
            st.stop()

        # Ensure temp build directory is fresh
        clean_temp_build()

        st.subheader("Progress Monitor")
        status_box = st.empty()
        progress_bar = st.progress(0)

        # -----------------------------------------------------
        # Step 1: Script Splitting via Gemini 2.5 Flash
        # -----------------------------------------------------
        status_box.info("🧠 Step 1/4: Analyzing and splitting script using Gemini ('gemini-flash-latest')...")
        genai.configure(api_key=api_key)
        model_gemini = genai.GenerativeModel('gemini-flash-latest')
        prompt = (
            "Deconstruct this voiceover script into distinct, sequential phrases or individual sentences "
            "where the visual context changes. If a single long sentence implies two distinct scenarios "
            "or visual transitions, split that sentence into two separate phases. "
            "Return exclusively a JSON array of strings."
        )

        try:
            response = model_gemini.generate_content(
                f"{prompt}\n\nScript:\n{script_input}",
                generation_config={"response_mime_type": "application/json"}
            )
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            phrases = json.loads(raw_text.strip())
        except Exception as e:
            st.error(f"Gemini Script Splitting Failed: {e}")
            st.stop()

        st.write("📋 **Deconstructed Narrative Phases:**")
        st.json(phrases, expanded=False)

        # -----------------------------------------------------
        # Step 2 & 3: Audio Synthesis, Semantic Matching & Slicing
        # -----------------------------------------------------
        status_box.info("🔊 Step 2/4: Synthesizing neural voiceover & performing semantic vector search...")
        model = get_embedding_model()

        # Pre-compute database embeddings
        db_descriptions = [item["clip_description"] for item in db]
        db_embeddings = model.encode(db_descriptions, convert_to_tensor=True)

        used_clips = []
        video_segments = []
        audio_segments = []

        total_phrases = len(phrases)

        for i, phrase in enumerate(phrases):
            progress_bar.progress((i) / total_phrases)
            status_box.info(f"🎞️ Processing Phase {i+1}/{total_phrases}: \"{phrase[:50]}...\"")

            # A. Audio Synthesis
            audio_path = os.path.join(TEMP_BUILD_DIR, f"phrase_{i}.mp3")
            run_async(generate_speech(phrase, audio_path, voice=voice_choice))
            audio_segments.append(audio_path)

            # B. Audio Duration
            audio_info = MP3(audio_path)
            target_duration = audio_info.info.length

            # C. Semantic Vector Search
            phrase_embedding = model.encode(phrase, convert_to_tensor=True)
            similarities = cos_sim(phrase_embedding, db_embeddings)[0]
            sorted_indices = similarities.argsort(descending=True).tolist()

            current_video_dur = 0.0
            clips_for_phrase = []

            for idx in sorted_indices:
                clip = db[idx]
                clip_id = f"{clip['video_source']}_{clip['start_time']}"

                # Anti-repetition guard
                if clip_id in used_clips:
                    continue

                input_video_path = os.path.join(RAW_ASSETS_DIR, clip["video_source"])
                if not os.path.exists(input_video_path):
                    continue  # Skip if source file is not in raw_assets

                used_clips.append(clip_id)
                clip_dur = float(clip["duration_seconds"])
                needed_dur = target_duration - current_video_dur
                cut_dur = min(clip_dur, needed_dur)
                cut_start = time_to_seconds(clip["start_time"])

                output_cut = os.path.join(TEMP_BUILD_DIR, f"p{i}_clip{len(clips_for_phrase)}.mp4")

                # Non-destructive fast stream copy slice
                cmd_cut = [
                    get_ffmpeg_exe(), "-y",
                    "-ss", str(cut_start),
                    "-t", str(cut_dur),
                    "-i", input_video_path,
                    "-an",          # Mute original clip audio
                    "-c:v", "copy", # Fast stream copy
                    output_cut
                ]
                res = subprocess.run(cmd_cut, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if res.returncode == 0 and os.path.exists(output_cut):
                    clips_for_phrase.append(output_cut)
                    current_video_dur += cut_dur

                if current_video_dur >= target_duration - 0.05:
                    break

            # Handle chaining if multiple clips were needed to cover duration
            if len(clips_for_phrase) > 1:
                list_file = os.path.join(TEMP_BUILD_DIR, f"concat_p{i}.txt")
                with open(list_file, "w", encoding="utf-8") as f:
                    for c in clips_for_phrase:
                        # Use forward slashes inside ffmpeg list file
                        f.write(f"file '{os.path.abspath(c).replace(os.sep, '/')}'\n")

                merged_vid = os.path.join(TEMP_BUILD_DIR, f"merged_p{i}.mp4")
                cmd_merge = [
                    get_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0",
                    "-i", list_file, "-c", "copy", merged_vid
                ]
                subprocess.run(cmd_merge, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                video_segments.append(merged_vid)
            elif len(clips_for_phrase) == 1:
                video_segments.append(clips_for_phrase[0])
            else:
                st.error(f"❌ Could not find matching local video files in `{RAW_ASSETS_DIR}/` for Phase {i+1}. Ensure raw `.mp4` files are present.")
                st.stop()

        # -----------------------------------------------------
        # Step 4: Final Assembly & Muxing
        # -----------------------------------------------------
        status_box.info("🎞️ Step 4/4: Stitching final video and audio tracks...")
        progress_bar.progress(0.9)

        # Concatenate all phrase videos
        final_video_only = os.path.join(TEMP_BUILD_DIR, "all_video.mp4")
        vid_list_file = os.path.join(TEMP_BUILD_DIR, "final_vid_list.txt")
        with open(vid_list_file, "w", encoding="utf-8") as f:
            for v in video_segments:
                f.write(f"file '{os.path.abspath(v).replace(os.sep, '/')}'\n")

        cmd_concat_vid = [
            get_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0",
            "-i", vid_list_file, "-c", "copy", final_video_only
        ]
        subprocess.run(cmd_concat_vid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Concatenate all phrase audio clips
        final_audio_only = os.path.join(TEMP_BUILD_DIR, "all_audio.mp3")
        aud_list_file = os.path.join(TEMP_BUILD_DIR, "final_aud_list.txt")
        with open(aud_list_file, "w", encoding="utf-8") as f:
            for a in audio_segments:
                f.write(f"file '{os.path.abspath(a).replace(os.sep, '/')}'\n")

        cmd_concat_aud = [
            get_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0",
            "-i", aud_list_file, "-c", "copy", final_audio_only
        ]
        subprocess.run(cmd_concat_aud, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Mux Video + Audio into final_render.mp4
        cmd_mux = [
            get_ffmpeg_exe(), "-y",
            "-i", final_video_only,
            "-i", final_audio_only,
            "-c:v", "copy",
            "-c:a", "aac",
            OUTPUT_FILE
        ]
        mux_res = subprocess.run(cmd_mux, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        progress_bar.progress(1.0)

        if mux_res.returncode == 0 and os.path.exists(OUTPUT_FILE):
            status_box.success("🎉 Video Assembly Successfully Completed!")
            st.video(OUTPUT_FILE)
            st.download_button(
                label="⬇️ Download Final Render",
                data=open(OUTPUT_FILE, "rb").read(),
                file_name="final_render.mp4",
                mime="video/mp4"
            )
        else:
            st.error("❌ FFmpeg Muxing failed. Check logs.")

# =========================================================
# TAB 2: DATABASE MANAGEMENT
# =========================================================
with tab2:
    st.subheader("📂 Master Database Status")
    db_data = load_database()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Indexed Clips", len(db_data))
    with col2:
        unique_vids = len(set(item.get("video_source", "") for item in db_data))
        st.metric("Source Videos", unique_vids)
    with col3:
        total_dur = sum(float(item.get("duration_seconds", 0)) for item in db_data)
        st.metric("Total Footage", f"{total_dur / 60:.1f} mins")

    st.divider()

    st.subheader("➕ Append Phase 1 Batch Results")

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    if "last_append_msg" in st.session_state:
        msg_type, msg_text = st.session_state.last_append_msg
        if msg_type == "success":
            st.success(msg_text)
        elif msg_type == "info":
            st.info(msg_text)
        elif msg_type == "error":
            st.error(msg_text)
        del st.session_state.last_append_msg

    uploaded_file = st.file_uploader(
        "Upload `batch_results.json` from Phase 1",
        type=["json"],
        key=f"json_uploader_{st.session_state.uploader_key}"
    )
    if uploaded_file is not None:
        if st.button("➕ Confirm & Append to Database", type="primary"):
            try:
                new_data = json.load(uploaded_file)
                if isinstance(new_data, list):
                    existing_keys = {
                        (item.get("video_source"), item.get("start_time"), item.get("end_time"))
                        for item in db_data
                    }
                    added_count = 0
                    for clip in new_data:
                        key = (clip.get("video_source"), clip.get("start_time"), clip.get("end_time"))
                        if key not in existing_keys:
                            db_data.append(clip)
                            existing_keys.add(key)
                            added_count += 1
                    save_database(db_data)
                    if added_count > 0:
                        st.session_state.last_append_msg = ("success", f"✅ Successfully added {added_count} new clips! ({len(new_data) - added_count} duplicates skipped)")
                    else:
                        st.session_state.last_append_msg = ("info", "ℹ️ All clips in this file are already in the database. 0 duplicates added.")
                    st.session_state.uploader_key += 1
                    st.rerun()
                else:
                    st.session_state.last_append_msg = ("error", "Uploaded JSON must be a list of clip objects.")
                    st.session_state.uploader_key += 1
                    st.rerun()
            except Exception as e:
                st.session_state.last_append_msg = ("error", f"Error importing JSON: {e}")
                st.session_state.uploader_key += 1
                st.rerun()

    with st.expander("🔍 Preview First 10 Database Entries"):
        st.json(db_data[:10] if db_data else [])
