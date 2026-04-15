import streamlit as st
import subprocess
import sys
python_exe = sys.executable
import json
import os
import time
import math
import warnings
import numpy as np
import pyaudio
import threading
import logging
from collections import deque
from streamlit.runtime.scriptrunner import add_script_run_ctx

# --- ISSUE 2: TRANSFORMERS LOG SPAM & WARNINGS ---
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

# --- Icon Definitions ---
MIC_ICON = """
<svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
<rect x="9" y="2" width="6" height="12" rx="3"/>
<path d="M5 10v2a7 7 0 0 0 14 0v-2"/>
<line x1="12" y1="19" x2="12" y2="22"/>
</svg>
"""

CAM_ICON = """
<svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
<path d="M23 7l-7 5 7 5V7z"/>
<rect x="1" y="5" width="15" height="14" rx="2"/>
</svg>
"""

FUSION_ICON = """
<svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
<polyline points="16 3 21 3 21 8"/>
<line x1="4" y1="20" x2="21" y2="3"/>
<polyline points="21 16 21 21 16 21"/>
<line x1="15" y1="15" x2="21" y2="21"/>
</svg>
"""

TEXT_ICON = """
<svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12"/>
<line x1="14" y1="2" x2="14" y2="8"/>
<line x1="8" y1="13" x2="16" y2="13"/>
</svg>
"""

SETTINGS_ICON = """
<svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
<circle cx="12" cy="12" r="3"/>
<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1 1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
</svg>
"""

st.set_page_config(
    page_title="AV Speech System",
    layout="wide"
)

st.markdown(f"""
<div style="display:flex; align-items:center; gap:12px;">
{MIC_ICON}
<h1>Audio-Visual Speech Understanding System</h1>
</div>
""", unsafe_allow_html=True)

# --- High-Speed Constants ---
MAX_WORDS = 120
# Streaming configuration is now handled within the loop

# --- Settings Sidebar ---
with st.sidebar:
    st.header("Settings")
    enable_refinement = st.checkbox("Enable T5 NLP Refinement", value=True)

# --- Optimized Model Loading (Locked to 'small' for stability) ---
@st.cache_resource
def get_whisper_model():
    from faster_whisper import WhisperModel
    print("🔥 Whisper model loaded: SMALL")
    return WhisperModel("small", device="cpu", compute_type="int8")

    return WhisperModel("small", device="cpu", compute_type="int8")

def run_script(command, timeout=120):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        lines = result.stdout.strip().split("\n")
        return json.loads(lines[-1])
    except Exception as e:
        return {"error": str(e)}

# --- Text Utilities ---
def merge_text(old, new):
    if not old: return new.strip()
    if new.lower() in old.lower():
        return old
    return old + " " + new

def is_complete_phrase(text):
    """Relaxed completion filter to allow short valid words like 'I', 'am'."""
    words = text.split()
    if len(words) == 0:
        return False
    return True

def remove_phrase_duplicates(text):
    words = text.split()
    result = []
    for i in range(len(words)):
        if i >= 2 and words[i] == words[i-2] and words[i-1] == words[i-3]: continue
        result.append(words[i])
    return " ".join(result)

def refine_text(text):
    words = text.split()
    if not words: return text
    replacements = {"i": "I", "im": "I'm", "dont": "don't", "cant": "can't"}
    words = [replacements.get(w.lower(), w) for w in words]
    words[0] = words[0].capitalize()
    sentence = " ".join(words)
    if not sentence.endswith((".", "?", "!")): sentence += "."
    return sentence

def light_stabilize(text):
    """Level 1: Lightning fast stabilization for live streaming."""
    if not text: return ""
    tokens = text.split()
    cleaned = []
    
    # 1. Remove duplicate consecutive words
    for w in tokens:
        if not cleaned or cleaned[-1].lower() != w.lower():
            cleaned.append(w)
            
    # 2. Trim repeated short phrase pairs (O(n))
    res = []
    i = 0
    while i < len(cleaned):
        if i + 3 < len(cleaned) and cleaned[i:i+2] == cleaned[i+2:i+4]:
            res.extend(cleaned[i:i+2])
            i += 4
        else:
            res.append(cleaned[i])
            i += 1
            
    return " ".join(res)

def format_final_text(text):
    """Level 2: Professional formatting for final output."""
    if not text: return ""
    
    # 1. Normalize spacing
    text = " ".join(text.split())
    
    # 2. Fix common ASR word cases
    replacements = {
        " i ": " I ",
        " im ": " I'm ",
        " dont ": " don't ",
        " cant ": " can't "
    }
    # Special handle for 'i' at the start
    if text.lower().startswith("i "):
        text = "I " + text[2:]
        
    for k, v in replacements.items():
        text = text.replace(k, v)
        
    # 3. Capitalize sentences
    sentences = text.split(".")
    processed_sentences = []
    for s in sentences:
        s = s.strip()
        if s:
            # Only uppercase first letter, leave rest as is to preserve ASR proper nouns
            if len(s) > 1:
                processed_sentences.append(s[0].upper() + s[1:])
            else:
                processed_sentences.append(s.upper())
    text = ". ".join(processed_sentences)
    
    # 4. Final punctuation
    if text and not text.endswith((".", "?", "!")):
        text += "."
        
    return text

# --- Shared Decoupled Buffer ---
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = deque(maxlen=int(16000 * 3.5)) # 3.5 seconds

def audio_capture_worker(stream, CHUNK):
    """THREAD A: High-priority continuous mic capture."""
    while st.session_state.recording_active:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_np = np.frombuffer(data, dtype=np.int16)
            st.session_state.audio_buffer.extend(audio_np)
            st.session_state.mic_level = np.abs(audio_np).mean()
        except:
            break

def audio_inference_worker(model):
    """THREAD B: Decoupled Whisper inference on buffer snapshots."""
    last_inference_time = time.time()
    silence_counter = 0
    
    while st.session_state.recording_active:
        # Check every 1.5s
        if (time.time() - last_inference_time) >= 1.0:
            # 1. Validation Gate
            current_level = st.session_state.get("mic_level", 0)
            if current_level < 20:
                silence_counter += 1
                if silence_counter >= 4:
                    st.session_state.audio_buffer.clear()
                last_inference_time = time.time()
                continue
            
            silence_counter = 0
            
            # 2. Stability Delay (Micro-tuned for speed)
            time.sleep(0.05)
            
            # 3. Snapshot, Normalize & Boost
            snapshot = (np.array(list(st.session_state.audio_buffer), dtype=np.float32) / 32768.0) * 1.2
            
            # 3. Deterministic Decoding (Beam Size 2 for Accuracy Boost)
            segments, _ = model.transcribe(
                snapshot,
                language="en",
                task="transcribe",
                beam_size=2,
                temperature=0.0,
                initial_prompt="simple english conversation",
                condition_on_previous_text=False
            )
            
            result = list(segments)
            if result:
                # 4. Extract Newest Segment Only (Prevents Repetition)
                new_text = result[-1].text.strip()
                
                # 5. Global Duplicate Protection
                if not new_text or new_text.lower() in st.session_state.full_transcript.lower():
                    last_inference_time = time.time()
                    continue
                
                # 6. Phrase Completion Filter (Prevents "i am go" fragments)
                if not is_complete_phrase(new_text):
                    last_inference_time = time.time()
                    continue

                # 7. Smart Merge & Stabilization
                st.session_state.full_transcript = merge_text(st.session_state.full_transcript, new_text)
                st.session_state.full_transcript = remove_phrase_duplicates(st.session_state.full_transcript)
                
                # Word limit management
                words = st.session_state.full_transcript.split()
                if len(words) > 120:
                    st.session_state.full_transcript = " ".join(words[-120:])
            
            last_inference_time = time.time()
        
        time.sleep(0.1)

def audio_stream_worker():
    """Master Orchestrator: Manages capture/inference thread lifecycle."""
    model = get_whisper_model()
    p = pyaudio.PyAudio()
    CHUNK = 2048
    
    # Device Logging
    print("\n--- Audio Devices ---")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Index {i}: {info['name']}")
    
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                        input=True, frames_per_buffer=CHUNK)
        
        # Launch Decoupled Threads
        t_capture = threading.Thread(target=audio_capture_worker, args=(stream, CHUNK), daemon=True)
        t_inference = threading.Thread(target=audio_inference_worker, args=(model,), daemon=True)
        
        add_script_run_ctx(t_capture)
        add_script_run_ctx(t_inference)
        
        t_capture.start()
        t_inference.start()
        
        # Keep internal worker alive
        while st.session_state.recording_active:
            time.sleep(0.5)
            
    finally:
        # Resource cleanup
        if 'stream' in locals():
            try:
                stream.stop_stream()
                stream.close()
            except: pass
        p.terminate()
        print("--- Continuous Audio Pipeline Closed ---")

# --- Session State Initialization ---
if "recording_active" not in st.session_state:
    st.session_state.recording_active = False
if "visual_active" not in st.session_state:
    st.session_state.visual_active = False
if "start_multimodal" not in st.session_state:
    st.session_state.start_multimodal = False
if "full_transcript" not in st.session_state:
    st.session_state.full_transcript = ""
if "visual_output" not in st.session_state:
    st.session_state.visual_output = None
if "mic_level" not in st.session_state:
    st.session_state.mic_level = 0

def start_audio_stream():
    st.session_state.recording_active = True
    st.session_state.full_transcript = ""
    audio_thread = threading.Thread(target=audio_stream_worker, daemon=True)
    add_script_run_ctx(audio_thread)
    audio_thread.start()

def start_visual_capture():
    # Only run once if not already active to prevent loop
    if not st.session_state.visual_active:
        st.session_state.visual_active = True
        output = run_script([python_exe, "visual/visual_inference.py"])
        st.session_state["visual_output"] = output
        st.session_state.visual_active = False
        return output
    return None

# --- Unified High-Speed Audio Transcription ---
st.markdown(f"""
<div style="display:flex; align-items:center; gap:10px;">
{MIC_ICON}
<h2>Continuous Transcription</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 2])
if col1.button("Start Recording", use_container_width=True):
    start_audio_stream()

if col2.button("Stop Recording", use_container_width=True):
    st.session_state.recording_active = False

if col3.button("Clear Transcript"):
    st.session_state.full_transcript = ""

if st.session_state.recording_active:
    st.info("Streaming Mode Active (5s window, 1.5s interval)... Speak now.")
    
    # UI must remain outside loop (ISSUE 1)
    status_placeholder = st.empty()
    transcript_placeholder = st.empty()
    
    # UI Status Feedback Logic
    mic_l = st.session_state.get("mic_level", 0)
    if mic_l < 10:
        status_msg = "No mic input detected"
        s_color = "#ff4b4b" # Red
    elif mic_l < 40:
        status_msg = "Low input / Quiet"
        s_color = "#ffa500" # Orange
    else:
        status_msg = "Listening..."
        s_color = "#28a745" # Green

    with status_placeholder.container():
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; color:{s_color}; font-weight:bold;">
        <span>🎤 Status: {status_msg}</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(int(mic_l), 100) / 100.0)

    transcript_placeholder.success(f"**Live Transcript:** {st.session_state.full_transcript}")
    
    # Trigger short rerun to update UI with background thread data
    time.sleep(0.8)
    st.rerun()

# Final Polish (Two-Level Stabilization Logic)
if st.session_state.full_transcript:
    if not st.session_state.recording_active:
        # Full professional formatting after stop
        final_output = format_final_text(st.session_state.full_transcript)
        st.divider()
        st.markdown(f"""
<div style="display:flex; align-items:center; gap:10px;">
{TEXT_ICON}
<span><strong>Final Transcript (Stabilized)</strong></span>
</div>
""", unsafe_allow_html=True)
        st.markdown(f"#### {final_output}")
    else:
        # Light stabilization during live recording
        live_output = light_stabilize(st.session_state.full_transcript)
        st.divider()
        st.markdown(f"""
<div style="display:flex; align-items:center; gap:10px;">
{TEXT_ICON}
<span><strong>Live Transcript (Optimized)</strong></span>
</div>
""", unsafe_allow_html=True)
        st.markdown(f"#### {live_output}")

st.divider()

# --- Visual Subsystem ---
st.markdown(f"""
<div style="display:flex; align-items:center; gap:10px;">
{CAM_ICON}
<h2>Visual Lip Reading</h2>
</div>
""", unsafe_allow_html=True)
if st.button("Start Visual Capture"):
    with st.spinner("Processing video..."):
        output = run_script([python_exe, "visual/visual_inference.py"])
    if "error" not in output:
        st.session_state["visual_output"] = output
        st.success(f"Predicted: {output.get('raw_text', '')}")
    else:
        st.error(f"Error: {output['error']}")

st.divider()

# --- Fusion Subsystem ---
st.markdown(f"""
<div style="display:flex; align-items:center; gap:10px;">
{FUSION_ICON}
<h2>Multi-Modal Fusion</h2>
</div>
""", unsafe_allow_html=True)

if st.button("Run Fusion & Refinement"):
    st.session_state.start_multimodal = True

if st.session_state.get("start_multimodal"):
    # 1. Automatic Capture Triggers
    if not st.session_state.recording_active:
        start_audio_stream()
        st.success("🎤 Audio Active")

    if not st.session_state.visual_output and not st.session_state.visual_active:
        st.success("📷 Camera Active")
        with st.spinner("Capturing visual data..."):
            start_visual_capture()

    st.info("Fusion Mode: Audio Priority")

    # 2. Data Wait Gate
    audio_text = st.session_state.full_transcript.strip()
    if not audio_text:
        st.info("Listening for audio...")
        st.stop()

    # 3. Decision Logic
    visual_output = st.session_state.get("visual_output")
    visual_text = visual_output.get("raw_text") if visual_output else None

    if audio_text and visual_text:
        # Full multi-modal fusion path
        with st.spinner("Fusing modalities..."):
            audio_bundle = {
                "modality": "audio",
                "raw_text": refine_text(audio_text),
                "confidence": 0.9
            }
            fusion_input = {"audio": audio_bundle, "visual": visual_output}

            try:
                result = subprocess.run(
                    [python_exe, "fusion/fusion_engine.py"],
                    input=json.dumps(fusion_input),
                    text=True, capture_output=True, timeout=10
                )
                fusion_output = json.loads(result.stdout.strip())
                final_text = fusion_output.get("final_text", audio_text)
            except Exception as e:
                st.error(f"Fusion script error: {e}")
                final_text = audio_text
    elif audio_text:
        final_text = refine_text(audio_text)
    else:
        final_text = visual_text

    if final_text:
        st.success(f"Output Generated: {final_text}")
        if enable_refinement:
            with st.spinner("Refining..."):
                refine_script_output = run_script([python_exe, "nlp/text_refinement_t5.py", final_text], timeout=20)
                if "refined_text" in refine_script_output:
                    st.write("**Refined Text (T5):**", refine_script_output["refined_text"])
    
    # End multimodal cycle if output achieved
    # st.session_state.start_multimodal = False
