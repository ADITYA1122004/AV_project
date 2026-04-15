import sys
import json
import math
import os
import warnings

# Suppress HuggingFace symlink warning on Windows & generic noise
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

try:
    from faster_whisper import WhisperModel
except ImportError:
    sys.exit(1)

# Optimization: Using 'small' model with int8 quantization for high-speed transcription
def run_asr(audio_path):
    try:
        # Note: In a subprocess architecture, this loads every time.
        # For "No model reload", consider caching at the orchestrator level.
        model = WhisperModel("small", device="cpu", compute_type="int8")
        
        segments, info = model.transcribe(
            audio_path,
            beam_size=1
        )
        
        seg_list = list(segments)
        text = " ".join([seg.text for seg in seg_list]).strip()
        
        if seg_list:
            avg_logprob = sum([s.avg_logprob for s in seg_list]) / len(seg_list)
            confidence = math.exp(avg_logprob)
        else:
            confidence = 0.0
            
        output = {
            "modality": "audio",
            "raw_text": text,
            "confidence": round(float(confidence), 4)
        }
        
        print(json.dumps(output))
        
    except Exception as e:
        output = {"modality": "audio", "raw_text": "", "confidence": 0.0, "error": str(e)}
        print(json.dumps(output))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
        
    audio_path = sys.argv[1]
    run_asr(audio_path)
