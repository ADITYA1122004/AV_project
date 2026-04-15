import sys
import os
import torch
import json
import warnings

import logging

# Suppress HuggingFace symlink warning on Windows & generic noise
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. Clean Model Map for easy extensibility
MODEL_MAP = {
    "Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "Italian": "Helsinki-NLP/opus-mt-en-it"
}

# 2. Model Caching (Note: Effective if script stays resident or is imported)
MODEL_CACHE = {}

def log(msg):
    """Writes logs to stderr so as not to interfere with stdout JSON parsing."""
    print(f"[TRANSLATOR] {msg}", file=sys.stderr)
    sys.stderr.flush()

def get_model(model_name):
    """Loads and caches the model/tokenizer with CPU optimization."""
    if model_name not in MODEL_CACHE:
        log(f"Loading and quantizing model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Apply dynamic quantization for CPU speed
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        # Ensure eval mode to disable dropout/training overhead
        model.eval()

        MODEL_CACHE[model_name] = (tokenizer, model)

    return MODEL_CACHE[model_name]

def translate_text(text, language):
    """
    Translates input text from English to target language.
    Includes performance optimizations and safe fallbacks.
    """
    
    # 3. Skip Translation for English (Efficiency)
    if language == "English" or language == "en":
        print(json.dumps({
            "modality": "translation",
            "translated_text": text
        }))
        sys.exit(0)

    try:
        # Resolve model name from map (handles both names and codes)
        model_id = MODEL_MAP.get(language, f"Helsinki-NLP/opus-mt-en-{language}")
        
        tokenizer, model = get_model(model_id)
        
        # 4. Input Length Limit (Performance/Memory control)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64
        )
        
        # Generate translation
        with torch.no_grad():
            translated = model.generate(**inputs)
            translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]
        
        # Guaranteed output contract
        print(json.dumps({
            "modality": "translation",
            "translated_text": translated_text,
            "target_lang": language
        }))
        
    except Exception as e:
        # 5. Safe Fallback: Return original text and error info
        print(json.dumps({
            "modality": "translation",
            "translated_text": text,
            "error": str(e)
        }))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        # Fail gracefully
        print(json.dumps({"error": "Missing arguments"}))
        sys.exit(1)
        
    input_text = sys.argv[1]
    target_lang = sys.argv[2]
    
    translate_text(input_text, target_lang)
