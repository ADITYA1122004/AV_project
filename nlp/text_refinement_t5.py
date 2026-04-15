import sys
import os
import torch
import warnings
import json

import logging

# Suppress HuggingFace symlink warning on Windows & generic noise
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.json_utils import output_json_nlp, log

def refine_text(fused_text):
    if not fused_text or fused_text.strip() == "":
        output_json_nlp(fused_text)
        return

    log("Loading T5 grammar correction model...")
    try:
        # Specialized grammar correction model
        model_name = "AventIQ-AI/T5-small-grammar-correction"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Apply dynamic quantization to int8 for CPU optimization
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        model.eval()
        
        # Input formulation
        # Note: Depending on the specific T5 tuning, it might not need "gec:" prefix,
        # but T5-small-grammar-correction often expects the plain text or specific prefix.
        input_ids = tokenizer(fused_text, return_tensors="pt")
        
        # Generator with safety guards
        with torch.no_grad():
            outputs = model.generate(
                **input_ids, 
                max_length=64, 
                num_beams=1, 
                early_stopping=True
            )
            refined_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        # Fallback if result is empty
        if not refined_text.strip():
            refined_text = fused_text
            
        output_json_nlp(refined_text)
        
    except Exception as e:
        log(f"Error in NLP refinement: {e}")
        output_json_nlp(fused_text) # fallback to original text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        log("Usage: python text_refinement_t5.py '<text_to_refine>'")
        sys.exit(1)
        
    fused_text = sys.argv[1]
    refine_text(fused_text)
