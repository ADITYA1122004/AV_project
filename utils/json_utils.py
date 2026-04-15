import json
import sys

def output_json(modality, raw_text, confidence):
    output = {
        "modality": modality,
        "raw_text": raw_text,
        "confidence": float(confidence)
    }
    print(json.dumps(output), file=sys.stdout)
    sys.stdout.flush()

def output_json_fusion(modality, fused_text):
    output = {
        "modality": modality,
        "fused_text": fused_text
    }
    print(json.dumps(output), file=sys.stdout)
    sys.stdout.flush()

def output_json_nlp(refined_text):
    output = {
        "refined_text": refined_text
    }
    print(json.dumps(output), file=sys.stdout)
    sys.stderr.flush()

def log(msg):
    # Log to stderr to avoid polluting JSON output
    print(msg, file=sys.stderr)
    sys.stderr.flush()
