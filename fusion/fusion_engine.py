import sys
import json


def fuse_outputs(audio, visual):
    audio_text = audio.get("raw_text", "")
    audio_conf = float(audio.get("confidence", 0))

    visual_text = visual.get("raw_text", "")
    visual_conf = float(visual.get("confidence", 0))

    # Rule 1: Strong audio confidence → prefer audio
    if audio_conf >= 0.75:
        return {
            "modality": "fusion",
            "final_text": audio_text,
            "confidence": audio_conf,
            "source": "audio"
        }

    # Rule 2: Acceptable visual confidence → use visual
    elif visual_conf >= 0.60:
        return {
            "modality": "fusion",
            "final_text": visual_text,
            "confidence": visual_conf,
            "source": "visual"
        }

    # Rule 3: Fallback → pick whichever has higher confidence
    else:
        if audio_conf >= visual_conf:
            return {
                "modality": "fusion",
                "final_text": audio_text,
                "confidence": audio_conf,
                "source": "audio_fallback"
            }
        else:
            return {
                "modality": "fusion",
                "final_text": visual_text,
                "confidence": visual_conf,
                "source": "visual_fallback"
            }


if __name__ == "__main__":
    try:
        # Read input JSON from stdin
        input_data = json.load(sys.stdin)

        audio = input_data.get("audio", {})
        visual = input_data.get("visual", {})

        result = fuse_outputs(audio, visual)

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))

