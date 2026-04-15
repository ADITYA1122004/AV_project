import pyaudio
import wave
import os
import sys

OUTPUT_PATH = "outputs/temp_audio.wav"

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5  # default duration

# Allow duration to be passed as a command-line argument
if len(sys.argv) > 1:
    try:
        RECORD_SECONDS = int(sys.argv[1])
    except ValueError:
        pass

def record_audio(duration=RECORD_SECONDS):
    os.makedirs("outputs", exist_ok=True)

    p = pyaudio.PyAudio()

    print(f"Recording for {duration} seconds... Speak now")

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(OUTPUT_PATH, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Saved audio to {OUTPUT_PATH}")


if __name__ == "__main__":
    record_audio()