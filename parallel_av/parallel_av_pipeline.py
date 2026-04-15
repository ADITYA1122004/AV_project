"""
parallel_av_pipeline.py
========================
Standalone Parallel Audio-Visual Capture Prototype
----------------------------------------------------
PURPOSE  : Demonstration of simultaneous mic + webcam capture using Python threading.
USAGE    : python parallel_av_pipeline.py
EXIT     : Press 'q' in the video window, or Ctrl+C in the terminal.

RULES    : This module is completely ISOLATED.
           - Does NOT import app.py
           - Does NOT use Streamlit
           - Does NOT use subprocess
           - Releases mic + camera on exit
"""

import threading
import time
import sys

import cv2
import numpy as np
import pyaudio

# ──────────────────────────────────────────────────────────────────────────────
# SHARED STATE
# ──────────────────────────────────────────────────────────────────────────────
audio_buffer: list = []
video_buffer: list = []
running: bool = True          # Global flag — threads watch this to exit cleanly

MAX_AUDIO: int = 50           # Max raw PCM chunks kept in memory
MAX_VIDEO: int = 20           # Max frames kept in memory

# 4.2 — Trigger cooldown: prevents log spam; fires at most once every 5 s
last_trigger_time: float = 0.0

# ──────────────────────────────────────────────────────────────────────────────
# AUDIO CAPTURE THREAD
# ──────────────────────────────────────────────────────────────────────────────

def audio_capture() -> None:
    """
    Reads 16-kHz mono PCM audio from the default microphone in a tight loop.
    Appends each CHUNK to the shared audio_buffer (bounded by MAX_AUDIO).
    Exits gracefully when the global `running` flag is cleared.
    """
    global running

    CHUNK   = 1024
    FORMAT  = pyaudio.paInt16
    CHANNELS = 1
    RATE    = 16_000

    p      = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print("[Audio] Microphone stream opened.")

    while running:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_buffer.append(data)

            # Keep buffer bounded — drop oldest chunk
            if len(audio_buffer) > MAX_AUDIO:
                audio_buffer.pop(0)

        except OSError:
            # Device temporarily unavailable — skip and retry
            continue
        except Exception:
            # Fail-safe: never crash the thread
            continue

    # ── Clean shutdown ──
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("[Audio] Microphone released.")


# ──────────────────────────────────────────────────────────────────────────────
# VIDEO CAPTURE THREAD
# ──────────────────────────────────────────────────────────────────────────────

def video_capture() -> None:
    """
    Reads frames from webcam (index 0) using DirectShow backend on Windows.
    Displays a live preview window; press 'q' to quit.
    Appends each frame to the shared video_buffer (bounded by MAX_VIDEO).
    Exits gracefully when the global `running` flag is cleared or 'q' is pressed.
    """
    global running

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[Video] ERROR: Cannot open webcam. Exiting video thread.")
        running = False
        return

    # Target resolution — 640×480 for low CPU load
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Target ~15 FPS to reduce CPU pressure
    cap.set(cv2.CAP_PROP_FPS, 15)

    print("[Video] Webcam stream opened at 640×480.")

    frame_skip_counter: int = 0
    SKIP_EVERY: int = 2       # Process every 2nd frame → ~7–8 effective FPS stored

    # 4.10 — Lightweight FPS monitor
    start_time: float = time.time()

    while running:
        ret, frame = cap.read()

        if not ret:
            # Frame grab failed — give the device a moment and retry
            time.sleep(0.05)
            continue

        # 4.1 — CPU throttle: prevent hot-spin on fast machines
        time.sleep(0.01)

        # ── Frame-skip logic (CPU saver) ──
        frame_skip_counter += 1
        if frame_skip_counter % SKIP_EVERY != 0:
            # Still show the raw frame for a smooth preview
            cv2.imshow("Parallel AV Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break
            continue

        # ── Store to buffer ──
        video_buffer.append(frame)
        if len(video_buffer) > MAX_VIDEO:
            video_buffer.pop(0)

        # ── Overlay HUD on the display copy ──
        display = frame.copy()
        a_len = len(audio_buffer)
        v_len = len(video_buffer)

        # 4.5 — Buffer counts
        cv2.putText(
            display,
            f"A:{a_len} V:{v_len}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # 4.6 — Running status indicator
        cv2.putText(
            display,
            "RUNNING",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # 4.7 — Buffer-ready indicator
        if v_len >= 15:
            cv2.putText(
                display,
                "READY",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Quit hint at bottom
        cv2.putText(
            display,
            "Press 'q' to quit",
            (10, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("Parallel AV Feed", display)

        # 4.10 — FPS monitor (printed every 10 stored frames)
        if frame_skip_counter % (SKIP_EVERY * 10) == 0:
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = (frame_skip_counter / SKIP_EVERY) / elapsed
                print(f"[Video] FPS: {fps:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    # ── Clean shutdown ──
    cap.release()
    cv2.destroyAllWindows()
    print("[Video] Webcam released.")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CONTROLLER
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Spawns audio + video threads, then enters a periodic status-print loop.
    Supports clean exit via:
      - 'q' key in the video window
      - Ctrl+C in the terminal
    """
    global running

    print("=" * 55)
    print("  Parallel AV Capture Prototype  (Demo Module)")
    print("  Press 'q' in the video window or Ctrl+C to exit.")
    print("=" * 55)

    t_audio = threading.Thread(target=audio_capture, name="AudioThread", daemon=True)
    t_video = threading.Thread(target=video_capture, name="VideoThread", daemon=True)

    t_audio.start()
    t_video.start()

    try:
        while running:
            a_len = len(audio_buffer)
            v_len = len(video_buffer)

            # 4.8 — Structured log output
            status = "ready" if v_len >= 15 else "collecting"
            print({
                "audio_chunks": a_len,
                "video_frames": v_len,
                "status": status,
            })

            # 4.2 — Trigger cooldown: fire at most once every 5 seconds
            global last_trigger_time
            if v_len >= 15 and time.time() - last_trigger_time > 5:
                print(">>> Ready for inference trigger")
                last_trigger_time = time.time()

            time.sleep(2)

    except KeyboardInterrupt:
        # 4.3 — Safe shutdown on Ctrl+C
        print("\nStopping system...")
        running = False

    # 4.3 — Join threads with timeout so we never hang indefinitely
    t_audio.join(timeout=4)
    t_video.join(timeout=4)

    # 4.9 — Clean exit message
    print("System shutdown complete.")
    sys.exit(0)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
