import numpy as np

def is_silence(audio_chunk, threshold=500):
    """
    Check if the audio chunk is silent based on RMS energy.
    audio_chunk: numpy array of int16 samples.
    Returns: bool (True if silent)
    """
    if len(audio_chunk) == 0:
        return True
    
    # Calculate RMS energy
    rms = np.sqrt(np.mean(np.square(audio_chunk, dtype=np.float32)))
    return rms < threshold
