import whisper
import numpy as np
import pyaudio
import torch
import time
import asyncio

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper model and move to GPU if available
model = whisper.load_model("medium", device=device)

# Threshold for detecting silence (this may need some tweaking based on your environment)
SILENCE_THRESHOLD = 0.01
SILENCE_CHUNKS = 18  # Number of consecutive chunks that are considered silence
MAX_BUFFER_DURATION = 15  # Max seconds to buffer before forcing transcription
MIN_NON_SILENT_DURATION = 0.3  # Minimum amount of speech duration (in seconds) before transcribing

def is_silence(audio_chunk, threshold=SILENCE_THRESHOLD):
    """Determine if the audio chunk is silence based on the average amplitude."""
    audio_np = np.frombuffer(audio_chunk, np.int16).astype(np.float32) / 32768.0
    return np.abs(audio_np).mean() < threshold

def enough_speech(buffer):
    """Check if the buffer contains enough non-silent audio to be transcribed."""
    non_silent_chunks = sum([not is_silence(chunk) for chunk in buffer])
    # Calculate the proportion of non-silent chunks
    non_silent_duration = non_silent_chunks / len(buffer)  # Proportion of non-silent chunks
    total_duration = len(buffer) * 1024 / 16000  # Approximate total duration in seconds
    return non_silent_duration > 0.1 and total_duration >= MIN_NON_SILENT_DURATION

async def record_audio_async(chunk_size=1024, sample_rate=16000):
    """Asynchronously capture audio chunks using PyAudio."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=chunk_size)
    print("Recording...")

    try:
        while True:
            data = stream.read(chunk_size)
            yield data
            await asyncio.sleep(0)  # Yield control to the event loop
    except KeyboardInterrupt:
        print("Stopping...")
        stream.stop_stream()
        stream.close()
        p.terminate()

async def process_buffer(buffer):
    """Process and transcribe the buffer if it contains enough speech."""
    if buffer and enough_speech(buffer):
        print(f"Transcribing... buffered {len(buffer)} chunks.")
        # Convert the buffered audio to a NumPy array
        audio_np = np.frombuffer(b"".join(buffer), np.int16).astype(np.float32) / 32768.0

        # Transcribe the accumulated audio using Whisper, specifying Traditional Chinese ('zh-tw')
        result = model.transcribe(audio_np, language='zh')

        # Print the transcription in real time
        print(result['text'])
    else:
        print("Discarded silent buffer.")

async def transcribe_audio_async():
    """Asynchronously capture and transcribe audio with a buffer for speech detection."""
    buffer = []
    silence_counter = 0
    start_time = time.time()  # Track how long we've been buffering

    async for audio_chunk in record_audio_async():
        # Append the chunk to the buffer
        buffer.append(audio_chunk)

        if is_silence(audio_chunk):
            silence_counter += 1
        else:
            silence_counter = 0

        # If we detect a long enough silence or if buffer time exceeds MAX_BUFFER_DURATION, process the buffered audio
        buffer_duration = time.time() - start_time
        if silence_counter > SILENCE_CHUNKS or buffer_duration > MAX_BUFFER_DURATION:
            # Process the buffer asynchronously
            await process_buffer(buffer)
            
            # Clear the buffer after transcription or discard
            buffer = []
            start_time = time.time()  # Reset the buffer timer

            # Reset the silence counter
            silence_counter = 0

        await asyncio.sleep(0)  # Yield control to the event loop

async def main():
    """Main function to run the asynchronous transcription."""
    await transcribe_audio_async()

if __name__ == "__main__":
    asyncio.run(main())