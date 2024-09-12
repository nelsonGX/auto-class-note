import asyncio
import os
import logging
import struct
from typing import List, Optional
import numpy as np
import whisper
import torch
from bitstring import BitStream
from pyrtmp import StreamClosedException
from pyrtmp.session_manager import SessionManager
from pyrtmp.rtmp import SimpleRTMPController, RTMPProtocol, SimpleRTMPServer
import audioop

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
# warn if running on CPU
if device == "cpu":
    logger.warning("GPU not available, running on CPU.")

# Load Whisper model and move to GPU if available
model = whisper.load_model("medium", device=device)

# Threshold for detecting silence (this may need some tweaking based on your environment)
SILENCE_THRESHOLD = 0.005
SILENCE_CHUNKS = 50  # Number of consecutive chunks that are considered silence
MAX_BUFFER_DURATION = 30  # Max seconds to buffer before forcing transcription
MIN_NON_SILENT_DURATION = 1.0  # Minimum amount of speech duration (in seconds) before transcribing

class AACParser:
    def __init__(self):
        self.buffer = b""

    def add_data(self, data: bytes):
        self.buffer += data

    def parse(self) -> Optional[bytes]:
        if len(self.buffer) < 2:
            return None

        while len(self.buffer) >= 2:
            packet_type = self.buffer[0]
            if packet_type == 0:  # AAC sequence header
                if len(self.buffer) < 4:
                    return None
                header_length = struct.unpack(">H", self.buffer[2:4])[0] + 4
                if len(self.buffer) >= header_length:
                    logger.info("Received AAC sequence header")
                    self.buffer = self.buffer[header_length:]
                else:
                    return None
            elif packet_type == 1:  # AAC raw
                frame_length = struct.unpack(">H", self.buffer[1:3])[0] + 3
                if len(self.buffer) >= frame_length:
                    frame = self.buffer[3:frame_length]
                    self.buffer = self.buffer[frame_length:]
                    return frame
                else:
                    return None
            else:
                logger.warning(f"Unknown AAC packet type: {packet_type}")
                self.buffer = self.buffer[1:]

        return None

class RawAudio:
    def __init__(self, stream: BitStream) -> None:
        self.format = stream.read("uint:4")
        self.sampling = stream.read("uint:2")
        self.size = stream.read("uint:1")
        self.channel = stream.read("uint:1")
        self.bytes = stream.read("bytes")

        # logger.info(f"Audio format: {self.format}, Sampling: {self.sampling}, Size: {self.size}, Channel: {self.channel}")

        self.pcm = self.decode_audio()

    def decode_audio(self) -> np.ndarray:
        if self.format == 10:  # AAC
            return self.decode_aac()
        else:
            logger.warning(f"Unsupported audio format: {self.format}. Returning empty PCM.")
            return np.array([], dtype=np.float32)

    def decode_aac(self) -> np.ndarray:
        try:
            # Convert to mono if stereo
            if self.channel == 1:  # Stereo
                pcm = audioop.tomono(self.bytes, 2, 1, 1)
            else:
                pcm = self.bytes
            
            # Resample to 16kHz if necessary
            if self.sampling != 1:  # Not 44.1kHz
                pcm = audioop.ratecv(pcm, 2, 1, 44100, 16000, None)[0]
            
            return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            logger.error(f"Error decoding AAC frame: {e}")
            return np.array([], dtype=np.float32)

def is_silence(audio_chunk: np.ndarray, threshold=SILENCE_THRESHOLD) -> bool:
    """Determine if the audio chunk is silence based on the average amplitude."""
    return np.abs(audio_chunk).mean() < threshold

def enough_speech(buffer: List[np.ndarray]) -> bool:
    """Check if the buffer contains enough non-silent audio to be transcribed."""
    if not buffer:
        return False
    non_silent_chunks = sum([not is_silence(chunk) for chunk in buffer])
    non_silent_duration = non_silent_chunks / len(buffer)  # Proportion of non-silent chunks
    total_duration = sum(len(chunk) for chunk in buffer) / 16000  # Assuming 16kHz sample rate
    return non_silent_duration > 0.1 and total_duration >= MIN_NON_SILENT_DURATION

async def process_buffer(buffer: List[np.ndarray]):
    """Process and transcribe the buffer if it contains enough speech."""
    if buffer and enough_speech(buffer):
        logger.info(f"Transcribing... buffered {len(buffer)} chunks.")
        # Concatenate all numpy arrays in the buffer
        audio_np = np.concatenate(buffer)

        # Log some debug information
        logger.debug(f"Audio shape: {audio_np.shape}, dtype: {audio_np.dtype}")
        logger.debug(f"Audio min: {audio_np.min()}, max: {audio_np.max()}, mean: {audio_np.mean()}")

        # Transcribe the accumulated audio using Whisper, specifying Traditional Chinese ('zh-tw')
        result = model.transcribe(audio_np, language='zh')

        # Print the transcription in real time
        logger.info(f"Transcription: {result['text']}")
    else:
        logger.info("Discarded buffer due to insufficient speech.")

class RTMP2STTController(SimpleRTMPController):
    def __init__(self):
        self.buffer: List[np.ndarray] = []
        self.silence_counter = 0
        self.last_process_time = asyncio.get_event_loop().time()
        self.aac_parser = AACParser()
        super().__init__()

    async def on_audio_message(self, session, message) -> None:
        try:
            self.aac_parser.add_data(message.payload)
            while True:
                aac_frame = self.aac_parser.parse()
                if aac_frame is None:
                    break
                
                raw_audio = RawAudio(BitStream(aac_frame))
                if raw_audio.format == 10:  # AAC
                    pcm_data = raw_audio.pcm
                    if len(pcm_data) > 0:
                        self.buffer.append(pcm_data)
                        
                        if is_silence(pcm_data):
                            self.silence_counter += 1
                        else:
                            self.silence_counter = 0

                        current_time = asyncio.get_event_loop().time()
                        buffer_duration = current_time - self.last_process_time

                        if self.silence_counter > SILENCE_CHUNKS or buffer_duration > MAX_BUFFER_DURATION:
                            await process_buffer(self.buffer)
                            self.buffer = []
                            self.silence_counter = 0
                            self.last_process_time = current_time

                        # logger.debug(f"Processed {len(pcm_data)} samples of PCM data")
                    else:
                        logger.info("Skipped empty PCM data")
                else:
                    logger.warning(f"Unsupported audio format: {raw_audio.format}")
        except Exception as e:
            logger.error(f"Error processing audio message: {e}", exc_info=True)
        await super().on_audio_message(session, message)

class SimpleServer(SimpleRTMPServer):
    async def create(self, host: str, port: int):
        loop = asyncio.get_event_loop()
        self.server = await loop.create_server(
            lambda: RTMPProtocol(controller=RTMP2STTController()),
            host=host,
            port=port,
        )

async def main():
    server = SimpleServer()
    await server.create(host='0.0.0.0', port=1935)
    await server.start()
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())