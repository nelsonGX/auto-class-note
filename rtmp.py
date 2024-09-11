import asyncio
import os
import logging
import struct
from typing import List, Optional

from bitstring import BitStream
from pyrtmp import StreamClosedException
from pyrtmp.flv import FLVFileWriter, FLVMediaType
from pyrtmp.session_manager import SessionManager
from pyrtmp.rtmp import SimpleRTMPController, RTMPProtocol, SimpleRTMPServer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class AACParser:
    def __init__(self, data: bytes):
        self.data = data
        self.bitstream = BitStream(data)

    def parse(self) -> Optional[bytes]:
        if len(self.data) < 2:
            return None

        packet_type = self.bitstream.read('uint:8')
        if packet_type == 0:  # AAC sequence header
            logger.info("Received AAC sequence header")
            return None
        elif packet_type == 1:  # AAC raw
            return self.data[1:]  # Skip the packet type byte
        else:
            logger.warning(f"Unknown AAC packet type: {packet_type}")
            return None

class RawAudio:
    def __init__(self, stream: BitStream) -> None:
        self.format = stream.read("uint:4")
        self.sampling = stream.read("uint:2")
        self.size = stream.read("uint:1")
        self.channel = stream.read("uint:1")
        self.bytes = stream.read("bytes")

        logger.info(f"Audio format: {self.format}, Sampling: {self.sampling}, Size: {self.size}, Channel: {self.channel}")

        self.pcm = self.decode_audio()

    def decode_audio(self) -> List[int]:
        if self.format == 10:  # AAC
            return self.decode_aac()
        else:
            logger.warning(f"Unsupported audio format: {self.format}. Returning empty PCM.")
            return []

    def decode_aac(self) -> List[int]:
        aac_parser = AACParser(self.bytes)
        aac_frame = aac_parser.parse()
        if aac_frame is None:
            return []
        
        # This is a very basic and imperfect conversion from AAC to PCM
        # It won't produce correct audio, but it will produce some data
        return [int.from_bytes(aac_frame[i:i+2], 'little', signed=True) 
                for i in range(0, len(aac_frame), 2)]

class WAVFileWriter:
    def __init__(self, output: str, sample_rate: int = 44100, bits_per_sample: int = 16, channels: int = 2):
        self.output = output
        self.sample_rate = sample_rate
        self.bits_per_sample = bits_per_sample
        self.channels = channels
        self.data_size = 0
        self.file = open(output, "wb")
        self._write_header()

    def _write_header(self):
        self.file.write(b'RIFF')
        self.file.write(struct.pack('<I', 36 + self.data_size))
        self.file.write(b'WAVE')
        self.file.write(b'fmt ')
        self.file.write(struct.pack('<I', 16))
        self.file.write(struct.pack('<H', 1))
        self.file.write(struct.pack('<H', self.channels))
        self.file.write(struct.pack('<I', self.sample_rate))
        byte_rate = self.sample_rate * self.channels * self.bits_per_sample // 8
        self.file.write(struct.pack('<I', byte_rate))
        block_align = self.channels * self.bits_per_sample // 8
        self.file.write(struct.pack('<H', block_align))
        self.file.write(struct.pack('<H', self.bits_per_sample))
        self.file.write(b'data')
        self.file.write(struct.pack('<I', self.data_size))

    def write(self, samples: List[int]):
        for sample in samples:
            self.file.write(struct.pack('<h', sample))
        self.data_size += len(samples) * self.bits_per_sample // 8
        
    def close(self):
        self._update_header()
        self.file.close()

    def _update_header(self):
        self.file.seek(4)
        self.file.write(struct.pack('<I', 36 + self.data_size))
        self.file.seek(40)
        self.file.write(struct.pack('<I', self.data_size))

class AACFileWriter:
    def __init__(self, output: str):
        self.output = output
        self.file = open(output, "wb")

    def write(self, aac_frame: bytes):
        self.file.write(aac_frame)

    def close(self):
        self.file.close()

class RTMP2WAVController(SimpleRTMPController):
    def __init__(self, output_directory: str, output_aac: bool = True):
        self.output_directory = output_directory
        self.output_aac = output_aac
        super().__init__()

    async def on_ns_publish(self, session, message) -> None:
        publishing_name = message.publishing_name
        wav_path = os.path.join(self.output_directory, f"{publishing_name}.wav")
        session.state = {
            'wav_writer': WAVFileWriter(output=wav_path),
            'aac_writer': AACFileWriter(os.path.join(self.output_directory, f"{publishing_name}.aac")) if self.output_aac else None
        }
        logger.info(f"Created WAV file: {wav_path}")
        if self.output_aac:
            logger.info(f"Created AAC file: {os.path.join(self.output_directory, f'{publishing_name}.aac')}")
        await super().on_ns_publish(session, message)

    async def on_audio_message(self, session, message) -> None:
        try:
            raw_audio = RawAudio(BitStream(message.payload))
            if raw_audio.format == 10:  # AAC
                aac_parser = AACParser(raw_audio.bytes)
                aac_frame = aac_parser.parse()
                if aac_frame is not None:
                    if session.state['aac_writer']:
                        session.state['aac_writer'].write(aac_frame)
                    session.state['wav_writer'].write(raw_audio.pcm)
                    logger.info(f"Wrote {len(aac_frame)} bytes to AAC file and {len(raw_audio.pcm)} samples to WAV file")
                else:
                    logger.info("Skipped AAC sequence header")
            else:
                session.state['wav_writer'].write(raw_audio.pcm)
                logger.info(f"Wrote {len(raw_audio.pcm)} audio samples to WAV file")
        except Exception as e:
            logger.error(f"Error processing audio message: {e}")
        await super().on_audio_message(session, message)

    async def on_stream_closed(self, session: SessionManager, exception: StreamClosedException) -> None:
        if hasattr(session, 'state'):
            if 'wav_writer' in session.state:
                session.state['wav_writer'].close()
                logger.info(f"Closed WAV file for stream: {session.stream_id}")
            if 'aac_writer' in session.state and session.state['aac_writer']:
                session.state['aac_writer'].close()
                logger.info(f"Closed AAC file for stream: {session.stream_id}")
        await super().on_stream_closed(session, exception)

class SimpleServer(SimpleRTMPServer):

    def __init__(self, output_directory: str):
        self.output_directory = output_directory
        super().__init__()

    async def create(self, host: str, port: int):
        loop = asyncio.get_event_loop()
        self.server = await loop.create_server(
            lambda: RTMPProtocol(controller=RTMP2WAVController(self.output_directory)),
            host=host,
            port=port,
        )

async def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server = SimpleServer(output_directory=current_dir)
    await server.create(host='0.0.0.0', port=1935)
    await server.start()
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())