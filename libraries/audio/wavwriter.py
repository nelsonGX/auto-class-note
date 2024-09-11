import struct
from typing import List, BinaryIO

class WAVWriter:
    def __init__(self, sample_rate: int = 44100, bits_per_sample: int = 16, channels: int = 1):
        self.sample_rate = sample_rate
        self.bits_per_sample = bits_per_sample
        self.channels = channels
        self.data_size = 0

    def write_header(self, file: BinaryIO) -> None:
        file.write(b'RIFF')
        file.write(struct.pack('<I', 36 + self.data_size))  # File size (to be filled later)
        file.write(b'WAVE')
        file.write(b'fmt ')
        file.write(struct.pack('<I', 16))  # Subchunk1Size
        file.write(struct.pack('<H', 1))   # AudioFormat (PCM)
        file.write(struct.pack('<H', self.channels))
        file.write(struct.pack('<I', self.sample_rate))
        byte_rate = self.sample_rate * self.channels * self.bits_per_sample // 8
        file.write(struct.pack('<I', byte_rate))
        block_align = self.channels * self.bits_per_sample // 8
        file.write(struct.pack('<H', block_align))
        file.write(struct.pack('<H', self.bits_per_sample))
        file.write(b'data')
        file.write(struct.pack('<I', self.data_size))  # Data size (to be filled later)

    def write_samples(self, file: BinaryIO, samples: List[int]) -> None:
        for sample in samples:
            file.write(struct.pack('<h', sample))
        self.data_size += len(samples) * self.bits_per_sample // 8

    def update_header(self, file: BinaryIO) -> None:
        file.seek(4)
        file.write(struct.pack('<I', 36 + self.data_size))
        file.seek(40)
        file.write(struct.pack('<I', self.data_size))

class WAVFileWriter:
    def __init__(self, output: str, sample_rate: int = 44100, bits_per_sample: int = 16, channels: int = 1):
        self.output = output
        self.writer = WAVWriter(sample_rate, bits_per_sample, channels)
        self.file = open(output, "wb")
        self.writer.write_header(self.file)

    def write(self, samples: List[int]):
        self.writer.write_samples(self.file, samples)

    def close(self):
        self.writer.update_header(self.file)
        self.file.close()