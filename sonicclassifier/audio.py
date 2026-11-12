"""Identical preprocessing for training and inference."""
from dataclasses import asdict, dataclass

import librosa
import numpy as np

GENRES = ('blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock')


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    seconds: float = 3.0
    n_mels: int = 64
    n_fft: int = 2048
    hop_length: int = 512

    def __post_init__(self):
        if self.sample_rate <= 0 or self.seconds <= 0 or self.n_mels < 8 or self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError('Invalid audio configuration')

    def to_dict(self):
        return asdict(self)


def load_audio(path, config):
    y, _ = librosa.load(path, sr=config.sample_rate, mono=True)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError(f'Empty or non-finite audio: {path}')
    return y


def segments(y, config):
    """Non-overlapping segments; pad the last segment, including short clips."""
    size = int(config.sample_rate * config.seconds)
    if size < 1 or not len(y):
        raise ValueError('Audio must contain samples')
    return [librosa.util.fix_length(y[i:i + size], size=size) for i in range(0, len(y), size)]


def mel_spectrogram(y, config):
    size = int(config.sample_rate * config.seconds)
    y = librosa.util.fix_length(np.asarray(y, dtype=np.float32), size=size)
    power = librosa.feature.melspectrogram(y=y, sr=config.sample_rate, n_fft=config.n_fft,
                                         hop_length=config.hop_length, n_mels=config.n_mels)
    db = librosa.power_to_db(power, ref=max(float(power.max()), 1e-10), top_db=80)
    # Fixed physical dB scale avoids fitting normalization on held-out data.
    return np.clip((db + 80) / 80, 0, 1).astype(np.float32)
