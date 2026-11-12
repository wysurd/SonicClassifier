from pathlib import Path

import numpy as np
import torch

from .audio import AudioConfig, GENRES, load_audio, mel_spectrogram, segments
from .model import GenreCNN


def load_checkpoint(path, device):
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    if checkpoint.get('genres') != list(GENRES) or checkpoint.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint or genre ordering')
    config = AudioConfig(**checkpoint['audio_config'])
    model = GenreCNN().to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, config, checkpoint


def predict(path, model, config, device):
    y = load_audio(path, config)
    features = [mel_spectrogram(clip, config) for clip in segments(y, config)]
    with torch.inference_mode():
        # Bound memory for longer uploads.
        outputs = []
        for start in range(0, len(features), 32):
            batch = torch.from_numpy(np.stack(features[start:start + 32]))[:, None].to(device)
            outputs.append(model(batch).softmax(1).cpu())
    return torch.cat(outputs).mean(0).numpy(), features[0]
