from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .audio import GENRES, load_audio, mel_spectrogram, segments


def discover(root):
    root = Path(root).resolve()
    records = []
    for label, genre in enumerate(GENRES):
        for path in sorted((root / genre).glob('*')):
            if path.suffix.lower() in {'.wav', '.au', '.flac', '.mp3', '.ogg'}:
                records.append({'path': str(path.relative_to(root)), 'label': label})
    if not records:
        raise ValueError('No audio found: expected ROOT/genre/song.wav (or .au/.flac/.mp3/.ogg)')
    return records


def split_songs(records, seed=42):
    # Split original files BEFORE generating segments.
    train, held = train_test_split(records, test_size=0.3, random_state=seed,
                                   stratify=[r['label'] for r in records])
    val, test = train_test_split(held, test_size=0.5, random_state=seed,
                                stratify=[r['label'] for r in held])
    return {'train': train, 'validation': val, 'test': test}


class SegmentDataset(Dataset):
    """Precompute small GTZAN features in RAM; retain song IDs for aggregation."""
    def __init__(self, root, records, config):
        self.items = []
        for record in records:
            y = load_audio(Path(root) / record['path'], config)
            for clip in segments(y, config):
                feature = torch.from_numpy(mel_spectrogram(clip, config))[None]
                self.items.append((feature, record['label'], record['path']))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]
