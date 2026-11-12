from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from sonicclassifier.audio import AudioConfig, GENRES, load_audio, mel_spectrogram, segments
from sonicclassifier.data import SegmentDataset, split_songs
from sonicclassifier.evaluation import evaluate
from sonicclassifier.inference import load_checkpoint, predict
from sonicclassifier.model import GenreCNN, get_device
from sonicclassifier.train import seed_everything


def test_audio_standardization(tmp_path):
    config = AudioConfig()
    # Stereo input at a different sample rate must become mono at the target rate.
    path = tmp_path / 'stereo.wav'
    sf.write(path, np.zeros((8000, 2)), 8000)
    y = load_audio(path, config)
    assert y.ndim == 1 and len(y) == 22050
    pieces = segments(np.zeros(22050 * 4), config)
    assert len(pieces) == 2
    features = [mel_spectrogram(x, config) for x in pieces]
    assert features[0].shape == features[1].shape == (64, 130)
    assert np.isfinite(features).all()
    assert np.min(features) >= 0 and np.max(features) <= 1
    with pytest.raises(ValueError):
        segments(np.array([]), config)


def test_song_split_is_reproducible_and_disjoint():
    records = [{'path': f'{genre}/{i}.wav', 'label': label}
               for label, genre in enumerate(GENRES) for i in range(10)]
    split = split_songs(records)
    assert split == split_songs(records)
    sets = [{r['path'] for r in rows} for rows in split.values()]
    assert [len(s) for s in sets] == [70, 15, 15]
    assert len(set.union(*sets)) == 100
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert all(len({r['label'] for r in rows}) == 10 for rows in split.values())


def test_training_checkpoint_and_prediction(tmp_path):
    seed_everything(42)
    torch.set_num_threads(2)
    config = AudioConfig(seconds=0.5)
    path = tmp_path / 'tone.wav'
    sf.write(path, np.sin(2 * np.pi * 440 * np.arange(22050) / 22050), 22050)
    dataset = SegmentDataset(tmp_path, [{'path': 'tone.wav', 'label': 2}], config)
    assert len(dataset) == 2
    model = GenreCNN()
    optimizer = torch.optim.Adam(model.parameters())
    x, labels, _ = next(iter(DataLoader(dataset, batch_size=2)))
    before = model.classifier[-1].weight.detach().clone()
    loss = torch.nn.functional.cross_entropy(model(x), labels)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.classifier[-1].weight)
    checkpoint = tmp_path / 'model.pt'
    torch.save({'format_version': 1, 'genres': list(GENRES), 'audio_config': config.to_dict(),
                'model_state': model.state_dict()}, checkpoint)
    restored, saved_config, _ = load_checkpoint(checkpoint, get_device('cpu'))
    probs, _ = predict(path, restored, saved_config, get_device('cpu'))
    assert probs.shape == (10,) and np.isclose(probs.sum(), 1)
    report = evaluate(restored, DataLoader(dataset, batch_size=1), get_device('cpu'))
    assert report['songs'] == 1  # Two segments must count as one song.
    assert np.asarray(report['confusion_matrix']).sum() == 1
    assert 'macro avg' in report['classification_report']
    with torch.inference_mode():
        assert torch.allclose(model.eval()(x), restored(x))


def test_streamlit_initial_page():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py').run(timeout=30)
    assert not app.exception
    assert app.title[0].value == '🎵 SonicClassifier'
