"""End-to-end plumbing validation with synthetic tones, not a music benchmark."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf

# Support running this script directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sonicclassifier.audio import GENRES
from sonicclassifier.inference import load_checkpoint, predict
from sonicclassifier.model import get_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = args.output / 'data'
    rng = np.random.default_rng(42)
    for label, genre in enumerate(GENRES):
        folder = data / genre
        folder.mkdir(parents=True)
        for song in range(10):
            t = np.arange(11025) / 22050
            y = 0.4 * np.sin(2 * np.pi * (150 + 100 * label + song) * t) + rng.normal(0, 0.01, len(t))
            sf.write(folder / f'{song}.wav', y, 22050)
    run = args.output / 'run'
    environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    subprocess.run([sys.executable, '-m', 'sonicclassifier.train', '--data', str(data),
                    '--output', str(run), '--epochs', '1', '--device', 'cpu'], check=True, env=environment)
    subprocess.run([sys.executable, '-m', 'sonicclassifier.evaluate', '--data', str(data),
                    '--run', str(run), '--device', 'cpu'], check=True, env=environment, stdout=subprocess.DEVNULL)
    model, config, _ = load_checkpoint(run / 'best.pt', get_device('cpu'))
    scores, _ = predict(data / 'blues/0.wav', model, config, get_device('cpu'))
    assert scores.shape == (10,) and np.isclose(scores.sum(), 1)
    print('Synthetic pipeline passed. No real music performance was measured.')


if __name__ == '__main__':
    main()
