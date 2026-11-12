import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from .audio import GENRES


def evaluate(model, loader, device):
    songs = {}
    model.eval()
    with torch.inference_mode():
        for x, labels, ids in loader:
            probabilities = model(x.to(device)).softmax(1).cpu().numpy()
            for probability, label, song in zip(probabilities, labels.tolist(), ids):
                if song not in songs:
                    songs[song] = (label, [])
                songs[song][1].append(probability)
    if not songs:
        raise ValueError('Cannot evaluate an empty dataset')
    true = [label for label, _ in songs.values()]
    pred = [int(np.mean(probs, axis=0).argmax()) for _, probs in songs.values()]
    report = classification_report(true, pred, labels=list(range(10)), target_names=GENRES,
                                   output_dict=True, zero_division=0)
    return {'accuracy': accuracy_score(true, pred), 'classification_report': report,
            'confusion_matrix': confusion_matrix(true, pred, labels=list(range(10))).tolist(),
            'songs': len(songs), 'aggregation': 'mean segment softmax probabilities'}


def save_report(report, output):
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay(np.array(report['confusion_matrix']), display_labels=GENRES).plot(ax=ax, xticks_rotation=45)
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png')
    plt.close(fig)
