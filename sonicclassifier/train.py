"""Run with python -m sonicclassifier.train --data data/genres_original."""
import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .audio import AudioConfig, GENRES
from .data import SegmentDataset, discover, split_songs
from .evaluation import evaluate, save_report
from .inference import load_checkpoint
from .model import GenreCNN, get_device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/run'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error('epochs, batch size, and learning rate must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('Output directory must be empty; choose a new run directory')
    seed_everything(args.seed)
    device = get_device(args.device)
    config = AudioConfig()
    splits = split_songs(discover(args.data), args.seed)
    (args.output / 'splits.json').write_text(json.dumps(splits, indent=2))
    loaders = {}
    for name in ('train', 'validation'):
        dataset = SegmentDataset(args.data, splits[name], config)
        loaders[name] = DataLoader(dataset, batch_size=args.batch_size, shuffle=name == 'train',
                                   generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    model = GenreCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    history, best = [], -1
    metadata = {'seed': args.seed, 'device': str(device), 'torch_version': str(torch.__version__),
                'audio_config': config.to_dict(), 'epochs': args.epochs, 'batch_size': args.batch_size,
                'learning_rate': args.learning_rate, 'data': str(args.data.resolve())}
    (args.output / 'config.json').write_text(json.dumps(metadata, indent=2))
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = count = 0
        for x, labels, _ in loaders['train']:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            count += len(labels)
        validation = evaluate(model, loaders['validation'], device)
        row = {'epoch': epoch, 'train_loss': loss_sum / count, 'validation_accuracy': validation['accuracy']}
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {'format_version': 1, 'genres': list(GENRES), 'audio_config': config.to_dict(),
                      'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                      'epoch': epoch, 'metadata': metadata}
        torch.save(checkpoint, args.output / 'last.pt')
        if validation['accuracy'] > best:
            best = validation['accuracy']
            torch.save(checkpoint, args.output / 'best.pt')
        (args.output / 'history.json').write_text(json.dumps(history, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r['epoch'] for r in history], [r['train_loss'] for r in history])
    axes[0].set(xlabel='Epoch', ylabel='Training cross entropy')
    axes[1].plot([r['epoch'] for r in history], [r['validation_accuracy'] for r in history])
    axes[1].set(xlabel='Epoch', ylabel='Validation song accuracy', ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(args.output / 'history.png')
    plt.close(fig)
    # Test set is first evaluated after selecting the checkpoint on validation.
    model, _, _ = load_checkpoint(args.output / 'best.pt', device)
    test_loader = DataLoader(SegmentDataset(args.data, splits['test'], config), batch_size=args.batch_size)
    save_report(evaluate(model, test_loader, device), args.output)


if __name__ == '__main__':
    main()
