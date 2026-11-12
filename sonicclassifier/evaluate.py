"""Evaluate a checkpoint on its persisted held-out songs."""
import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from .data import SegmentDataset
from .evaluation import evaluate, save_report
from .inference import load_checkpoint
from .model import get_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    args = p.parse_args()
    device = get_device(args.device)
    model, config, _ = load_checkpoint(args.run / 'best.pt', device)
    records = json.loads((args.run / 'splits.json').read_text())['test']
    report = evaluate(model, DataLoader(SegmentDataset(args.data, records, config), batch_size=32), device)
    save_report(report, args.run)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
