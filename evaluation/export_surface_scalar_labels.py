import argparse
import csv
import os

import numpy as np


DEFAULT_FIELDS = (
    "final_entropy,mean_entropy,max_entropy,"
    "final_top1_prob,mean_top1_prob,min_top1_prob,"
    "final_margin,mean_margin,min_margin"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export scalar surface signals and labels from a large labeled chunk .pt file."
    )
    parser.add_argument("--label-path", required=True, help="Path to labeled chunk .pt data.")
    parser.add_argument("--label-key", default="label", help="Chunk label key. label=1 means prefix-correct.")
    parser.add_argument("--fields", default=DEFAULT_FIELDS, help="Comma-separated scalar fields to export.")
    parser.add_argument("--output-path", required=True, help="Output CSV path.")
    return parser.parse_args()


def parse_csv(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def scalar_value(value):
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().to(torch.float32).numpy()
    return float(np.asarray(value, dtype=np.float32).reshape(-1)[0])


def main():
    import torch

    args = parse_args()
    fields = parse_csv(args.fields)
    rows = torch.load(args.label_path, map_location="cpu", weights_only=False)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    columns = ["label", "error_label"] + fields
    written = 0
    with open(args.output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            label = int(row.get(args.label_key, -1))
            if label not in {0, 1}:
                continue
            output = {"label": label, "error_label": 1 - label}
            missing = False
            for field in fields:
                if field not in row:
                    missing = True
                    break
                output[field] = scalar_value(row[field])
            if missing:
                continue
            writer.writerow(output)
            written += 1

    print(f"Wrote: {args.output_path}")
    print(f"Rows: {written}")


if __name__ == "__main__":
    main()
