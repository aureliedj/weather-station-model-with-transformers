"""
main.py

Entry point for the Station-MAE project.
Parses config, sets up data, builds model, runs training.

Usage:
    python main.py
    python main.py --config configs/base.yaml
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Station-MAE training entry point")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Starting Station-MAE with config: {args.config}")

    # ------------------------------------------------------------------ #
    # These imports will expand as each module is implemented             #
    # ------------------------------------------------------------------ #

    # 1. Load config
    # from utils.config import load_config
    # cfg = load_config(args.config)

    # 2. Build dataset
    # from data.dataset import StationMAEDataset
    # train_ds = StationMAEDataset(cfg)

    # 3. Build model
    # from model.mae import StationMAE
    # model = StationMAE(cfg)

    # 4. Run training
    # from engine.train import train
    # train(model, train_ds, cfg)

    print("main.py reached — ready to wire up components.")


if __name__ == "__main__":
    main()