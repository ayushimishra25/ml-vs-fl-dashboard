#!/usr/bin/env python3
from __future__ import annotations

import argparse

from centralized_intrusion_detection import run_training


def build_default_args() -> argparse.Namespace:
    return argparse.Namespace(
        csv="/Users/ayushimishra/Downloads/W-IIOTID_cleaned_and_encoded (1).csv",
        target_col="Attack_label",
        target_mode="derived_multiclass",
        output_dir="/Users/ayushimishra/Documents/Codex/2026-04-26/files-mentioned-by-the-user-w/centralized_results",
        max_rows=None,
        train_frac=0.70,
        val_frac=0.15,
        epochs=15,
        batch_size=512,
        learning_rate=0.01,
        hidden_dims="128,64",
        l2_lambda=1e-4,
        use_class_weights=True,
        seed=42,
        skip_plots=False,
    )


def main() -> None:
    args = build_default_args()
    run_training(args)


if __name__ == "__main__":
    main()
