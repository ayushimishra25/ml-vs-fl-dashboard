#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from federated_intrusion_detection import (
    clone_parameters,
    compute_class_weights,
    evaluate_model,
    initialize_mlp,
    parse_hidden_dims,
    plot_confusion_matrix,
    plot_history,
    prepare_dataset,
    save_model,
    train_local_model,
)
from ids_labeling import DEFAULT_TARGET_MODE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Centralized baseline training for intrusion detection on CSV datasets."
    )
    parser.add_argument("--csv", required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--target-col",
        default="Attack_label",
        help="Target label column. Defaults to Attack_label.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["column", "derived_multiclass"],
        default=DEFAULT_TARGET_MODE,
        help="Use an existing label column or derive a multiclass attack label from the binary attack data.",
    )
    parser.add_argument(
        "--output-dir",
        default="centralized_results",
        help="Directory where metrics, plots, and the trained model will be saved.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional stratified sample size if memory is tight.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Training fraction for the stratified split.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Validation fraction for the stratified split.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of centralized training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for SGD.",
    )
    parser.add_argument(
        "--hidden-dims",
        default="128,64",
        help="Comma-separated hidden layer sizes, for example 128,64.",
    )
    parser.add_argument(
        "--l2-lambda",
        type=float,
        default=1e-4,
        help="L2 regularization strength.",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use inverse-frequency class weighting during training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation if you only need metrics and the saved model.",
    )

    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    return args


def run_training(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    dataset = prepare_dataset(
        csv_path=csv_path,
        target_col=args.target_col,
        target_mode=args.target_mode,
        max_rows=args.max_rows,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    num_classes = len(dataset.label_names)
    layer_dims = [dataset.X_train.shape[1], *hidden_dims, num_classes]
    params = initialize_mlp(layer_dims, seed=args.seed)
    class_weights = (
        compute_class_weights(dataset.y_train, num_classes)
        if args.use_class_weights
        else np.ones(num_classes, dtype=np.float32)
    )

    history_rows: List[Dict[str, float | int]] = []
    best_params = clone_parameters(params)
    best_metric = -np.inf
    best_epoch = 0

    print(f"CSV path: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Rows used: {dataset.sampled_rows}")
    print(f"Features after encoding: {dataset.X_train.shape[1]}")
    print(f"Classes: {dataset.label_names}")
    print(f"Train/Val/Test sizes: {len(dataset.y_train)}/{len(dataset.y_val)}/{len(dataset.y_test)}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        params, train_loss = train_local_model(
            global_params=params,
            X_local=dataset.X_train,
            y_local=dataset.y_train,
            local_epochs=1,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            class_weights=class_weights,
            l2_lambda=args.l2_lambda,
            seed=args.seed + epoch,
        )

        val_metrics = evaluate_model(
            X=dataset.X_val,
            y=dataset.y_val,
            params=params,
            label_names=dataset.label_names,
        )

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_weighted_f1": float(val_metrics["weighted_f1"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
        }
        history_rows.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_acc={row['val_accuracy']:.4f} | "
            f"val_macro_f1={row['val_macro_f1']:.4f}"
        )

        if row["val_macro_f1"] > best_metric:
            best_metric = row["val_macro_f1"]
            best_epoch = epoch
            best_params = clone_parameters(params)

    print("-" * 80)
    print(f"Best model selected from epoch {best_epoch} with validation macro-F1={best_metric:.4f}")

    test_metrics = evaluate_model(
        X=dataset.X_test,
        y=dataset.y_test,
        params=best_params,
        label_names=dataset.label_names,
    )
    print(
        f"Test loss={test_metrics['loss']:.4f} | "
        f"test_acc={test_metrics['accuracy']:.4f} | "
        f"test_macro_f1={test_metrics['macro_f1']:.4f} | "
        f"test_weighted_f1={test_metrics['weighted_f1']:.4f}"
    )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(output_dir / "history.csv", index=False)

    confusion_df = pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=dataset.label_names,
        columns=dataset.label_names,
    )
    confusion_df.to_csv(output_dir / "confusion_matrix.csv")

    report_df = pd.DataFrame(test_metrics["classification_report"])
    report_df.to_csv(output_dir / "classification_report.csv", index=False)

    if not args.skip_plots:
        plot_ready = history_df.rename(columns={"epoch": "round"})
        plot_history(plot_ready, output_dir / "training_curves.png")
        plot_confusion_matrix(
            matrix=test_metrics["confusion_matrix"],
            label_names=dataset.label_names,
            output_path=output_dir / "confusion_matrix.png",
        )

    save_model(
        params=best_params,
        dataset=dataset,
        output_path=output_dir / "best_model.npz",
    )

    run_summary = {
        "model_type": "centralized_mlp",
        "csv_path": str(csv_path),
        "target_col": dataset.target_col,
        "target_mode": dataset.target_mode,
        "source_target_col": dataset.source_target_col,
        "rows_used": dataset.sampled_rows,
        "num_features": dataset.X_train.shape[1],
        "label_names": dataset.label_names,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dims": hidden_dims,
        "best_epoch": best_epoch,
        "test_metrics": {
            "loss": float(test_metrics["loss"]),
            "accuracy": float(test_metrics["accuracy"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "weighted_f1": float(test_metrics["weighted_f1"]),
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        },
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
