#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ids_labeling import DEFAULT_TARGET_MODE, apply_target_mode


ArrayDict = Dict[str, np.ndarray]


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    label_names: List[str]
    mean: np.ndarray
    std: np.ndarray
    sampled_rows: int
    target_col: str
    target_mode: str
    source_target_col: str
    drop_columns: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated learning for intrusion detection on CSV datasets."
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
        default="federated_results",
        help="Directory where metrics, plots, and the trained model will be saved.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional stratified sample size if you do not want to use the full CSV.",
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
        "--num-clients",
        type=int,
        default=5,
        help="Number of federated clients to simulate.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=15,
        help="Number of global federated rounds.",
    )
    parser.add_argument(
        "--client-fraction",
        type=float,
        default=1.0,
        help="Fraction of clients participating in each round.",
    )
    parser.add_argument(
        "--partition",
        choices=["iid", "dirichlet"],
        default="dirichlet",
        help="How training data is split across clients.",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.8,
        help="Dirichlet concentration for non-IID partitioning.",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=2,
        help="Number of local epochs per round.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Mini-batch size for local training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for local SGD.",
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
    if not 0.0 < args.train_frac < 1.0:
        raise ValueError("--train-frac must be between 0 and 1.")
    if not 0.0 <= args.val_frac < 1.0:
        raise ValueError("--val-frac must be between 0 and 1.")
    if args.train_frac + args.val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1.0.")
    if args.num_clients < 2:
        raise ValueError("--num-clients must be at least 2.")
    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1.")
    if not 0.0 < args.client_fraction <= 1.0:
        raise ValueError("--client-fraction must be in (0, 1].")
    if args.partition == "dirichlet" and args.dirichlet_alpha <= 0.0:
        raise ValueError("--dirichlet-alpha must be positive.")
    return args


def parse_hidden_dims(raw_value: str) -> List[int]:
    dims = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value <= 0:
            raise ValueError("Hidden dimensions must be positive integers.")
        dims.append(value)
    if not dims:
        raise ValueError("--hidden-dims must contain at least one layer size.")
    return dims


def allocate_counts(total: int, fractions: np.ndarray) -> np.ndarray:
    raw = fractions * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(counts.sum())
    if remainder > 0:
        priorities = np.argsort(raw - counts)[::-1]
        counts[priorities[:remainder]] += 1
    return counts


def stratified_sample_dataframe(
    df: pd.DataFrame, target_col: str, max_rows: int, seed: int
) -> pd.DataFrame:
    if max_rows is None or max_rows >= len(df):
        return df

    rng = np.random.default_rng(seed)
    class_counts = df[target_col].value_counts(sort=False)
    fractions = class_counts.to_numpy(dtype=np.float64) / len(df)
    desired_counts = allocate_counts(max_rows, fractions)

    sampled_groups = []
    for idx, label in enumerate(class_counts.index):
        group = df[df[target_col] == label]
        take = min(int(desired_counts[idx]), len(group))
        if take == 0:
            continue
        sample_positions = rng.choice(len(group), size=take, replace=False)
        sampled_groups.append(group.iloc[sample_positions])

    sampled = pd.concat(sampled_groups, axis=0)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled


def stratified_split_indices(
    y: np.ndarray, train_frac: float, val_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: List[np.ndarray] = []
    val_indices: List[np.ndarray] = []
    test_indices: List[np.ndarray] = []

    unique_labels = np.unique(y)
    split_fractions = np.array([train_frac, val_frac, 1.0 - train_frac - val_frac])

    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)
        counts = allocate_counts(len(label_indices), split_fractions)
        train_end = counts[0]
        val_end = train_end + counts[1]
        train_indices.append(label_indices[:train_end])
        val_indices.append(label_indices[train_end:val_end])
        test_indices.append(label_indices[val_end:])

    train_idx = np.concatenate(train_indices)
    val_idx = np.concatenate(val_indices)
    test_idx = np.concatenate(test_indices)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def prepare_dataset(
    csv_path: Path,
    target_col: str,
    target_mode: str,
    max_rows: int | None,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> DatasetBundle:
    df = pd.read_csv(csv_path)
    df, effective_target_col, source_target_col, drop_columns = apply_target_mode(
        df=df,
        target_col=target_col,
        target_mode=target_mode,
    )

    df = stratified_sample_dataframe(df, effective_target_col, max_rows, seed)
    sampled_rows = len(df)

    target = df[effective_target_col].copy()
    features = df.drop(columns=drop_columns, errors="ignore").copy()

    bool_columns = [col for col in features.columns if features[col].dtype == bool]
    for column in bool_columns:
        features[column] = features[column].astype(np.int8)

    features = pd.get_dummies(features, drop_first=False)
    feature_names = features.columns.astype(str).tolist()

    label_codes, unique_labels = pd.factorize(target, sort=True)
    if len(unique_labels) < 2:
        raise ValueError("The target column must contain at least 2 classes.")
    label_names = [str(value) for value in unique_labels.tolist()]

    X = features.to_numpy(dtype=np.float32, copy=True)
    y = label_codes.astype(np.int64)

    train_idx, val_idx, test_idx = stratified_split_indices(
        y=y, train_frac=train_frac, val_frac=val_frac, seed=seed
    )

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    mean = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0

    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)

    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        label_names=label_names,
        mean=mean,
        std=std,
        sampled_rows=sampled_rows,
        target_col=effective_target_col,
        target_mode=target_mode,
        source_target_col=source_target_col,
        drop_columns=drop_columns,
    )


def clone_parameters(params: ArrayDict) -> ArrayDict:
    return {name: value.copy() for name, value in params.items()}


def initialize_mlp(layer_dims: List[int], seed: int) -> ArrayDict:
    rng = np.random.default_rng(seed)
    params: ArrayDict = {}
    last_layer_index = len(layer_dims) - 1

    for layer in range(1, len(layer_dims)):
        fan_in = layer_dims[layer - 1]
        fan_out = layer_dims[layer]
        scale = math.sqrt(2.0 / fan_in)
        if layer == last_layer_index:
            scale = math.sqrt(1.0 / fan_in)
        params[f"W{layer}"] = rng.normal(
            loc=0.0, scale=scale, size=(fan_in, fan_out)
        ).astype(np.float32)
        params[f"b{layer}"] = np.zeros((1, fan_out), dtype=np.float32)
    return params


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def forward_pass(X: np.ndarray, params: ArrayDict) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    cache: Dict[str, np.ndarray] = {"A0": X}
    num_layers = len(params) // 2
    activations = X

    for layer in range(1, num_layers):
        z = activations @ params[f"W{layer}"] + params[f"b{layer}"]
        activations = np.maximum(z, 0.0)
        cache[f"Z{layer}"] = z
        cache[f"A{layer}"] = activations

    logits = activations @ params[f"W{num_layers}"] + params[f"b{num_layers}"]
    cache[f"Z{num_layers}"] = logits
    return logits, cache


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    class_counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    total = class_counts.sum()
    weights = total / (num_classes * np.maximum(class_counts, 1.0))
    return weights / weights.mean()


def loss_and_gradients(
    X_batch: np.ndarray,
    y_batch: np.ndarray,
    params: ArrayDict,
    class_weights: np.ndarray,
    l2_lambda: float,
) -> Tuple[float, ArrayDict]:
    logits, cache = forward_pass(X_batch, params)
    probabilities = softmax(logits)

    sample_weights = class_weights[y_batch]
    log_probs = -np.log(probabilities[np.arange(len(y_batch)), y_batch] + 1e-12)
    data_loss = np.mean(log_probs * sample_weights)
    reg_loss = 0.5 * l2_lambda * sum(
        np.sum(params[name] ** 2) for name in params if name.startswith("W")
    )
    total_loss = float(data_loss + reg_loss)

    num_layers = len(params) // 2
    grads: ArrayDict = {}

    dlogits = probabilities
    dlogits[np.arange(len(y_batch)), y_batch] -= 1.0
    dlogits *= sample_weights[:, None]
    dlogits /= len(y_batch)

    upstream = dlogits
    for layer in range(num_layers, 0, -1):
        prev_activation = cache["A0"] if layer == 1 else cache[f"A{layer - 1}"]
        grads[f"W{layer}"] = prev_activation.T @ upstream + l2_lambda * params[f"W{layer}"]
        grads[f"b{layer}"] = upstream.sum(axis=0, keepdims=True)

        if layer > 1:
            upstream = upstream @ params[f"W{layer}"].T
            upstream[cache[f"Z{layer - 1}"] <= 0.0] = 0.0

    return total_loss, grads


def train_local_model(
    global_params: ArrayDict,
    X_local: np.ndarray,
    y_local: np.ndarray,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    class_weights: np.ndarray,
    l2_lambda: float,
    seed: int,
) -> Tuple[ArrayDict, float]:
    local_params = clone_parameters(global_params)
    rng = np.random.default_rng(seed)
    losses: List[float] = []

    for _ in range(local_epochs):
        permutation = rng.permutation(len(y_local))
        for start in range(0, len(y_local), batch_size):
            batch_idx = permutation[start : start + batch_size]
            X_batch = X_local[batch_idx]
            y_batch = y_local[batch_idx]
            loss, grads = loss_and_gradients(
                X_batch=X_batch,
                y_batch=y_batch,
                params=local_params,
                class_weights=class_weights,
                l2_lambda=l2_lambda,
            )
            losses.append(loss)
            for name in local_params:
                local_params[name] -= learning_rate * grads[name]

    return local_params, float(np.mean(losses))


def fedavg(local_models: List[ArrayDict], sample_counts: List[int]) -> ArrayDict:
    total_samples = float(sum(sample_counts))
    aggregated: ArrayDict = {}

    for name in local_models[0]:
        combined = np.zeros_like(local_models[0][name])
        for model, count in zip(local_models, sample_counts):
            combined += model[name] * (count / total_samples)
        aggregated[name] = combined.astype(np.float32)
    return aggregated


def make_client_partitions(
    y: np.ndarray,
    num_clients: int,
    strategy: str,
    alpha: float,
    min_client_size: int,
    seed: int,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)

    if strategy == "iid":
        shuffled = rng.permutation(len(y))
        return [split.astype(np.int64) for split in np.array_split(shuffled, num_clients)]

    unique_labels = np.unique(y)
    attempts = 0
    adjusted_min_client_size = min(min_client_size, max(1, len(y) // (2 * num_clients)))

    while attempts < 200:
        client_buckets = [[] for _ in range(num_clients)]
        for label in unique_labels:
            label_indices = np.where(y == label)[0]
            rng.shuffle(label_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            counts = rng.multinomial(len(label_indices), proportions)
            splits = np.split(label_indices, np.cumsum(counts)[:-1])
            for client_id, split in enumerate(splits):
                client_buckets[client_id].extend(split.tolist())

        client_sizes = [len(bucket) for bucket in client_buckets]
        if min(client_sizes) >= adjusted_min_client_size:
            partitions = []
            for bucket in client_buckets:
                client_indices = np.array(bucket, dtype=np.int64)
                rng.shuffle(client_indices)
                partitions.append(client_indices)
            return partitions
        attempts += 1

    raise RuntimeError(
        "Could not create a Dirichlet partition with the current alpha and client count. "
        "Try increasing --dirichlet-alpha or decreasing --num-clients."
    )


def predict_proba(X: np.ndarray, params: ArrayDict) -> np.ndarray:
    logits, _ = forward_pass(X, params)
    return softmax(logits)


def predict_labels(X: np.ndarray, params: ArrayDict) -> np.ndarray:
    return np.argmax(predict_proba(X, params), axis=1)


def confusion_matrix_np(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def per_class_report(
    matrix: np.ndarray, label_names: List[str]
) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    for class_id, label_name in enumerate(label_names):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum() - tp)
        fn = int(matrix[class_id, :].sum() - tp)
        support = int(matrix[class_id, :].sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rows.append(
            {
                "class_id": class_id,
                "label": label_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return rows


def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    params: ArrayDict,
    label_names: List[str],
) -> Dict[str, object]:
    probabilities = predict_proba(X, params)
    predictions = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix_np(y, predictions, num_classes=len(label_names))

    loss = float(
        -np.log(probabilities[np.arange(len(y)), y] + 1e-12).mean()
    )
    accuracy = float((predictions == y).mean())

    rows = per_class_report(matrix, label_names)
    macro_f1 = float(np.mean([row["f1"] for row in rows]))
    weighted_f1 = float(
        np.average(
            [row["f1"] for row in rows],
            weights=[row["support"] for row in rows],
        )
    )
    balanced_accuracy = float(np.mean([row["recall"] for row in rows]))

    return {
        "loss": loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": matrix,
        "classification_report": rows,
        "predictions": predictions,
    }


def get_pyplot(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_history(history_df: pd.DataFrame, output_path: Path) -> None:
    plt = get_pyplot(output_path.parent / ".mplconfig")
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_df["round"], history_df["val_loss"], marker="o", linewidth=1.8)
    plt.xlabel("Round")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss")
    plt.grid(alpha=0.25)

    plt.subplot(1, 2, 2)
    plt.plot(history_df["round"], history_df["val_accuracy"], marker="o", label="Accuracy")
    plt.plot(history_df["round"], history_df["val_macro_f1"], marker="o", label="Macro F1")
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.grid(alpha=0.25)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    matrix: np.ndarray, label_names: List[str], output_path: Path
) -> None:
    plt = get_pyplot(output_path.parent / ".mplconfig")
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    positions = np.arange(len(label_names))
    plt.xticks(positions, label_names, rotation=45, ha="right")
    plt.yticks(positions, label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_model(
    params: ArrayDict,
    dataset: DatasetBundle,
    output_path: Path,
) -> None:
    np.savez_compressed(
        output_path,
        **params,
        mean=dataset.mean.astype(np.float32),
        std=dataset.std.astype(np.float32),
        feature_names=np.array(dataset.feature_names, dtype=object),
        label_names=np.array(dataset.label_names, dtype=object),
        target_col=np.array([dataset.target_col], dtype=object),
        target_mode=np.array([dataset.target_mode], dtype=object),
        source_target_col=np.array([dataset.source_target_col], dtype=object),
        drop_columns=np.array(dataset.drop_columns, dtype=object),
    )


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
    global_params = initialize_mlp(layer_dims, seed=args.seed)

    class_weights = (
        compute_class_weights(dataset.y_train, num_classes)
        if args.use_class_weights
        else np.ones(num_classes, dtype=np.float32)
    )

    min_client_size = max(32, min(args.batch_size, len(dataset.y_train) // args.num_clients))
    client_partitions = make_client_partitions(
        y=dataset.y_train,
        num_clients=args.num_clients,
        strategy=args.partition,
        alpha=args.dirichlet_alpha,
        min_client_size=min_client_size,
        seed=args.seed,
    )

    client_sizes = [len(indices) for indices in client_partitions]
    history_rows: List[Dict[str, float | int]] = []
    best_params = clone_parameters(global_params)
    best_metric = -np.inf
    best_round = 0

    rng = np.random.default_rng(args.seed)
    clients_per_round = max(1, math.ceil(args.client_fraction * args.num_clients))

    print(f"CSV path: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Rows used: {dataset.sampled_rows}")
    print(f"Features after encoding: {dataset.X_train.shape[1]}")
    print(f"Classes: {dataset.label_names}")
    print(f"Train/Val/Test sizes: {len(dataset.y_train)}/{len(dataset.y_val)}/{len(dataset.y_test)}")
    print(f"Client sizes: {client_sizes}")
    print("-" * 80)

    for round_idx in range(1, args.rounds + 1):
        selected_clients = rng.choice(
            args.num_clients, size=clients_per_round, replace=False
        )
        local_models: List[ArrayDict] = []
        local_losses: List[float] = []
        local_counts: List[int] = []

        for client_id in selected_clients:
            client_indices = client_partitions[client_id]
            local_model, local_loss = train_local_model(
                global_params=global_params,
                X_local=dataset.X_train[client_indices],
                y_local=dataset.y_train[client_indices],
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                class_weights=class_weights,
                l2_lambda=args.l2_lambda,
                seed=args.seed + round_idx * 1000 + int(client_id),
            )
            local_models.append(local_model)
            local_losses.append(local_loss)
            local_counts.append(len(client_indices))

        global_params = fedavg(local_models, local_counts)
        val_metrics = evaluate_model(
            X=dataset.X_val,
            y=dataset.y_val,
            params=global_params,
            label_names=dataset.label_names,
        )

        round_row = {
            "round": round_idx,
            "client_count": len(selected_clients),
            "avg_local_loss": float(np.mean(local_losses)),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_weighted_f1": float(val_metrics["weighted_f1"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
        }
        history_rows.append(round_row)

        print(
            f"Round {round_idx:02d}/{args.rounds} | "
            f"local_loss={round_row['avg_local_loss']:.4f} | "
            f"val_loss={round_row['val_loss']:.4f} | "
            f"val_acc={round_row['val_accuracy']:.4f} | "
            f"val_macro_f1={round_row['val_macro_f1']:.4f}"
        )

        if round_row["val_macro_f1"] > best_metric:
            best_metric = round_row["val_macro_f1"]
            best_round = round_idx
            best_params = clone_parameters(global_params)

    print("-" * 80)
    print(f"Best model selected from round {best_round} with validation macro-F1={best_metric:.4f}")

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
        plot_history(history_df, output_dir / "training_curves.png")
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
        "csv_path": str(csv_path),
        "target_col": dataset.target_col,
        "target_mode": dataset.target_mode,
        "source_target_col": dataset.source_target_col,
        "rows_used": dataset.sampled_rows,
        "num_features": dataset.X_train.shape[1],
        "label_names": dataset.label_names,
        "num_clients": args.num_clients,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dims": hidden_dims,
        "partition": args.partition,
        "dirichlet_alpha": args.dirichlet_alpha,
        "best_round": best_round,
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
