#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np
import pandas as pd

from federated_intrusion_detection import predict_proba
from ids_labeling import DEFAULT_TARGET_MODE, humanize_label, prepare_dataframe_for_model


ArrayDict = Dict[str, np.ndarray]


@dataclass
class LoadedModel:
    name: str
    params: ArrayDict
    mean: np.ndarray
    std: np.ndarray
    feature_names: List[str]
    label_names: List[str]
    target_col: str
    target_mode: str
    source_target_col: str
    drop_columns: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate live IDS traffic and compare centralized ML vs federated FL predictions."
    )
    parser.add_argument("--csv", required=True, help="Path to the source CSV file.")
    parser.add_argument(
        "--ml-model",
        required=True,
        help="Path to the centralized best_model.npz file.",
    )
    parser.add_argument(
        "--fl-model",
        required=True,
        help="Path to the federated best_model.npz file.",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=25,
        help="Number of simulated events to stream.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds to wait between events. Use 0 for instant playback.",
    )
    parser.add_argument(
        "--mode",
        choices=["replay", "synthetic"],
        default="synthetic",
        help="Replay rows from the dataset or create light synthetic variants.",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["natural", "balanced"],
        default="balanced",
        help="Sample rows in their original frequency or balance evenly across classes.",
    )
    parser.add_argument(
        "--focus-class",
        default="all",
        help="Optional class label to oversample during the stream. Use 'all' to disable.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.05,
        help="Noise scale for synthetic numeric perturbations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-log",
        default=None,
        help="Optional CSV file to store the streamed predictions.",
    )
    args = parser.parse_args()

    if args.events < 1:
        raise ValueError("--events must be at least 1.")
    if args.interval < 0:
        raise ValueError("--interval must be non-negative.")
    if args.noise_scale < 0.0:
        raise ValueError("--noise-scale must be non-negative.")
    return args


def load_model(model_path: Path, name: str) -> LoadedModel:
    archive = np.load(model_path, allow_pickle=True)
    params: ArrayDict = {}
    for key in archive.files:
        if key.startswith("W") or key.startswith("b"):
            params[key] = archive[key].astype(np.float32)

    target_col = str(archive["target_col"].tolist()[0])
    target_mode = (
        str(archive["target_mode"].tolist()[0])
        if "target_mode" in archive.files
        else DEFAULT_TARGET_MODE
    )
    source_target_col = (
        str(archive["source_target_col"].tolist()[0])
        if "source_target_col" in archive.files
        else target_col
    )
    drop_columns = (
        [str(item) for item in archive["drop_columns"].tolist()]
        if "drop_columns" in archive.files
        else [target_col]
    )

    return LoadedModel(
        name=name,
        params=params,
        mean=archive["mean"].astype(np.float32),
        std=archive["std"].astype(np.float32),
        feature_names=[str(item) for item in archive["feature_names"].tolist()],
        label_names=[str(item) for item in archive["label_names"].tolist()],
        target_col=target_col,
        target_mode=target_mode,
        source_target_col=source_target_col,
        drop_columns=drop_columns,
    )


def encode_rows(rows: pd.DataFrame, model: LoadedModel) -> np.ndarray:
    features = rows.drop(columns=model.drop_columns, errors="ignore").copy()
    bool_columns = [col for col in features.columns if features[col].dtype == bool]
    for column in bool_columns:
        features[column] = features[column].astype(np.int8)

    features = pd.get_dummies(features, drop_first=False)
    features = features.reindex(columns=model.feature_names, fill_value=0)
    X = features.to_numpy(dtype=np.float32, copy=True)
    X = (X - model.mean) / model.std
    return X.astype(np.float32)


def predict_label(rows: pd.DataFrame, model: LoadedModel) -> tuple[str, float]:
    X = encode_rows(rows, model)
    probabilities = predict_proba(X, model.params)[0]
    best_index = int(np.argmax(probabilities))
    return model.label_names[best_index], float(probabilities[best_index])


def choose_pool(
    source_df: pd.DataFrame,
    target_col: str,
    sampling_strategy: str,
    focus_class: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if target_col not in source_df.columns:
        return source_df

    string_labels = source_df[target_col].astype(str)
    requested_focus = str(focus_class)
    if requested_focus.lower() != "all":
        focused_pool = source_df[string_labels == requested_focus]
        return focused_pool if not focused_pool.empty else source_df

    if sampling_strategy == "balanced":
        unique_labels = sorted(string_labels.unique().tolist())
        chosen_label = unique_labels[int(rng.integers(len(unique_labels)))]
        balanced_pool = source_df[string_labels == chosen_label]
        return balanced_pool if not balanced_pool.empty else source_df

    return source_df


def synthetic_row(
    source_df: pd.DataFrame,
    target_col: str,
    numeric_columns: List[str],
    column_scales: Dict[str, float],
    sampling_strategy: str,
    focus_class: str,
    rng: np.random.Generator,
) -> pd.Series:
    pool = choose_pool(
        source_df=source_df,
        target_col=target_col,
        sampling_strategy=sampling_strategy,
        focus_class=focus_class,
        rng=rng,
    )

    row = pool.iloc[int(rng.integers(len(pool)))].copy()
    for column in numeric_columns:
        base_value = float(row[column])
        row[column] = max(0.0, base_value + rng.normal(0.0, column_scales[column]))
    return row


def event_stream(
    df: pd.DataFrame,
    mode: str,
    target_col: str,
    events: int,
    sampling_strategy: str,
    focus_class: str,
    noise_scale: float,
    seed: int,
) -> Iterator[pd.Series]:
    rng = np.random.default_rng(seed)
    numeric_columns = [
        column
        for column in df.columns
        if column != target_col and pd.api.types.is_float_dtype(df[column].dtype)
    ]
    column_scales = {
        column: max(float(df[column].std(ddof=0)) * noise_scale, 1e-6)
        for column in numeric_columns
    }

    for _ in range(events):
        if mode == "replay":
            pool = choose_pool(
                source_df=df,
                target_col=target_col,
                sampling_strategy=sampling_strategy,
                focus_class=focus_class,
                rng=rng,
            )
            yield pool.iloc[int(rng.integers(len(pool)))].copy()
        else:
            yield synthetic_row(
                source_df=df,
                target_col=target_col,
                numeric_columns=numeric_columns,
                column_scales=column_scales,
                sampling_strategy=sampling_strategy,
                focus_class=focus_class,
                rng=rng,
            )


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    ml_model_path = Path(args.ml_model).expanduser().resolve()
    fl_model_path = Path(args.fl_model).expanduser().resolve()

    source_df = pd.read_csv(csv_path)
    ml_model = load_model(ml_model_path, "ML")
    fl_model = load_model(fl_model_path, "FL")

    if ml_model.target_col != fl_model.target_col:
        raise ValueError("ML and FL models were trained with different target columns.")
    if ml_model.target_mode != fl_model.target_mode:
        raise ValueError("ML and FL models were trained with different target modes.")

    df, target_col = prepare_dataframe_for_model(
        df=source_df,
        target_mode=ml_model.target_mode,
        target_col=ml_model.target_col,
        source_target_col=ml_model.source_target_col,
        drop_columns=ml_model.drop_columns,
    )

    log_rows: List[Dict[str, object]] = []
    ml_correct = 0
    fl_correct = 0
    agreement_count = 0
    has_ground_truth = target_col in df.columns

    print(f"Source CSV: {csv_path}")
    print(f"Centralized model: {ml_model_path}")
    print(f"Federated model: {fl_model_path}")
    print(f"Simulation mode: {args.mode}")
    print(
        f"Events: {args.events} | Interval: {args.interval} seconds | "
        f"Sampling: {args.sampling_strategy} | Focus: {args.focus_class}"
    )
    print("-" * 140)

    stream = event_stream(
        df=df,
        mode=args.mode,
        target_col=target_col,
        events=args.events,
        sampling_strategy=args.sampling_strategy,
        focus_class=args.focus_class,
        noise_scale=args.noise_scale,
        seed=args.seed,
    )

    for event_id, event in enumerate(stream, start=1):
        event_frame = pd.DataFrame([event])
        true_label = str(event[target_col]) if has_ground_truth else "unknown"

        ml_label, ml_confidence = predict_label(event_frame, ml_model)
        fl_label, fl_confidence = predict_label(event_frame, fl_model)

        true_human = humanize_label(true_label) if has_ground_truth else "Unknown"
        ml_human = humanize_label(ml_label)
        fl_human = humanize_label(fl_label)

        if has_ground_truth and ml_label == true_label:
            ml_correct += 1
        if has_ground_truth and fl_label == true_label:
            fl_correct += 1
        if ml_label == fl_label:
            agreement_count += 1

        print(
            f"Event {event_id:03d} | "
            f"true={true_human:20s} | "
            f"ML={ml_human:20s} ({ml_confidence:.3f}) | "
            f"FL={fl_human:20s} ({fl_confidence:.3f}) | "
            f"agree={'yes' if ml_label == fl_label else 'no'}"
        )

        log_rows.append(
            {
                "event_id": event_id,
                "true_label": true_label,
                "true_label_name": true_human,
                "ml_label": ml_label,
                "ml_label_name": ml_human,
                "ml_confidence": ml_confidence,
                "fl_label": fl_label,
                "fl_label_name": fl_human,
                "fl_confidence": fl_confidence,
                "models_agree": int(ml_label == fl_label),
            }
        )

        if args.interval > 0:
            time.sleep(args.interval)

    print("-" * 140)
    if has_ground_truth:
        print(
            f"Centralized accuracy on streamed events: {ml_correct / args.events:.4f} | "
            f"Federated accuracy on streamed events: {fl_correct / args.events:.4f}"
        )
    print(f"Model agreement rate: {agreement_count / args.events:.4f}")

    if args.output_log:
        output_path = Path(args.output_log).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(log_rows).to_csv(output_path, index=False)
        print(f"Saved event log to: {output_path}")


if __name__ == "__main__":
    main()
