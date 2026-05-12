#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    workspace = Path(
        "/Users/ayushimishra/Documents/Codex/2026-04-26/files-mentioned-by-the-user-w"
    )
    ml_model = workspace / "centralized_results" / "best_model.npz"
    fl_model = workspace / "federated_results" / "best_model.npz"

    missing = [str(path) for path in [ml_model, fl_model] if not path.exists()]
    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(
            "Train both models before starting the dashboard. Missing files:\n"
            f"{missing_text}"
        )

    for model_path in [ml_model, fl_model]:
        metadata = np.load(model_path, allow_pickle=True)
        target_mode = (
            str(metadata["target_mode"].tolist()[0])
            if "target_mode" in metadata.files
            else "column"
        )
        if target_mode != "derived_multiclass":
            raise ValueError(
                "The saved models are still in binary mode. Re-run "
                "`run_centralized_ids_one_click.py` and `run_federated_ids_one_click.py` "
                "to regenerate multiclass models before starting the dashboard."
            )

    command = [
        sys.executable,
        str(workspace / "ids_dashboard_server.py"),
        "--csv",
        "/Users/ayushimishra/Downloads/W-IIOTID_cleaned_and_encoded (1).csv",
        "--ml-model",
        str(ml_model),
        "--fl-model",
        str(fl_model),
        "--host",
        "127.0.0.1",
        "--port",
        "8050",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
