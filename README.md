# Federated Intrusion Detection

This project trains centralized and federated neural networks on your CSV using only `numpy`, `pandas`, and `matplotlib`.

Important note: the attached file exposes one ground-truth label column, `Attack_label`, and it contains 2 classes (`0` and `1`). To support a richer multiclass demo on this dataset, the code now includes a `derived_multiclass` target mode that splits attack traffic into multiple attack families using the available protocol, service, port, login, file, and privilege features. That multiclass target is derived, not native ground truth.

## Files

- `federated_intrusion_detection.py`: end-to-end training and evaluation script
- `centralized_intrusion_detection.py`: centralized baseline trainer
- `live_ids_simulator.py`: live stream simulator that compares centralized ML and federated FL predictions
- `ids_dashboard_server.py`: local dashboard server for watching the stream visually
- `requirements.txt`: minimal dependencies

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run on your dataset

```bash
python3 federated_intrusion_detection.py \
  --csv "/Users/ayushimishra/Downloads/W-IIOTID_cleaned_and_encoded (1).csv" \
  --target-col Attack_label \
  --target-mode derived_multiclass \
  --output-dir federated_results \
  --num-clients 5 \
  --rounds 15 \
  --local-epochs 2 \
  --batch-size 512 \
  --learning-rate 0.01 \
  --hidden-dims 128,64 \
  --partition dirichlet \
  --dirichlet-alpha 0.8 \
  --use-class-weights
```

## Useful options

- `--max-rows 120000`: use a stratified sample if memory is tight
- `--target-mode derived_multiclass`: derive multiclass labels from the binary attack data
- `--partition iid`: switch to IID client partitioning
- `--client-fraction 0.6`: only sample some clients each round
- `--hidden-dims 256,128,64`: deeper network
- `--skip-plots`: skip `png` chart generation and save only the model plus CSV/JSON metrics

## Centralized baseline

```bash
python3 centralized_intrusion_detection.py \
  --csv "/Users/ayushimishra/Downloads/W-IIOTID_cleaned_and_encoded (1).csv" \
  --target-col Attack_label \
  --target-mode derived_multiclass \
  --output-dir centralized_results \
  --epochs 15 \
  --batch-size 512 \
  --learning-rate 0.01 \
  --hidden-dims 128,64 \
  --use-class-weights
```

## Live simulator

Train both models first, then run:

```bash
python3 live_ids_simulator.py \
  --csv "/Users/ayushimishra/Downloads/W-IIOTID_cleaned_and_encoded (1).csv" \
  --ml-model centralized_results/best_model.npz \
  --fl-model federated_results/best_model.npz \
  --events 25 \
  --interval 0.5 \
  --mode synthetic \
  --sampling-strategy balanced \
  --focus-class all \
  --noise-scale 0.05 \
  --output-log simulation_logs/live_predictions.csv
```

This will simulate a small stream of traffic and classify each event immediately with both the centralized ML model and the federated FL model across the derived multiclass labels.

## Dashboard

Train both models first, then run:

```bash
python3 run_ids_dashboard_one_click.py
```

Then open:

```text
http://127.0.0.1:8050
```

The dashboard lets you start and stop a synthetic or replay stream, watch each event arrive, compare centralized ML vs federated FL predictions, monitor running accuracy and agreement, and see a small live confusion trend chart.

## Deploy

For a full beginner-friendly deployment guide, see [DEPLOY_RENDER.md](/Users/ayushimishra/Documents/Codex/2026-04-26/files-mentioned-by-the-user-w/DEPLOY_RENDER.md).

## Outputs

The script writes the following into the output directory:

- `metrics.json`
- `history.csv`
- `classification_report.csv`
- `confusion_matrix.csv`
- `training_curves.png`
- `confusion_matrix.png`
- `best_model.npz`
