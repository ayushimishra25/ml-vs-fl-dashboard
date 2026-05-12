#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import threading
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

import pandas as pd

from ids_labeling import humanize_label, prepare_dataframe_for_model
from live_ids_simulator import event_stream, load_model, predict_label


class DashboardState:
    def __init__(self, csv_path: Path, ml_model_path: Path, fl_model_path: Path) -> None:
        self.csv_path = csv_path
        self.ml_model_path = ml_model_path
        self.fl_model_path = fl_model_path

        source_df = pd.read_csv(csv_path)
        self.ml_model = load_model(ml_model_path, "ML")
        self.fl_model = load_model(fl_model_path, "FL")
        if self.ml_model.target_col != self.fl_model.target_col:
            raise ValueError("ML and FL models use different target columns.")
        if self.ml_model.target_mode != self.fl_model.target_mode:
            raise ValueError("ML and FL models use different target modes.")

        self.df, self.target_col = prepare_dataframe_for_model(
            df=source_df,
            target_mode=self.ml_model.target_mode,
            target_col=self.ml_model.target_col,
            source_target_col=self.ml_model.source_target_col,
            drop_columns=self.ml_model.drop_columns,
        )
        self.class_names = [str(label) for label in self.ml_model.label_names]
        self.class_display_names = {
            label: humanize_label(label) for label in self.class_names
        }

        preview_candidates = [
            "Protocol",
            "Service",
            "Duration",
            "Scr_bytes",
            "Des_bytes",
            "total_bytes",
            "paket_rate",
            "byte_rate",
        ]
        self.preview_fields = [field for field in preview_candidates if field in self.df.columns]
        if not self.preview_fields:
            self.preview_fields = [
                column for column in self.df.columns if column != self.target_col
            ][:8]

        self.lock = threading.Lock()
        self.stop_signal = threading.Event()
        self.worker: threading.Thread | None = None
        self.max_events = 300
        self.max_trend_points = 160
        self.events: deque[Dict[str, Any]] = deque(maxlen=self.max_events)
        self.trend_points: deque[Dict[str, float | int]] = deque(maxlen=self.max_trend_points)
        self.current_event: Dict[str, Any] | None = None
        self.running = False
        self.run_id = 0
        self.stats = self._fresh_stats()
        self.last_config = self.default_config()

    def default_config(self) -> Dict[str, Any]:
        return {
            "events": 60,
            "interval": 0.6,
            "mode": "synthetic",
            "sampling_strategy": "balanced",
            "focus_class": "all",
            "noise_scale": 0.05,
            "seed": 42,
        }

    def _fresh_stats(self) -> Dict[str, Any]:
        return {
            "total_events": 0,
            "ml_correct": 0,
            "fl_correct": 0,
            "agreement_count": 0,
            "true_counts": {label: 0 for label in self.class_names},
            "ml_counts": {label: 0 for label in self.class_names},
            "fl_counts": {label: 0 for label in self.class_names},
        }

    def start(self, config: Dict[str, Any]) -> None:
        with self.lock:
            if self.running:
                raise RuntimeError("A simulation is already running.")
            self.events.clear()
            self.trend_points.clear()
            self.current_event = None
            self.stats = self._fresh_stats()
            self.run_id += 1
            self.running = True
            self.last_config = config
            self.stop_signal.clear()
            run_id = self.run_id

        self.worker = threading.Thread(
            target=self._run_stream,
            args=(config, run_id),
            daemon=True,
        )
        self.worker.start()

    def stop(self) -> None:
        self.stop_signal.set()
        with self.lock:
            self.running = False

    def _run_stream(self, config: Dict[str, Any], run_id: int) -> None:
        stream = event_stream(
            df=self.df,
            mode=config["mode"],
            target_col=self.target_col,
            events=config["events"],
            sampling_strategy=config["sampling_strategy"],
            focus_class=config["focus_class"],
            noise_scale=config["noise_scale"],
            seed=config["seed"],
        )

        for index, event in enumerate(stream, start=1):
            if self.stop_signal.is_set():
                break

            event_frame = pd.DataFrame([event])
            true_label = str(event[self.target_col]) if self.target_col in event else "unknown"
            ml_label, ml_confidence = predict_label(event_frame, self.ml_model)
            fl_label, fl_confidence = predict_label(event_frame, self.fl_model)

            payload = {
                "event_id": index,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "true_label": true_label,
                "true_label_name": self.class_display_names.get(true_label, humanize_label(true_label)),
                "ml_label": ml_label,
                "ml_label_name": self.class_display_names.get(ml_label, humanize_label(ml_label)),
                "ml_confidence": round(ml_confidence, 4),
                "fl_label": fl_label,
                "fl_label_name": self.class_display_names.get(fl_label, humanize_label(fl_label)),
                "fl_confidence": round(fl_confidence, 4),
                "agree": ml_label == fl_label,
                "preview": {
                    field: self._clean_value(event[field]) for field in self.preview_fields
                },
            }

            with self.lock:
                if run_id != self.run_id:
                    return
                self.events.append(payload)
                self.current_event = payload
                self._update_stats(payload)

            if config["interval"] > 0 and self.stop_signal.wait(config["interval"]):
                break

        with self.lock:
            if run_id == self.run_id:
                self.running = False

    def _update_stats(self, payload: Dict[str, Any]) -> None:
        stats = self.stats
        stats["total_events"] += 1
        total = stats["total_events"]

        true_label = payload["true_label"]
        ml_label = payload["ml_label"]
        fl_label = payload["fl_label"]

        if true_label in stats["true_counts"]:
            stats["true_counts"][true_label] += 1
        if ml_label in stats["ml_counts"]:
            stats["ml_counts"][ml_label] += 1
        if fl_label in stats["fl_counts"]:
            stats["fl_counts"][fl_label] += 1

        if ml_label == true_label:
            stats["ml_correct"] += 1
        if fl_label == true_label:
            stats["fl_correct"] += 1
        if payload["agree"]:
            stats["agreement_count"] += 1

        self.trend_points.append(
            {
                "event_id": payload["event_id"],
                "ml_confusion_rate": round(1.0 - (stats["ml_correct"] / total), 4),
                "fl_confusion_rate": round(1.0 - (stats["fl_correct"] / total), 4),
            }
        )

    def snapshot(self, after_event_id: int = 0) -> Dict[str, Any]:
        with self.lock:
            new_events = [
                event for event in list(self.events) if event["event_id"] > after_event_id
            ]
            total = max(1, self.stats["total_events"])
            classes_seen = sum(1 for count in self.stats["true_counts"].values() if count > 0)
            return {
                "ready": True,
                "running": self.running,
                "config": self.last_config,
                "class_names": self.class_names,
                "class_display_names": self.class_display_names,
                "events": new_events,
                "current_event": self.current_event,
                "preview_fields": self.preview_fields,
                "trend_points": list(self.trend_points),
                "stats": {
                    **self.stats,
                    "ml_accuracy": round(self.stats["ml_correct"] / total, 4),
                    "fl_accuracy": round(self.stats["fl_correct"] / total, 4),
                    "agreement_rate": round(self.stats["agreement_count"] / total, 4),
                    "classes_seen": classes_seen,
                    "class_count": len(self.class_names),
                },
            }

    @staticmethod
    def _clean_value(value: Any) -> Any:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float):
            return round(value, 4)
        return value


class DashboardHandler(BaseHTTPRequestHandler):
    state: DashboardState
    static_dir: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_file("index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/app.js":
            self._serve_file("app.js", "application/javascript; charset=utf-8")
            return
        if parsed.path == "/styles.css":
            self._serve_file("styles.css", "text/css; charset=utf-8")
            return
        if parsed.path == "/api/state":
            query = parse_qs(parsed.query)
            after = int(query.get("after", ["0"])[0])
            self._send_json(self.state.snapshot(after_event_id=after))
            return
        if parsed.path == "/healthz":
            self._send_json({"ok": True, "running": self.state.running})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/start":
            try:
                payload = self._read_json()
                config = self.state.default_config()
                config.update(
                    {
                        "events": int(payload.get("events", config["events"])),
                        "interval": float(payload.get("interval", config["interval"])),
                        "mode": str(payload.get("mode", config["mode"])),
                        "sampling_strategy": str(
                            payload.get("sampling_strategy", config["sampling_strategy"])
                        ),
                        "focus_class": str(payload.get("focus_class", config["focus_class"])),
                        "noise_scale": float(payload.get("noise_scale", config["noise_scale"])),
                        "seed": int(payload.get("seed", config["seed"])),
                    }
                )
                self.state.start(config)
                self._send_json({"ok": True, "config": config})
            except Exception as error:
                self._send_json({"ok": False, "error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            return

        if parsed.path == "/api/stop":
            self.state.stop()
            self._send_json({"ok": True})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _serve_file(self, filename: str, content_type: str) -> None:
        file_path = self.static_dir / filename
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Missing static asset")
            return
        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _read_json(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length else b"{}"
        return json.loads(raw.decode("utf-8"))


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parent
    default_csv = os.environ.get(
        "CSV_PATH",
        str(workspace / "deployment_assets" / "stream_source.csv"),
    )
    default_ml_model = os.environ.get(
        "ML_MODEL_PATH",
        str(workspace / "deployment_assets" / "models" / "centralized_best_model.npz"),
    )
    default_fl_model = os.environ.get(
        "FL_MODEL_PATH",
        str(workspace / "deployment_assets" / "models" / "federated_best_model.npz"),
    )
    default_host = os.environ.get("HOST", "0.0.0.0")
    default_port = int(os.environ.get("PORT", "8050"))

    parser = argparse.ArgumentParser(
        description="Run the live IDS dashboard server."
    )
    parser.add_argument("--csv", default=default_csv, help="Path to the source CSV file.")
    parser.add_argument("--ml-model", default=default_ml_model, help="Path to the centralized model.")
    parser.add_argument("--fl-model", default=default_fl_model, help="Path to the federated model.")
    parser.add_argument("--host", default=default_host, help="Host to bind the server to.")
    parser.add_argument("--port", type=int, default=default_port, help="Port to bind the server to.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(__file__).resolve().parent
    static_dir = workspace / "dashboard"

    DashboardHandler.state = DashboardState(
        csv_path=Path(args.csv).expanduser().resolve(),
        ml_model_path=Path(args.ml_model).expanduser().resolve(),
        fl_model_path=Path(args.fl_model).expanduser().resolve(),
    )
    DashboardHandler.static_dir = static_dir

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        DashboardHandler.state.stop()
        server.server_close()


if __name__ == "__main__":
    main()
