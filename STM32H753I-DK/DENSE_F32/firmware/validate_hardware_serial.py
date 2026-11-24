"""Validazione seriale completa ECG5000 per STM32H573I-DK.

Il dataset e il modello sono esclusivamente letti; ogni esecuzione crea una
nuova cartella di risultati sotto results/hardware_validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import serial
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = EXPERIMENT_DIR.parents[1] / "Dataset" / "ECG1000TEST.txt"
MODEL_PATH = EXPERIMENT_DIR / "model" / "dense_f32_baseline_r001_float32.tflite"
EXPECTED_SHA256 = "652e44d3b7d2dd7d8e969e48858cdc27fcbf416502a145e3889c731d54db67f0"
LABELS = np.arange(1, 6, dtype=np.int32)
REQUEST_SIZE, RESPONSE_SIZE = 572, 40

class ValidationError(RuntimeError):
    pass

def fnv1a32(data: bytes) -> int:
    value = 0x811C9DC5
    for byte in data:
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    if not DATASET_PATH.is_file():
        raise ValidationError(f"Dataset non trovato: {DATASET_PATH}")
    values = np.loadtxt(DATASET_PATH, dtype=np.float32)
    if values.shape != (1000, 141) or not np.all(np.isfinite(values)):
        raise ValidationError(f"Dataset non valido: shape={values.shape}, valori finiti={np.all(np.isfinite(values))}")
    labels = values[:, 0].astype(np.int32)
    if not np.all(values[:, 0] == labels) or not np.all(np.isin(labels, LABELS)) or int(np.sum(labels == 5)) != 5:
        raise ValidationError("Label del dataset non coerenti con ECG5000.")
    return labels, np.ascontiguousarray(values[:, 1:], dtype=np.float32)

def build_request(sequence: int, signal: np.ndarray) -> bytes:
    if signal.shape != (140,) or signal.dtype != np.float32:
        raise ValidationError("Segnale non costituito da 140 Float32.")
    body = struct.pack("<I", sequence) + signal.astype("<f4", copy=False).tobytes(order="C")
    frame = b"ECG1" + body + struct.pack("<I", fnv1a32(body))
    if len(frame) != REQUEST_SIZE:
        raise ValidationError(f"Frame ECG1 lungo {len(frame)}, atteso {REQUEST_SIZE}.")
    return frame

def read_exact(port: serial.Serial, count: int, deadline: float) -> bytes:
    data = bytearray()
    while len(data) < count:
        if time.monotonic() >= deadline:
            raise ValidationError(f"Timeout seriale: {len(data)}/{count} byte ricevuti.")
        chunk = port.read(count - len(data))
        if chunk:
            data.extend(chunk)
    return bytes(data)

def read_response(port: serial.Serial, timeout: float) -> tuple[int, int, int, int, np.ndarray]:
    deadline = time.monotonic() + timeout
    window = bytearray()
    while time.monotonic() < deadline:
        byte = port.read(1)
        if not byte:
            continue
        window.extend(byte)
        if len(window) > 4:
            del window[0]
        if bytes(window) == b"RES2":
            raw = b"RES2" + read_exact(port, 36, deadline)
            body = raw[4:36]
            if fnv1a32(body) != struct.unpack_from("<I", raw, 36)[0]:
                raise ValidationError("Checksum FNV-1a RES2 non valido.")
            sequence, status, predicted, _reserved, elapsed_us = struct.unpack_from("<IBBHI", body, 0)
            probabilities = np.frombuffer(body[12:32], dtype="<f4").astype(np.float32)
            if not np.all(np.isfinite(probabilities)):
                raise ValidationError("Probabilita hardware non finite.")
            return sequence, status, predicted, elapsed_us, probabilities
    raise ValidationError("Firma RES2 non ricevuta.")

def new_run_dir(root: Path) -> Path:
    candidate = root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate

def write_reports(run_dir: Path, rows: list[dict[str, object]], labels: np.ndarray, tflite: np.ndarray, hardware: np.ndarray, report: dict[str, object]) -> None:
    fields = list(rows[0])
    with (run_dir / "hardware_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    matrix = confusion_matrix(labels, hardware, labels=LABELS)
    with (run_dir / "hardware_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["true_class\\predicted_class", *LABELS.tolist()])
        for label, row in zip(LABELS, matrix, strict=True): writer.writerow([int(label), *row.tolist()])
    p, r, f, support = precision_recall_fscore_support(labels, hardware, labels=LABELS, zero_division=0)
    with (run_dir / "classification_report_hardware.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["class", "precision", "recall", "f1_score", "support", "correct_predictions"])
        for label, pv, rv, fv, sv in zip(LABELS, p, r, f, support, strict=True):
            writer.writerow([int(label), float(pv), float(rv), float(fv), int(sv), int(np.sum((labels == label) & (hardware == label)))])
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "summary.md").write_text(
        "# Validazione hardware ECG5000\n\n"
        f"- Board: {report['board_name']}\n- ECG valutati: {report['items_evaluated']}\n"
        f"- Accuracy hardware: {report['hardware_accuracy']:.4%}\n- Balanced accuracy: {report['hardware_balanced_accuracy']:.4%}\n"
        f"- Discordanze hardware/TFLite: {report['prediction_disagreements']}\n"
        f"- Delta probabilita massimo: {report['absolute_difference_max']:.9e}\n"
        f"- DWT inference: {report['inference_time_us_mean']:.3f} us medio ({report['inference_time_us_min']}-{report['inference_time_us_max']} us)\n"
        f"- Round-trip medio: {report['round_trip_ms_mean']:.3f} ms\n\n"
        "Il round-trip include USB, seriale e protocollo. Il DWT misura la sola inferenza firmware.\n",
        encoding="utf-8")

def validate(args: argparse.Namespace) -> Path:
    run_dir = new_run_dir(Path(args.output_root))
    log = (run_dir / "validation.log").open("w", encoding="utf-8", buffering=1)
    def info(message: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} | {message}"; print(line, flush=True); print(line, file=log)
    try:
        info("[1/6] Verifica modello e dataset in sola lettura.")
        if sha256(MODEL_PATH) != EXPECTED_SHA256: raise ValidationError("Checksum TFLite inatteso.")
        labels, signals = load_dataset()
        if not 1 <= args.limit <= len(labels): raise ValidationError("--limit deve essere tra 1 e 1000.")
        labels, signals = labels[:args.limit], signals[:args.limit]
        info(f"  OK: modello SHA-256 e {len(labels)} ECG verificati.")
        info("[2/6] Caricamento riferimento TFLite Float32.")
        interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH)); interpreter.allocate_tensors()
        input_info, output_info = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        if tuple(input_info['shape']) != (1, 140) or input_info['dtype'] != np.float32 or tuple(output_info['shape']) != (1, 5) or output_info['dtype'] != np.float32: raise ValidationError("Tensor TFLite inattesi.")
        info(f"[3/6] Apertura {args.port} a {args.baudrate} baud.")
        rows, refs, hws, deltas, dwt, rtt = [], [], [], [], [], []
        retries_used = 0
        with serial.Serial(args.port, args.baudrate, timeout=0.1, write_timeout=args.timeout) as port:
            time.sleep(0.15); port.reset_input_buffer(); port.reset_output_buffer()
            info("[4/6] Invio, inferenza e confronto dei segnali.")
            for index, (label, signal) in enumerate(zip(labels, signals, strict=True)):
                interpreter.set_tensor(input_info['index'], signal.reshape(1, 140)); interpreter.invoke()
                ref_probs = np.asarray(interpreter.get_tensor(output_info['index'])[0], dtype=np.float32)
                ref_class = int(np.argmax(ref_probs)) + 1
                error: Exception | None = None
                for attempt in range(args.retries + 1):
                    started = time.perf_counter()
                    try:
                        port.write(build_request(index, signal)); port.flush()
                        sequence, status, hw_class, elapsed, hw_probs = read_response(port, args.timeout)
                        elapsed_rtt = (time.perf_counter() - started) * 1000
                        if sequence != index or status != 0 or hw_class not in LABELS: raise ValidationError(f"Risposta non valida: sequence={sequence}, status={status}, class={hw_class}.")
                        break
                    except (serial.SerialException, ValidationError) as caught:
                        error = caught
                        if attempt == args.retries: raise ValidationError(f"ECG {index}: fallito dopo {attempt + 1} tentativi: {caught}") from caught
                        retries_used += 1; port.reset_input_buffer()
                diff = np.abs(ref_probs - hw_probs); refs.append(ref_class); hws.append(hw_class); deltas.append(diff); dwt.append(elapsed); rtt.append(elapsed_rtt)
                row: dict[str, object] = {"index_zero_based": index, "line_one_based": index + 1, "true_class": int(label), "tflite_predicted_class": ref_class, "hardware_predicted_class": hw_class, "hardware_status": status, "classes_match_tflite": hw_class == ref_class, "correct_hardware": hw_class == int(label), "max_abs_difference": float(diff.max()), "mean_abs_difference": float(diff.mean()), "inference_time_us": elapsed, "inference_time_ms": elapsed / 1000, "round_trip_ms": elapsed_rtt}
                for class_index in range(5): row[f"tflite_probability_class_{class_index + 1}"], row[f"hardware_probability_class_{class_index + 1}"] = float(ref_probs[class_index]), float(hw_probs[class_index])
                rows.append(row)
                if index < 3 or (index + 1) % 50 == 0 or index + 1 == len(labels): info(f"  ECG {index + 1:4d}/{len(labels)} | TFLite={ref_class} | STM32={hw_class} | AI={elapsed} us | delta={diff.max():.3e} | RT={elapsed_rtt:.2f} ms")
        info("[5/6] Calcolo metriche e salvataggio report.")
        refs_a, hws_a, all_deltas = np.asarray(refs), np.asarray(hws), np.concatenate(deltas)
        report: dict[str, object] = {"completed_at": datetime.now().isoformat(timespec="seconds"), "board_name": args.board_name, "serial_port": args.port, "baudrate": args.baudrate, "dataset": str(DATASET_PATH), "tflite_model": str(MODEL_PATH), "tflite_sha256": EXPECTED_SHA256, "items_evaluated": len(rows), "dataset_class_distribution": {str(k): int(np.sum(labels == k)) for k in LABELS}, "hardware_accuracy": float(accuracy_score(labels, hws_a)), "hardware_balanced_accuracy": float(balanced_accuracy_score(labels, hws_a)), "tflite_accuracy": float(accuracy_score(labels, refs_a)), "tflite_balanced_accuracy": float(balanced_accuracy_score(labels, refs_a)), "prediction_disagreements": int(np.sum(refs_a != hws_a)), "disagreement_indices_zero_based": np.flatnonzero(refs_a != hws_a).astype(int).tolist(), "absolute_difference_max": float(all_deltas.max()), "absolute_difference_mean": float(all_deltas.mean()), "inference_time_us_min": int(np.min(dwt)), "inference_time_us_mean": float(np.mean(dwt)), "inference_time_us_max": int(np.max(dwt)), "round_trip_ms_min": float(np.min(rtt)), "round_trip_ms_mean": float(np.mean(rtt)), "round_trip_ms_max": float(np.max(rtt)), "retries_used": retries_used, "hardware_confusion_matrix": confusion_matrix(labels, hws_a, labels=LABELS).tolist(), "notes": ["DWT misura la sola inferenza; il round-trip include USB, seriale e protocollo.", "Dataset e modello sono letti senza modifiche."]}
        write_reports(run_dir, rows, labels, refs_a, hws_a, report)
        info(f"[6/6] COMPLETATA: accuracy={report['hardware_accuracy']:.4%}; discordanti={report['prediction_disagreements']}; risultati={run_dir}")
        return run_dir
    finally:
        log.close()

def main() -> int:
    parser = argparse.ArgumentParser(description="Valida ECG5000 su STM32H573I-DK via ECG1/RES2.")
    parser.add_argument("--port", required=True); parser.add_argument("--board-name", default="STM32H573I-DK"); parser.add_argument("--baudrate", type=int, default=115200); parser.add_argument("--timeout", type=float, default=3.0); parser.add_argument("--retries", type=int, default=2); parser.add_argument("--limit", type=int, default=1000); parser.add_argument("--output-root", default=str(EXPERIMENT_DIR / "results" / "hardware_validation")); args = parser.parse_args()
    if args.timeout <= 0 or args.retries < 0: parser.error("timeout deve essere positivo e retries non negativo")
    try: validate(args)
    except Exception as error: print(f"ERRORE BLOCCANTE: {error}", file=sys.stderr, flush=True); return 1
    return 0

if __name__ == "__main__": raise SystemExit(main())
