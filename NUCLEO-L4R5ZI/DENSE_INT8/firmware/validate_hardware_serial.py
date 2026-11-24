"""Valida il firmware ECG5000 INT8 sulla NUCLEO-L4R5ZI tramite protocollo ECG1.

Non modifica ne' il modello ne' il dataset: invia al microcontrollore un ECG alla
volta e confronta la sua uscita Float32 con il modello TFLite di riferimento.
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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import serial
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


FIRMWARE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = FIRMWARE_DIR.parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
DATASET_PATH = EXPERIMENT_DIR.parents[1] / "Dataset" / "ECG1000TEST.txt"
TFLITE_PATH = EXPERIMENT_DIR / "model" / "dense_int8.tflite"
EXPECTED_TFLITE_SHA256 = (EXPERIMENT_DIR / "model" / "tflite_sha256.txt").read_text(encoding="ascii").strip()

N_SAMPLES = 1000
N_COLUMNS = 141
N_SIGNAL_VALUES = 140
REQUEST_SIZE = 572
RESPONSE_SIZE = 40
LABELS = np.arange(1, 6, dtype=np.int32)


class ValidationError(RuntimeError):
    """Errore che rende non affidabile la validazione hardware."""


class Reporter:
    """Scrive simultaneamente messaggi leggibili a terminale e nel file di log."""

    def __init__(self, log_path: Path) -> None:
        self._log: BinaryIO = log_path.open("wb")

    def close(self) -> None:
        self._log.close()

    def info(self, message: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} | {message}"
        print(line, flush=True)
        self._log.write((line + "\n").encode("utf-8"))
        self._log.flush()


@dataclass
class HardwareResponse:
    sequence: int
    status: int
    predicted_class: int
    inference_microseconds: int
    probabilities: list[float]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fnv1a32(data: bytes) -> int:
    value = 0x811C9DC5
    for byte in data:
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def load_test_set(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise ValidationError(f"Dataset non trovato: {path}")
    try:
        values = np.loadtxt(path, dtype=np.float32)
    except ValueError as error:
        raise ValidationError(f"Dataset con valore non numerico o mancante: {error}") from error

    if values.shape != (N_SAMPLES, N_COLUMNS):
        raise ValidationError(
            f"Dimensione dataset non valida: {values.shape}; atteso ({N_SAMPLES}, {N_COLUMNS})."
        )
    if not np.all(np.isfinite(values)):
        raise ValidationError("Dataset contiene valori non finiti.")

    labels_float = values[:, 0]
    if not np.all(labels_float == np.rint(labels_float)):
        raise ValidationError("Il dataset contiene label non intere.")
    labels = labels_float.astype(np.int32)
    if not np.all(np.isin(labels, LABELS)):
        raise ValidationError("Il dataset contiene label fuori dall'intervallo 1-5.")
    if int(np.sum(labels == 5)) != 5:
        raise ValidationError("Il dataset non contiene esattamente 5 item della Classe 5.")

    signals = np.ascontiguousarray(values[:, 1:], dtype=np.float32)
    return labels, signals


def build_request(sequence: int, signal: np.ndarray) -> bytes:
    if signal.shape != (N_SIGNAL_VALUES,) or signal.dtype != np.float32:
        raise ValidationError("Segnale non coerente con 140 Float32.")
    body = struct.pack("<I", sequence) + signal.astype("<f4", copy=False).tobytes(order="C")
    frame = b"ECG1" + body + struct.pack("<I", fnv1a32(body))
    if len(frame) != REQUEST_SIZE:
        raise ValidationError(f"Frame richiesta non valido: {len(frame)} byte.")
    return frame


def read_exact(port: serial.Serial, size: int, timeout_seconds: float) -> bytes:
    deadline = time.monotonic() + timeout_seconds
    received = bytearray()
    while len(received) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValidationError(f"Timeout seriale: ricevuti {len(received)}/{size} byte.")
        chunk = port.read(size - len(received))
        if chunk:
            received.extend(chunk)
    return bytes(received)


def read_response(port: serial.Serial, timeout_seconds: float) -> HardwareResponse:
    """Trova RES2 anche se e' presente il messaggio testuale di boot."""
    deadline = time.monotonic() + timeout_seconds
    sliding = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValidationError("Timeout: firma RES2 non ricevuta dalla scheda.")
        byte = port.read(1)
        if not byte:
            continue
        sliding.extend(byte)
        if len(sliding) > 4:
            del sliding[0]
        if bytes(sliding) == b"RES2":
            tail = read_exact(port, RESPONSE_SIZE - 4, max(0.01, deadline - time.monotonic()))
            raw = b"RES2" + tail
            break

    body = raw[4:36]
    received_checksum = struct.unpack_from("<I", raw, 36)[0]
    if fnv1a32(body) != received_checksum:
        raise ValidationError("Checksum FNV-1a non valido nella risposta della scheda.")
    sequence, status, predicted_class, _reserved, inference_microseconds = struct.unpack_from("<IBBHI", body, 0)
    probabilities = np.frombuffer(body[12:32], dtype="<f4").astype(np.float32)
    if not np.all(np.isfinite(probabilities)):
        raise ValidationError("La scheda ha restituito probabilita' non finite.")
    if status == 0 and predicted_class not in LABELS:
        raise ValidationError(f"Classe restituita non valida: {predicted_class}.")
    return HardwareResponse(
        sequence=int(sequence),
        status=int(status),
        predicted_class=int(predicted_class),
        inference_microseconds=int(inference_microseconds),
        probabilities=[float(item) for item in probabilities],
    )


def run_tflite(interpreter: tf.lite.Interpreter, input_detail: dict, output_detail: dict, signal: np.ndarray) -> np.ndarray:
    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]
    if input_scale <= 0 or output_scale <= 0:
        raise ValidationError("Scale di quantizzazione TFLite non valide.")
    quantized_input = np.clip(
        np.rint(signal.reshape(1, N_SIGNAL_VALUES) / input_scale + input_zero_point), -128, 127
    ).astype(np.int8)
    interpreter.set_tensor(input_detail["index"], quantized_input)
    interpreter.invoke()
    result = interpreter.get_tensor(output_detail["index"])[0].astype(np.float32)
    return (result - output_zero_point) * output_scale


def create_run_directory(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_reports(
    run_dir: Path,
    board_name: str,
    rows: list[dict[str, object]],
    true_labels: np.ndarray,
    tflite_predictions: np.ndarray,
    hardware_predictions: np.ndarray,
    report: dict[str, object],
) -> None:
    prediction_fields = [
        "index_zero_based",
        "line_one_based",
        "true_class",
        "tflite_predicted_class",
        "hardware_predicted_class",
        "hardware_status",
        "classes_match_tflite",
        "correct_hardware",
        "max_abs_difference",
        "mean_abs_difference",
        "inference_time_us",
        "inference_time_ms",
        "round_trip_ms",
        "tflite_probability_class_1",
        "tflite_probability_class_2",
        "tflite_probability_class_3",
        "tflite_probability_class_4",
        "tflite_probability_class_5",
        "hardware_probability_class_1",
        "hardware_probability_class_2",
        "hardware_probability_class_3",
        "hardware_probability_class_4",
        "hardware_probability_class_5",
    ]
    with (run_dir / "hardware_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(rows)

    matrix = confusion_matrix(true_labels, hardware_predictions, labels=LABELS)
    with (run_dir / "hardware_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class\\predicted_class", *LABELS.tolist()])
        for label, matrix_row in zip(LABELS, matrix, strict=True):
            writer.writerow([int(label), *matrix_row.tolist()])

    precision, recall, f1_score, support = precision_recall_fscore_support(
        true_labels, hardware_predictions, labels=LABELS, zero_division=0
    )
    with (run_dir / "classification_report_hardware.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1_score", "support", "correct_predictions"])
        for label, p_value, r_value, f_value, support_value in zip(
            LABELS, precision, recall, f1_score, support, strict=True
        ):
            correct = int(np.sum((true_labels == label) & (hardware_predictions == label)))
            writer.writerow([int(label), float(p_value), float(r_value), float(f_value), int(support_value), correct])

    (run_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with (run_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"ECG5000 — validazione hardware seriale {board_name}\n")
        handle.write(f"Data e ora: {report['completed_at']}\n")
        handle.write(f"Item valutati: {report['items_evaluated']}\n")
        handle.write(f"Accuracy hardware: {report['hardware_accuracy']:.6f}\n")
        handle.write(f"Balanced accuracy hardware: {report['hardware_balanced_accuracy']:.6f}\n")
        handle.write(f"Classe hardware/TFLite discordanti: {report['prediction_disagreements']}\n")
        handle.write(f"Differenza assoluta massima: {report['absolute_difference_max']:.9e}\n")
        handle.write(f"Differenza assoluta media: {report['absolute_difference_mean']:.9e}\n")
        handle.write(f"Inferenza STM32: minimo {report['inference_time_us_min']} us, ")
        handle.write(f"media {report['inference_time_us_mean']:.3f} us, ")
        handle.write(f"massimo {report['inference_time_us_max']} us\n")
        handle.write(f"Tempo medio round-trip: {report['round_trip_ms_mean']:.3f} ms\n")
        handle.write("Nota: il round-trip include USB/seriale, protocollo e inferenza; non e' la sola latenza AI.\n")


def validate(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    run_dir = create_run_directory(output_root)
    reporter = Reporter(run_dir / "validation.log")
    try:
        reporter.info(f"ECG5000 — validazione hardware {args.board_name}: avvio.")
        reporter.info("[1/6] Verifica modello TFLite e dataset di test, senza modificare alcun file.")
        tflite_hash = sha256(TFLITE_PATH)
        if tflite_hash != EXPECTED_TFLITE_SHA256:
            raise ValidationError(
                f"Checksum TFLite inatteso: {tflite_hash}; atteso {EXPECTED_TFLITE_SHA256}."
            )
        labels, signals = load_test_set(DATASET_PATH)
        if args.limit < 1 or args.limit > len(labels):
            raise ValidationError(f"Limit non valido: {args.limit}; intervallo ammesso 1-{len(labels)}.")
        labels = labels[:args.limit]
        signals = signals[:args.limit]
        distribution = {str(label): int(np.sum(labels == label)) for label in LABELS}
        reporter.info(
            f"  OK: {len(labels)} righe, 141 valori ciascuna, classi {distribution}; SHA-256 TFLite verificato."
        )

        reporter.info("[2/6] Caricamento del riferimento TFLite full-INT8.")
        interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        if tuple(input_detail["shape"].tolist()) != (1, 140) or input_detail["dtype"] != np.int8:
            raise ValidationError(f"Input TFLite inatteso: {input_detail['shape']} {input_detail['dtype']}.")
        if tuple(output_detail["shape"].tolist()) != (1, 5) or output_detail["dtype"] != np.int8:
            raise ValidationError(f"Output TFLite inatteso: {output_detail['shape']} {output_detail['dtype']}.")
        reporter.info(
            f"  OK: input [1, 140] INT8 scale={input_detail['quantization']}; "
            f"output [1, 5] INT8 scale={output_detail['quantization']}."
        )

        reporter.info(f"[3/6] Apertura collegamento seriale {args.port} a {args.baudrate} baud.")
        rows: list[dict[str, object]] = []
        tflite_classes: list[int] = []
        hardware_classes: list[int] = []
        absolute_differences: list[np.ndarray] = []
        round_trip_times: list[float] = []
        inference_times_us: list[int] = []
        retry_count = 0
        with serial.Serial(
            port=args.port,
            baudrate=args.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
            write_timeout=args.timeout,
        ) as port:
            time.sleep(0.15)
            port.reset_input_buffer()
            port.reset_output_buffer()
            reporter.info("  OK: porta aperta. Il firmware ricevera' un ECG (140 Float32) per volta.")
            reporter.info("[4/6] Invio, inferenza e confronto: avanzamento ogni 50 ECG (e primi tre).")
            for index, (true_label, signal) in enumerate(zip(labels, signals, strict=True)):
                tflite_probabilities = run_tflite(interpreter, input_detail, output_detail, signal)
                tflite_prediction = int(np.argmax(tflite_probabilities)) + 1
                request = build_request(index, signal)
                last_error: Exception | None = None
                for attempt in range(args.retries + 1):
                    started = time.perf_counter()
                    try:
                        port.write(request)
                        port.flush()
                        response = read_response(port, args.timeout)
                        round_trip_ms = (time.perf_counter() - started) * 1000.0
                        if response.sequence != index:
                            raise ValidationError(
                                f"Sequenza inattesa: ricevuta {response.sequence}, attesa {index}."
                            )
                        if response.status != 0:
                            raise ValidationError(f"Firmware ha restituito stato di errore {response.status}.")
                        break
                    except (serial.SerialException, ValidationError) as error:
                        last_error = error
                        if attempt == args.retries:
                            raise ValidationError(
                                f"ECG {index}: errore dopo {args.retries + 1} tentativi: {error}"
                            ) from error
                        retry_count += 1
                        reporter.info(
                            f"  ECG {index + 1}: tentativo {attempt + 1} non riuscito ({error}); riprovo."
                        )
                        port.reset_input_buffer()
                else:  # pragma: no cover - protegge una condizione teoricamente impossibile.
                    raise ValidationError(f"ECG {index}: nessuna risposta ({last_error}).")

                hardware_probabilities = np.asarray(response.probabilities, dtype=np.float32)
                hardware_prediction = int(response.predicted_class)
                differences = np.abs(tflite_probabilities - hardware_probabilities)
                classes_match = hardware_prediction == tflite_prediction
                is_correct = hardware_prediction == int(true_label)
                tflite_classes.append(tflite_prediction)
                hardware_classes.append(hardware_prediction)
                absolute_differences.append(differences)
                round_trip_times.append(round_trip_ms)
                inference_times_us.append(response.inference_microseconds)
                record: dict[str, object] = {
                    "index_zero_based": index,
                    "line_one_based": index + 1,
                    "true_class": int(true_label),
                    "tflite_predicted_class": tflite_prediction,
                    "hardware_predicted_class": hardware_prediction,
                    "hardware_status": response.status,
                    "classes_match_tflite": classes_match,
                    "correct_hardware": is_correct,
                    "max_abs_difference": float(np.max(differences)),
                    "mean_abs_difference": float(np.mean(differences)),
                    "inference_time_us": response.inference_microseconds,
                    "inference_time_ms": response.inference_microseconds / 1000.0,
                    "round_trip_ms": float(round_trip_ms),
                }
                for class_index in range(5):
                    record[f"tflite_probability_class_{class_index + 1}"] = float(tflite_probabilities[class_index])
                    record[f"hardware_probability_class_{class_index + 1}"] = float(hardware_probabilities[class_index])
                rows.append(record)

                if args.verbose or index < 3 or (index + 1) % 50 == 0 or index + 1 == len(labels):
                    reporter.info(
                        f"  ECG {index + 1:4d}/{len(labels)} | reale={int(true_label)} | "
                        f"TFLite={tflite_prediction} | STM32={hardware_prediction} | "
                        f"AI={response.inference_microseconds} us | delta_max={np.max(differences):.3e} | "
                        f"round-trip={round_trip_ms:.2f} ms"
                    )

        reporter.info("[5/6] Calcolo metriche hardware e controllo coerenza con TFLite.")
        tflite_array = np.asarray(tflite_classes, dtype=np.int32)
        hardware_array = np.asarray(hardware_classes, dtype=np.int32)
        all_differences = np.concatenate(absolute_differences)
        matrix = confusion_matrix(labels, hardware_array, labels=LABELS)
        precision, recall, f1_score, support = precision_recall_fscore_support(
            labels, hardware_array, labels=LABELS, zero_division=0
        )
        per_class = {
            str(label): {
                "precision": float(p_value),
                "recall": float(r_value),
                "f1_score": float(f_value),
                "support": int(support_value),
                "correct_predictions": int(np.sum((labels == label) & (hardware_array == label))),
            }
            for label, p_value, r_value, f_value, support_value in zip(
                LABELS, precision, recall, f1_score, support, strict=True
            )
        }
        disagreements = np.flatnonzero(tflite_array != hardware_array).astype(int).tolist()
        class_5_items = [
            row
            for row in rows
            if int(row["true_class"]) == 5
        ]
        report: dict[str, object] = {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": str(DATASET_PATH),
            "tflite_model": str(TFLITE_PATH),
            "tflite_sha256": tflite_hash,
            "serial_port": args.port,
            "baudrate": args.baudrate,
            "board_name": args.board_name,
            "items_evaluated": len(rows),
            "dataset_class_distribution": distribution,
            "hardware_accuracy": float(accuracy_score(labels, hardware_array)),
            "hardware_balanced_accuracy": float(balanced_accuracy_score(labels, hardware_array)),
            "tflite_accuracy": float(accuracy_score(labels, tflite_array)),
            "tflite_balanced_accuracy": float(balanced_accuracy_score(labels, tflite_array)),
            "prediction_disagreements": len(disagreements),
            "disagreement_indices_zero_based": disagreements,
            "absolute_difference_max": float(np.max(all_differences)),
            "absolute_difference_mean": float(np.mean(all_differences)),
            "inference_time_us_min": int(np.min(inference_times_us)),
            "inference_time_us_mean": float(np.mean(inference_times_us)),
            "inference_time_us_max": int(np.max(inference_times_us)),
            "round_trip_ms_min": float(np.min(round_trip_times)),
            "round_trip_ms_mean": float(np.mean(round_trip_times)),
            "round_trip_ms_max": float(np.max(round_trip_times)),
            "retries_used": retry_count,
            "hardware_confusion_matrix": matrix.tolist(),
            "hardware_per_class": per_class,
            "class_5_items": class_5_items,
            "notes": [
                "Il tempo round-trip include USB/seriale, protocollo e inferenza; non e' la sola latenza AI.",
                "Nessun file di dataset, modello Keras o modello TFLite viene modificato.",
            ],
        }
        save_reports(run_dir, args.board_name, rows, labels, tflite_array, hardware_array, report)
        reporter.info(
            f"  Accuracy STM32={report['hardware_accuracy']:.4%}; "
            f"TFLite={report['tflite_accuracy']:.4%}; discordanti={len(disagreements)}."
        )
        reporter.info(
            f"  Differenza assoluta: max={report['absolute_difference_max']:.3e}, "
            f"media={report['absolute_difference_mean']:.3e}; retry={retry_count}."
        )
        reporter.info(
            f"  Inferenza STM32: min={report['inference_time_us_min']} us, "
            f"media={report['inference_time_us_mean']:.3f} us, "
            f"max={report['inference_time_us_max']} us."
        )
        reporter.info(
            f"  Round-trip PC-scheda: min={report['round_trip_ms_min']:.3f} ms, "
            f"medio={report['round_trip_ms_mean']:.3f} ms, "
            f"max={report['round_trip_ms_max']:.3f} ms."
        )
        reporter.info(
            f"  Classe 5: {per_class['5']['correct_predictions']}/{per_class['5']['support']} corretti."
        )
        reporter.info("[6/6] File di report salvati. Validazione completata correttamente.")
        reporter.info(f"  Cartella risultati: {run_dir}")
        return run_dir
    finally:
        reporter.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida 1.000 ECG sul firmware INT8 NUCLEO-L4R5ZI via seriale.")
    parser.add_argument("--port", required=True, help="Porta seriale della ST-LINK Virtual COM.")
    parser.add_argument("--board-name", default="NUCLEO-L4R5ZI", help="Identificativo della scheda nei report.")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baud rate seriale (default: 115200).")
    parser.add_argument("--timeout", type=float, default=3.0, help="Timeout per risposta in secondi (default: 3).")
    parser.add_argument("--retries", type=int, default=2, help="Tentativi aggiuntivi per ciascun ECG (default: 2).")
    parser.add_argument("--limit", type=int, default=N_SAMPLES, help="Numero di ECG da elaborare (default: 1000).")
    parser.add_argument("--verbose", action="store_true", help="Mostra una riga di avanzamento per ogni ECG.")
    parser.add_argument(
        "--output-root",
        default=str(EXPERIMENT_DIR / "results" / "hardware_validation"),
        help="Cartella che conterra' una nuova sottocartella per ogni esecuzione.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.retries < 0 or args.timeout <= 0:
        print("ERRORE: timeout deve essere positivo e retries non negativo.", file=sys.stderr)
        return 2
    try:
        validate(args)
    except Exception as error:  # Il launcher deve ricevere un exit code non nullo.
        print(f"ERRORE BLOCCANTE: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
