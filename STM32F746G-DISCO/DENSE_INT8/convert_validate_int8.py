"""Converte la baseline Keras in TFLite INT8 e ne valida l'accuratezza.

Usa il TRAIN soltanto come representative dataset per calibrare le scale; nessun
training, modifica del modello Keras o modifica dei file ECG viene effettuata.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT.parents[1] / "Dataset"
KERAS_MODEL = ROOT / "model" / "dense_f32_baseline_r001.keras"
TFLITE_MODEL = ROOT / "model" / "dense_int8.tflite"
SHA_FILE = ROOT / "model" / "tflite_sha256.txt"
REPORT_DIR = ROOT / "reports" / "tflite_int8_validation"
LABELS = np.arange(1, 6, dtype=np.int32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: Path, expected_rows: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, dtype=np.float32)
    if values.shape != (expected_rows, 141):
        raise RuntimeError(f"Dataset {path.name}: forma {values.shape}; attesa ({expected_rows}, 141).")
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"Dataset {path.name}: valori mancanti o non finiti.")
    labels = values[:, 0].astype(np.int32)
    if not np.all(values[:, 0] == labels) or not np.all(np.isin(labels, LABELS)):
        raise RuntimeError(f"Dataset {path.name}: label non valide.")
    return labels, np.ascontiguousarray(values[:, 1:], dtype=np.float32)


def quantize(values: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    if scale <= 0:
        raise RuntimeError("Scala di quantizzazione TFLite non valida.")
    info = np.iinfo(detail["dtype"])
    return np.clip(np.rint(values / scale + zero_point), info.min, info.max).astype(detail["dtype"])


def dequantize(values: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    return (values.astype(np.float32) - zero_point) * scale


def main() -> None:
    tf.keras.utils.set_random_seed(123)
    train_labels, train_signals = load_dataset(DATASET_DIR / "ECG3200TRAIN.txt", 3200)
    test_labels, test_signals = load_dataset(DATASET_DIR / "ECG1000TEST.txt", 1000)
    if int(np.sum(test_labels == 5)) != 5:
        raise RuntimeError("Il test set non contiene esattamente 5 item della Classe 5.")
    if not KERAS_MODEL.is_file():
        raise RuntimeError(f"Modello Keras non trovato: {KERAS_MODEL}")

    model = tf.keras.models.load_model(KERAS_MODEL, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for item in train_signals:
            yield [item.reshape(1, 140).astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    TFLITE_MODEL.write_bytes(converter.convert())
    model_hash = sha256(TFLITE_MODEL)
    SHA_FILE.write_text(model_hash + "\n", encoding="ascii")

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if tuple(input_detail["shape"].tolist()) != (1, 140) or input_detail["dtype"] != np.int8:
        raise RuntimeError(f"Input TFLite inatteso: {input_detail['shape']} {input_detail['dtype']}.")
    if tuple(output_detail["shape"].tolist()) != (1, 5) or output_detail["dtype"] != np.int8:
        raise RuntimeError(f"Output TFLite inatteso: {output_detail['shape']} {output_detail['dtype']}.")

    keras_probabilities = model.predict(test_signals, batch_size=32, verbose=0).astype(np.float32)
    tflite_probabilities = []
    for signal in test_signals:
        interpreter.set_tensor(input_detail["index"], quantize(signal.reshape(1, 140), input_detail))
        interpreter.invoke()
        tflite_probabilities.append(dequantize(interpreter.get_tensor(output_detail["index"])[0], output_detail))
    tflite_probabilities = np.asarray(tflite_probabilities, dtype=np.float32)
    keras_predictions = np.argmax(keras_probabilities, axis=1).astype(np.int32) + 1
    tflite_predictions = np.argmax(tflite_probabilities, axis=1).astype(np.int32) + 1
    differences = np.abs(keras_probabilities - tflite_probabilities)
    disagreements = np.flatnonzero(keras_predictions != tflite_predictions).astype(int).tolist()

    precision, recall, f1_score, support = precision_recall_fscore_support(
        test_labels, tflite_predictions, labels=LABELS, zero_division=0
    )
    per_class = {
        str(label): {
            "precision": float(p), "recall": float(r), "f1_score": float(f), "support": int(s),
            "correct_predictions": int(np.sum((test_labels == label) & (tflite_predictions == label))),
        }
        for label, p, r, f, s in zip(LABELS, precision, recall, f1_score, support, strict=True)
    }
    report = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "keras_model": str(KERAS_MODEL),
        "tflite_model": str(TFLITE_MODEL),
        "tflite_sha256": model_hash,
        "tflite_size_bytes": TFLITE_MODEL.stat().st_size,
        "representative_dataset": str(DATASET_DIR / "ECG3200TRAIN.txt"),
        "test_dataset": str(DATASET_DIR / "ECG1000TEST.txt"),
        "input": {"shape": input_detail["shape"].tolist(), "dtype": str(input_detail["dtype"]), "quantization": list(input_detail["quantization"])},
        "output": {"shape": output_detail["shape"].tolist(), "dtype": str(output_detail["dtype"]), "quantization": list(output_detail["quantization"])},
        "keras_accuracy": float(accuracy_score(test_labels, keras_predictions)),
        "tflite_accuracy": float(accuracy_score(test_labels, tflite_predictions)),
        "tflite_balanced_accuracy": float(balanced_accuracy_score(test_labels, tflite_predictions)),
        "prediction_disagreements_vs_keras": len(disagreements),
        "disagreement_indices_zero_based": disagreements,
        "absolute_difference_max": float(np.max(differences)),
        "absolute_difference_mean": float(np.mean(differences)),
        "confusion_matrix": confusion_matrix(test_labels, tflite_predictions, labels=LABELS).tolist(),
        "per_class": per_class,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "int8_tflite_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (REPORT_DIR / "predictions_keras_vs_int8.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index_zero_based", "true_class", "keras_predicted_class", "int8_predicted_class", "max_abs_difference", "mean_abs_difference"])
        for index, (truth, keras_pred, int8_pred, diff) in enumerate(zip(test_labels, keras_predictions, tflite_predictions, differences, strict=True)):
            writer.writerow([index, int(truth), int(keras_pred), int(int8_pred), float(np.max(diff)), float(np.mean(diff))])
    print(f"TFLite INT8 creato: {TFLITE_MODEL}")
    print(f"SHA-256: {model_hash}; dimensione: {TFLITE_MODEL.stat().st_size} byte")
    print(f"Input: {input_detail['shape'].tolist()} {input_detail['dtype']} scale={input_detail['quantization']}")
    print(f"Output: {output_detail['shape'].tolist()} {output_detail['dtype']} scale={output_detail['quantization']}")
    print(f"Accuracy Keras={report['keras_accuracy']:.4%}; TFLite INT8={report['tflite_accuracy']:.4%}; discordanti={len(disagreements)}")
    print(f"Delta uscite: max={report['absolute_difference_max']:.3e}; media={report['absolute_difference_mean']:.3e}")


if __name__ == "__main__":
    main()
