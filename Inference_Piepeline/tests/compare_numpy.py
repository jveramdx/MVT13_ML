from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.signal import find_peaks, medfilt

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR.parent / "Preprocessed_Datasets" / "MVT13_Starter_Voltage_Array.csv"
TEST_BIN = SCRIPT_DIR / ("test_numpy_functions.exe" if os.name == "nt" else "test_numpy_functions")
METHODS = ("fft", "rfftfreq", "find_peaks", "medfilt")


def load_samples(limit: int = 50) -> List[Dict[str, np.ndarray]]:
    samples: List[Dict[str, np.ndarray]] = []
    with open(DATASET_PATH, newline="") as fp:
        reader = csv.reader(fp)
        next(reader, None)  # header
        for idx, row in enumerate(reader):
            if len(samples) >= limit:
                break
            if not row:
                continue
            try:
                record_id = int(float(row[0]))
            except ValueError:
                continue
            values: List[float] = []
            for token in row[1:]:
                token = token.strip()
                if not token:
                    continue
                try:
                    values.append(float(token))
                except ValueError:
                    continue
            samples.append(
                {
                    "index": len(samples),
                    "record_id": record_id,
                    "values": np.asarray(values, dtype=np.float32),
                }
            )
    return samples


def parse_float_sequence(text: str) -> np.ndarray:
    if not text:
        return np.array([], dtype=np.float32)
    return np.asarray([float(tok) for tok in text.split(";") if tok], dtype=np.float32)


def parse_int_sequence(text: str) -> np.ndarray:
    if not text:
        return np.array([], dtype=np.int32)
    return np.asarray([int(tok) for tok in text.split(";") if tok], dtype=np.int32)


def parse_complex_sequence(text: str) -> np.ndarray:
    if not text:
        return np.array([], dtype=np.complex64)
    values: List[np.complex64] = []
    for tok in text.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        tok = tok.replace("i", "j")
        values.append(np.complex64(complex(tok)))
    return np.asarray(values, dtype=np.complex64)


PARSERS = {
    "fft": parse_complex_sequence,
    "rfftfreq": parse_float_sequence,
    "find_peaks": parse_int_sequence,
    "medfilt": parse_float_sequence,
}

SERIALIZERS = {
    "fft": lambda arr: ";".join(f"{val.real:.9g}{val.imag:+.9g}i" for val in arr),
    "rfftfreq": lambda arr: ";".join(f"{float(val):.9g}" for val in arr),
    "find_peaks": lambda arr: ";".join(str(int(val)) for val in arr),
    "medfilt": lambda arr: ";".join(f"{float(val):.9g}" for val in arr),
}


def run_c_tests(method: str) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    if not TEST_BIN.exists():
        raise FileNotFoundError(f"C test binary not found at {TEST_BIN}")

    cmd = [str(TEST_BIN)]
    if method != "all":
        cmd.append(method)

    raw = subprocess.check_output(cmd, text=True, cwd=str(SCRIPT_DIR))
    results: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 3)
        if len(parts) != 4:
            continue
        name, sample_idx_str, record_id_str, payload = parts
        if name not in PARSERS:
            continue
        try:
            sample_idx = int(sample_idx_str)
            record_id = int(record_id_str)
        except ValueError:
            continue
        parser = PARSERS[name]
        parsed = parser(payload)
        results.setdefault(name, {})[sample_idx] = {
            "record_id": record_id,
            "values": parsed,
        }
    return results


def compute_python_reference(method: str, values: np.ndarray) -> np.ndarray:
    if method == "fft":
        data = values.astype(np.complex64, copy=False)
        return np.fft.fft(data, norm=None).astype(np.complex64, copy=False)
    if method == "rfftfreq":
        n = values.shape[0]
        return np.fft.rfftfreq(n, d=0.01).astype(np.float32, copy=False)
    if method == "find_peaks":
        peaks, _ = find_peaks(values.astype(np.float32, copy=False))
        return peaks.astype(np.int32, copy=False)
    if method == "medfilt":
        n = values.shape[0]
        if n == 0:
            return np.array([], dtype=np.float32)
        k = 5 if n >= 5 else (n if n % 2 else max(1, n - 1))
        filtered = medfilt(values.astype(np.float32, copy=False), kernel_size=k)
        return filtered.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported method {method}")


def difference_metric(method: str, python_values: np.ndarray, c_values: Optional[np.ndarray]) -> float:
    if c_values is None:
        return float("nan")
    if method == "find_peaks":
        py_set = set(int(v) for v in python_values.tolist())
        c_set = set(int(v) for v in c_values.tolist())
        return float(len(py_set.symmetric_difference(c_set)))
    if python_values.shape != c_values.shape:
        return float("nan")
    diff = np.abs(python_values - c_values)
    if diff.size == 0:
        return 0.0
    return float(np.max(diff))


def serialize(method: str, values: Optional[np.ndarray]) -> str:
    if values is None:
        return ""
    if values.size == 0:
        return ""
    serializer = SERIALIZERS[method]
    return serializer(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare C numpy-function mimics against NumPy/SciPy using dataset samples.")
    parser.add_argument(
        "--method",
        choices=("all",) + METHODS,
        default="all",
        help="Run comparison for a single method or all methods.",
    )
    args = parser.parse_args()

    methods_to_run: Sequence[str] = METHODS if args.method == "all" else (args.method,)

    samples = load_samples(limit=50)
    c_results = run_c_tests(args.method)

    comparison_rows: List[Dict[str, object]] = []
    for method in methods_to_run:
        for sample in samples:
            idx = sample["index"]
            record_id = sample["record_id"]
            values = sample["values"]
            py_values = compute_python_reference(method, values)
            c_entry = c_results.get(method, {}).get(idx)
            c_values = None
            if c_entry and c_entry["record_id"] == record_id:
                c_values = c_entry["values"]
            diff = difference_metric(method, py_values, c_values)
            print(f"{method} sample {idx} (record {record_id}): diff metric = {diff}")
            comparison_rows.append(
                {
                    "method": method,
                    "sample_index": idx,
                    "record_id": record_id,
                    "python_result": serialize(method, py_values),
                    "c_result": serialize(method, c_values),
                    "difference_metric": diff,
                }
            )

    out_path = SCRIPT_DIR / "numpy_c_comparison.csv"
    with open(out_path, "w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["method", "sample_index", "record_id", "python_result", "c_result", "difference_metric"],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"Wrote comparison results to {out_path}")


if __name__ == "__main__":
    main()
