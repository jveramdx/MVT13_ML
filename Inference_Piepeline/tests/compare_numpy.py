# compare_numpy.py
import os
import subprocess
import numpy as np
from scipy.signal import find_peaks, medfilt
import csv

def parse_complex_list(s):
    vals = []
    for tok in s.split(";"):
        if not tok:
            continue
        if "i" in tok:
            tok = tok.replace("i", "j")
            vals.append(complex(tok))
        else:
            vals.append(float(tok))
    return vals


def main():
    script_dir = os.path.dirname(__file__)
    test_bin = os.path.join(script_dir, "test_numpy_functions.exe")
    c_output = subprocess.check_output([test_bin], text=True).splitlines()

    results = []
    for line in c_output:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 2)
        if len(parts) != 3:
            continue
        func, test_id, out = parts
        try:
            tid = int(test_id)
        except ValueError:
            continue

        if func == "fft":
            n = 4 + tid
            angles = 2 * np.pi * np.arange(n, dtype=np.float32) / np.float32(n)
            a = (np.cos(angles) + 1j * np.sin(angles)).astype(np.complex64)
            ref = np.fft.fft(a, norm=None).astype(np.complex64)
            got = np.array(parse_complex_list(out), dtype=np.complex64)
        elif func == "rfftfreq":
            n = 8 + 2 * tid
            # Use float32 for d to match C implementation precision
            d = np.float32(0.1 + 0.05 * tid)
            ref = np.fft.rfftfreq(n, d).astype(np.float32)
            got = np.array([float(x) for x in out.split(";")], dtype=np.float32)
        elif func == "find_peaks":
            x = np.array([np.sin(0.3 * i) + (0.5 if tid % 3 == 0 else 0) for i in range(20)], dtype=np.float32)
            if tid % 2 == 0:
                peaks, _ = find_peaks(x, height=0.2, distance=(tid % 3) + 1)
            else:
                peaks, _ = find_peaks(x, distance=(tid % 3) + 1)
            ref = peaks
            got = np.array([int(i) for i in out.split(";") if i], dtype=np.int32)
        elif func == "medfilt":
            kern = (tid % 3) * 2 + 3
            vol = np.array([(i + tid) % 5 for i in range(9)], dtype=np.float32)
            ref = medfilt(vol, kernel_size=kern).astype(np.float32)
            got = np.array([float(x) for x in out.split(";")], dtype=np.float32)
        else:
            continue

        # compare with both relative and absolute tolerances to accommodate small magnitudes
        if func == "fft":
            # compare magnitude and phase with practical tolerances
            mag_ref = np.abs(ref)
            mag_got = np.abs(got)
            phase_ref = np.angle(ref)
            phase_got = np.angle(got)
            # tolerances: relative mag 1e-5, absolute phase 1e-2 rad
            mag_ok = np.max(np.abs(mag_ref - mag_got) / (mag_ref + np.finfo(np.float32).eps)) < 1e-5
            phase_ok = np.max(np.abs((phase_ref - phase_got + np.pi) % (2*np.pi) - np.pi)) < 1e-2
            ok = bool(mag_ok and phase_ok)
        elif func == "rfftfreq":
            # rfftfreq differences small but accept absolute up to 5e-6
            ok = np.allclose(ref, got, rtol=0, atol=5e-6)
        else:
            ok = np.allclose(ref, got, rtol=1e-6, atol=5e-7)
        maxdiff = float(np.max(np.abs(ref - got)))
        status = 'PASS' if ok else 'FAIL'
        print(f"{func} {tid} {status} - max|diff|: {maxdiff}")
        if func == 'fft' and not ok:
            diffs = np.abs(ref - got)
            # compute exponent and mantissa for ref
            abs_ref = np.abs(ref)
            # avoid log of zero
            exp_ref = np.where(abs_ref > 0,
                               np.floor(np.log10(abs_ref)).astype(int),
                               0)
            mant_ref = np.where(abs_ref > 0,
                                ref / (10.0 ** exp_ref),
                                0.0)
            mant_got = np.where(abs_ref > 0,
                                got / (10.0 ** exp_ref),
                                0.0)
            mant_diff = np.abs(mant_ref - mant_got)
            rel_mant = np.where(np.abs(mant_ref) > 0,
                                mant_diff / np.abs(mant_ref),
                                diffs / (abs_ref + np.finfo(np.float32).eps))
            bad = np.where(rel_mant > 1e-6)[0]
            print(f"FFT mantissa-comparison failures (tid={tid}):")
            for i in bad:
                print(f" idx {i}: exp={exp_ref[i]}, mant_ref={mant_ref[i]}, mant_got={mant_got[i]}, mant_diff={mant_diff[i]}, rel_mant={rel_mant[i]}")
        results.append([func, tid, ref.tolist(), got.tolist(), maxdiff])

    csv_file = os.path.join(script_dir, "fft_compare_results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["func", "test_id", "numpy", "c_impl", "max_abs_diff"])
        for row in results:
            writer.writerow(row)
    print(f"Results written to {csv_file}")


if __name__ == "__main__":
    main()