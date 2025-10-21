"""
Compare selected time-domain features between the C pipeline output and the
golden Python-generated CSV (Preprocessed_Datasets/MVT13_Final_MVT_Data.csv).

Usage:
    python compare_features.py --gold ../Preprocessed_Datasets/MVT13_Final_MVT_Data.csv \
        --out MVT13_features_output.csv --n 20

The script prints per-feature max absolute difference across matched rows and
flags mismatches beyond a small tolerance.
"""
import argparse
import pandas as pd
import numpy as np

FEATURES = [
    'Spike_Count','Dip_Count','Spike_Prom_Sum','Spike_Width_Mean_Ms',
    'Longest_Flat','Hf_Energy','Spectral_Entropy','Roll_Var',
    'Edge_Start_Diff','Edge_End_Diff','Min_Drop','Recovery_Slope',
    'Poly_Resid','Segment_Slope_Var','Zero_Cross_Rate',
    'Step_Count_Sustained','Max_Step_Mag','Bp_Low','Bp_Mid','Bp_High',
    'Bp_Mid_Ratio','Bp_High_Ratio','Resid_Spectral_Entropy',
    'Rel_Below_Frac','Rel_Below_Longest_Ms','Win_Range_Max',
    'Tail_Std','Tail_Ac1','Crest_Factor','Line_Length','Mid_Duty_Cycle_Low'
]

def load_csv(path):
    return pd.read_csv(path, dtype=str)


def compare(gold_df, out_df, n=20, tol=1e-3):
    # normalize id column name
    idcol = None
    for c in gold_df.columns:
        if 'Test' in c and 'Detail' in c:
            idcol = c; break
    if idcol is None:
        raise RuntimeError('Could not find Test_Record_Detail_ID column in gold CSV')
    gold_df[idcol] = gold_df[idcol].astype(str)
    out_df[idcol] = out_df[idcol].astype(str)

    common = sorted(list(set(gold_df[idcol]).intersection(set(out_df[idcol]))))
    if not common:
        print('No overlapping IDs between gold and out')
        return
    sel = common[:n]
    print(f'Comparing {len(sel)} rows (sampled by ID)')

    report = []
    for feat in FEATURES:
        if feat not in gold_df.columns or feat not in out_df.columns:
            report.append((feat, None, 'MISSING'))
            continue
        diffs = []
        for tid in sel:
            gval = gold_df.loc[gold_df[idcol]==tid, feat].values
            oval = out_df.loc[out_df[idcol]==tid, feat].values
            if len(gval)==0 or len(oval)==0:
                continue
            try:
                gv = float(gval[0])
                ov = float(oval[0])
                diffs.append(abs(gv-ov))
            except Exception:
                # skip non-numeric
                pass
        if not diffs:
            report.append((feat, None, 'NO_DATA'))
        else:
            m = float(np.nanmax(diffs))
            ok = m <= tol or np.isclose(m, 0.0, atol=tol)
            report.append((feat, m, 'OK' if ok else 'MISMATCH'))

    # Print report
    print('\nFeature comparison report:')
    for feat, m, status in report:
        if m is None:
            print(f'{feat:30s} {status}')
        else:
            print(f'{feat:30s} {status:8s} max_abs_diff={m:.6g}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gold', default='../Preprocessed_Datasets/MVT13_Final_MVT_Data.csv')
    p.add_argument('--out', default='../Preprocessed_Datasets/MVT13_features_output.csv')
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--tol', type=float, default=1e-3)
    args = p.parse_args()

    gold = load_csv(args.gold)
    out = load_csv(args.out)
    compare(gold, out, n=args.n, tol=args.tol)
