# gp_metrics_summary.py
import os
import json
import pandas as pd

OUT_DIR = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project\Results"

def load_all_metrics():
    rows = []
    for f in os.listdir(OUT_DIR):
        if f.startswith("metrics_") and f.endswith(".json"):
            p = os.path.join(OUT_DIR, f)
            try:
                with open(p, 'r', encoding='utf-8') as fh:
                    m = json.load(fh)
                # add filename info (tag)
                m['_file'] = f
                rows.append(m)
            except Exception:
                # skip unreadable files
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # prefer a stable ordering if rmse present
    if 'rmse' in df.columns:
        df = df.sort_values('rmse').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def main():
    df = load_all_metrics()
    if df.empty:
        print('No metrics_*.json found in', OUT_DIR)
        return

    out_xlsx = os.path.join(OUT_DIR, 'metrics_summary.xlsx')
    try:
        # try to write Excel (requires openpyxl or xlsxwriter)
        df.to_excel(out_xlsx, index=False)
        print('Saved:', out_xlsx)
    except Exception as e:
        # fallback to CSV
        out_csv = os.path.join(OUT_DIR, 'metrics_summary.csv')
        df.to_csv(out_csv, index=False)
        print('Excel write failed (', str(e), '), saved CSV instead:', out_csv)


if __name__ == '__main__':
    main()
