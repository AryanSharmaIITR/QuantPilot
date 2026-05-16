import os
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT_DIR / "data" / "preprocessed" / "stock_data"
TARGET_FILE = ROOT_DIR / "data" / "preprocessed" / "stock_data_stacked.csv"


def stack_stock_csvs(source_dir: Path = SOURCE_DIR, target_file: Path = TARGET_FILE, stock_col: str = "stock_name") -> pd.DataFrame:
    """Read all CSV files under source_dir, add a stock name column, and save one stacked CSV."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    csv_files = sorted(source_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {source_dir}")

    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df[stock_col] = csv_file.stem
        frames.append(df)

    stacked_df = pd.concat(frames, ignore_index=True)
    stacked_df.to_csv(target_file, index=False)
    print(f"Saved stacked CSV with {len(stacked_df)} rows from {len(frames)} files to: {target_file}")


if __name__ == "__main__":
    stack_stock_csvs()
