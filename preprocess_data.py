"""
preprocess_data.py

Software-defined statistical boundary filter for the Edge-AI Air Quality
Monitoring Station.

Raw field data from low-cost MOS gas sensors and the PMS7003 laser sensor
is prone to serial buffer mashing, frame corruption, and total dropouts.
This script screens every logged row against known physical operating
limits and drops anything that falls outside them before the data reaches
the machine learning layer.

Validity rules:
  - All analog gas sensor voltages (MQ135, MQ7, FC22) must be <= 1023
    (10-bit ADC ceiling on the Arduino Uno).
  - PM2.5 reading must fall between 0.1 and 800 ug/m3.

Usage:
    python3 preprocess_data.py --input master_dataset.csv --output clean_dataset.csv
"""

import argparse
import sys
import pandas as pd

ANALOG_MAX = 1023
PM25_MIN = 0.1
PM25_MAX = 800

REQUIRED_COLUMNS = [
    "Time", "MQ135", "MQ7", "FC22",
    "SGP41_VOC", "SGP41_NOX", "PM1_0", "PM2_5", "PM10",
]


def is_valid_row(row: pd.Series) -> bool:
    """Apply boundary checks to a single sensor reading row."""
    try:
        gas_values = [row["MQ135"], row["MQ7"], row["FC22"]]
        if any(v > ANALOG_MAX or v < 0 for v in gas_values):
            return False

        pm25 = row["PM2_5"]
        if not (PM25_MIN <= pm25 <= PM25_MAX):
            return False

        return True
    except (TypeError, ValueError):
        return False


def preprocess(input_path: str, output_path: str) -> None:
    print("[INFO] Initializing Edge-AI Preprocessing Module...")

    df = pd.read_csv(input_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[ERROR] Input file is missing required columns: {missing}")
        sys.exit(1)

    total_frames = len(df)
    valid_mask = df.apply(is_valid_row, axis=1)

    for idx, valid in valid_mask.items():
        if not valid:
            print(f"[WARN] Line {idx + 1}: anomalous reading detected. Dropping frame...")

    clean_df = df[valid_mask].reset_index(drop=True)
    dropped = total_frames - len(clean_df)

    clean_df.to_csv(output_path, index=False)

    print("[INFO] Preprocessing Complete. "
          f"Total frames analyzed: {total_frames}")
    print(f"[INFO] Corrupted packets dropped: {dropped} | "
          f"Valid baseline rows retained: {len(clean_df)}")
    print(f"[INFO] Clean dataset written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boundary-filter raw AQI sensor logs.")
    parser.add_argument("--input", default="master_dataset.csv", help="Path to raw input CSV")
    parser.add_argument("--output", default="clean_dataset.csv", help="Path to write cleaned CSV")
    args = parser.parse_args()

    preprocess(args.input, args.output)
