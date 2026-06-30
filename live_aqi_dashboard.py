"""
live_aqi_dashboard.py

Real-time terminal dashboard for the Edge-AI Air Quality Monitoring Station.

Reads live sensor packets over USB serial from the Arduino telemetry tier,
runs them through the boundary filter, feeds clean readings to the
on-chip Random Forest model (and optionally SEMPO), classifies the result
into standard AQI health bands, and projects a 24-hour forecast using a
simple autoregressive moving average.

Designed to run continuously as a systemd background service
(see aqi_3day.service) — handles serial port reassignment and reconnects
automatically if the connection drops.

Usage:
    python3 live_aqi_dashboard.py --port /dev/ttyACM0 --model rf
"""

import argparse
import time
import serial
import joblib
import pandas as pd
from collections import deque

from preprocess_data import is_valid_row, REQUIRED_COLUMNS
from sempo_inference import sempo_predict

FEATURE_COLUMNS = ["MQ135", "MQ7", "FC22", "SGP41_VOC", "SGP41_NOX"]
HISTORY_WINDOW = 24  # rows kept for the moving-average forecast

AQI_BANDS = [
    (0, 50, "Good", "🟢"),
    (51, 100, "Moderate (Acceptable Ambient Quality)", "🟡"),
    (101, 150, "Unhealthy for Sensitive Groups", "🟠"),
    (151, 200, "Unhealthy (Action Required / Ventilate)", "🔴"),
    (201, 300, "Very Unhealthy", "🟣"),
    (301, 500, "Hazardous", "🟤"),
]


def pm25_to_aqi(pm25: float) -> int:
    """Simplified linear PM2.5 -> AQI conversion for dashboard display purposes."""
    breakpoints = [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500),
    ]
    for c_lo, c_hi, a_lo, a_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return round(((a_hi - a_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + a_lo)
    return 500


def classify_aqi(aqi: int) -> str:
    for lo, hi, label, icon in AQI_BANDS:
        if lo <= aqi <= hi:
            return f"{icon} {label}"
    return "⚫ Out of range"


def connect_serial(port: str, baud: int = 9600, retries: int = 5) -> serial.Serial:
    """Attempt to open the serial port, retrying on failure (handles dynamic
    USB port reassignment between /dev/ttyACM0 and /dev/ttyACM1)."""
    candidate_ports = [port, "/dev/ttyACM0", "/dev/ttyACM1"]
    for attempt in range(retries):
        for p in candidate_ports:
            try:
                conn = serial.Serial(p, baud, timeout=2)
                print(f"[INFO] Connected to serial port: {p}")
                return conn
            except serial.SerialException:
                continue
        print(f"[WARN] Serial connection attempt {attempt + 1}/{retries} failed. Retrying...")
        time.sleep(2)
    raise RuntimeError("Could not establish serial connection after multiple retries.")


def parse_packet(line: str):
    """Parses a raw CSV packet into a labeled row dict, or None if malformed."""
    try:
        parts = [float(x) for x in line.strip().split(",")]
        if len(parts) != 8:
            return None
        return dict(zip(
            ["MQ135", "MQ7", "FC22", "SGP41_VOC", "SGP41_NOX", "PM1_0", "PM2_5", "PM10"],
            parts,
        ))
    except ValueError:
        return None


def render_dashboard(reading: dict, aqi: int, forecast_aqi: int, model_name: str):
    print("=" * 65)
    print(f"  🌿 EDGE-AI ENVIRONMENTAL MONITORING NODE ({model_name.upper()})  🌿")
    print("=" * 65)
    print(f"📊 CURRENT AIR QUALITY INDEX : AQI: {aqi} | Status: {classify_aqi(aqi)}")
    print(f"🔬 Predicted PM2.5 Density   : {reading['PM2_5']:.2f} µg/m³")
    print("-" * 65)
    print("🧪 METAL-OXIDE CHEMICAL GAS LAYER:")
    print(f"  - MQ135 NH3/CO2 Profile   : {reading['MQ135'] / 1023 * 100:.1f}% load count")
    print(f"  - MQ7 Carbon Monoxide Load: {reading['MQ7'] / 1023 * 100:.1f}% load count")
    print(f"  - FC22 Combustible Matrix : {reading['FC22'] / 1023 * 100:.1f}% load count")
    print(f"  - SGP41 Core Indexes      : VOC index: {reading['SGP41_VOC']:.1f} | "
          f"NOx index: {reading['SGP41_NOX']:.1f}")
    print("-" * 65)
    print("📅 24-HOUR FORECAST PREDICTION MODEL (TOMORROW):")
    print(f"  - Projected Tomorrow AQI : AQI: {forecast_aqi} | {classify_aqi(forecast_aqi)}")
    print("=" * 65)


def run_dashboard(port: str, model_choice: str, model_path: str):
    ser = connect_serial(port)
    history = deque(maxlen=HISTORY_WINDOW)

    if model_choice == "rf":
        model = joblib.load(model_path)

    while True:
        try:
            raw_line = ser.readline().decode("utf-8", errors="ignore")
            if not raw_line:
                continue

            reading = parse_packet(raw_line)
            if reading is None:
                continue

            row = pd.Series(reading)
            if not is_valid_row(row):
                print("[WARN] Dropped corrupted live packet.")
                continue

            history.append(reading)
            hist_df = pd.DataFrame(history)

            if model_choice == "rf":
                X = pd.DataFrame([reading], columns=FEATURE_COLUMNS)
                reading["PM2_5"] = float(model.predict(X)[0])
            else:
                reading["PM2_5"] = sempo_predict(hist_df)

            aqi = pm25_to_aqi(reading["PM2_5"])

            # Simple autoregressive moving-average forecast over recent AQI history
            recent_aqis = [pm25_to_aqi(r["PM2_5"]) for r in history]
            forecast_aqi = round(sum(recent_aqis) / len(recent_aqis))

            render_dashboard(reading, aqi, forecast_aqi, model_choice)

        except (serial.SerialException, OSError):
            print("[WARN] Serial connection lost. Attempting to reconnect...")
            ser = connect_serial(port)

        except KeyboardInterrupt:
            print("\n[INFO] Dashboard stopped by user.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Edge-AI AQI terminal dashboard.")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port for Arduino telemetry")
    parser.add_argument("--model", choices=["rf", "sempo"], default="rf", help="Inference model to use")
    parser.add_argument("--model-path", default="rf_model.joblib", help="Path to trained RF model")
    args = parser.parse_args()

    run_dashboard(args.port, args.model, args.model_path)
