# Edge-AI Air Quality Monitoring Station

An autonomous, headless air quality monitoring node that runs entirely on local hardware — no cloud, no internet dependency. Built around a dual-tier architecture (Arduino Uno + Raspberry Pi 4), it logs gas and particulate sensor data, runs it through a self-healing preprocessing pipeline, and produces real-time PM2.5 predictions and 24-hour AQI forecasts using both a classic Random Forest model and a lightweight deep learning time-series foundation model (SEMPO).

## Why this project

Standard air quality monitoring relies on expensive, mechanically fragile laser-scattering particle counters. This project explores whether a cluster of cheap, durable metal-oxide gas sensors can be combined with edge machine learning to produce a reliable software-defined alternative — one that runs continuously, in the field, without a constant network connection or human supervision.

## System Architecture

The node is split across two tiers connected over a USB serial UART bridge running at 9600 baud:

| Tier | Hardware | Responsibility |
|---|---|---|
| **Telemetry Tier** | Arduino Uno | High-frequency raw sensor acquisition from MQ135, MQ7, FC22, and SGP41 |
| **Processing Tier** | Raspberry Pi 4 | Background daemon, data cleansing, ML inference, AQI dashboard |

**Sensors used:**
- **MQ135** — Benzene, CO₂, NH₃ (general air quality)
- **MQ7** — Carbon monoxide, via a two-phase heating cycle (1.5V/60s measure, 5V/90s clean)
- **FC22** — Combustible gases / LPG leak detection
- **SGP41** — Digital VOC and NOx indices (I²C)
- **PMS7003** — Laser-scattering PM1.0 / PM2.5 / PM10

Each serial packet follows the format:

```
<MQ135, MQ7, FC22, SGP41_VOC, SGP41_NOX, PM1.0, PM2.5, PM10>
```

### Headless Persistence

The data-logging pipeline runs as a native `systemd` service (`aqi_3day.service`), so it survives reboots, requires no open terminal session, and auto-restarts within 10 seconds of any crash. It also handles dynamic USB port reassignment (`/dev/ttyACM0` ↔ `/dev/ttyACM1`) via a self-healing retry loop.

## The Core Engineering Problem: Dirty Sensor Data

Real-world field testing surfaced three categories of hardware/transmission failure:

| Failure Mode | Symptom |
|---|---|
| Buffer mashing | Serial frames merge, producing impossible readings (e.g. MQ135 spiking to 5.1 × 10⁶) |
| Laser glitches | PMS7003 timing errors produce PM2.5 spikes past 25,000 µg/m³ |
| Total dropouts | Flatline 0.0 readings across all channels during connection loss |

A statistical boundary filter (`preprocess_data.py`) screens every incoming packet against physical operating limits — gas voltage ≤ 1023 and PM2.5 between 0.1–800 — and drops anything outside range. In a 59-row field test, this filter caught and removed 57 corrupted rows, correcting a falsely "Hazardous" prediction of 3,724.80 µg/m³ down to a validated real-world baseline.

## Machine Learning Layer

Two models were implemented and compared:

**1. Random Forest Regressor** — an ensemble of bootstrapped decision trees mapping raw gas voltages to PM2.5 density. Lightweight and fast (`O(D·N)` inference), well-suited to constrained edge hardware.

**2. SEMPO (Spectral-Enhanced Mixture of Prompts)** — a 6.5M-parameter time-series foundation model, pretrained on 83M environmental data points. Its Energy-Aware Spectral Decomposition (EASD) module uses an FFT to split incoming signals into high- and low-energy frequency bands, isolating genuine ambient trends from sensor drift and noise spikes. A Mixture-of-Prompts Transformer then adapts a frozen backbone to local conditions on the fly using a pool of lightweight prompt experts — without retraining the whole model. Inference completes in ~22 seconds on a Raspberry Pi 4.

## Live Dashboard

A terminal-based dashboard displays real-time sensor readings, classifies predicted PM2.5 into standard AQI health bands, and projects a 24-hour forecast using an autoregressive moving average filter.

## Key Results

- Preprocessing filter: 57/59 corrupted rows correctly identified and dropped in field testing
- Random Forest baseline prediction corrected from an erroneous 3,724.80 µg/m³ to a validated ~28–78 µg/m³ range
- SEMPO inference runtime: ~22 seconds on Raspberry Pi 4 CPU, no GPU required

## Future Work

- Automated 24-hour re-calibration loop that retrains models on freshly cleaned data
- Porting the Random Forest model to ultra-low-power microcontrollers (ESP32 / Pi Pico) via TinyML

## Tech Stack

`Python` · `Arduino C++` · `scikit-learn` · `systemd` · `Raspberry Pi 4` · `Arduino Uno`

---

*This project was completed as a final-year B.Tech major project in Computer Engineering at NIAMT Ranchi, under the supervision of Dr. Manju Mathew.*
