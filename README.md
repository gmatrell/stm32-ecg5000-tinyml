# ECG5000 TinyML on three STM32 boards

This reviewer package is selected **only from `Esp_ECG5000`**. It documents 15 deployments of a frozen Dense 140–32–16–5 ECG5000 classifier: five variants on each of NUCLEO-L4R5ZI, STM32F746G-DISCO, and STM32H573I-DK. The variants are Float32 and INT8 with X-CUBE-AI, Float32 and INT8 with reference TFLM kernels, and INT8 with TFLM/CMSIS-NN kernels.

## Start here

- [Results](RESULTS.md) lists all 15 configurations, measured inference times, and theoretical energy estimates.
- [Data and provenance](DATA.md) records the study split hashes and explains why dataset files are excluded from this repository.
- [Build notes](BUILD_NOTES.md) describe the selected firmware and model artifacts and their external dependencies.
- [Serial protocol](SERIAL_PROTOCOL.md) defines the host/board exchange.

Each experiment contains its frozen TFLite model, a `results/experiment_summary.json`, the complete run's `hardware_predictions.csv` and `summary.json`, an original theoretical-energy report, a sanitized firmware-size summary, and application firmware source. Each CSV row includes the DWT time reported by the firmware.

## Interpretation

The DWT time was **measured on the MCU** and sent to the host. Energy was then **estimated** from that measured time and a profile or datasheet current. No current or complete-board energy was physically measured. NUCLEO and F746 estimates include modeled enabled-peripheral currents; the H573 estimate is an MCU-only typical LDO reference. The profiles differ across platforms, so their energy values should not be interpreted as an isolated CPU or clock comparison.

The source folder for STM32H573I-DK is historically named `STM32H753I-DK`; the experiment records identify the actual board as H573.

This repository provides the selected evidence for reviewer analysis. [Scope and open items](OPEN_ITEMS.md) records publication choices and remaining provenance limits.

