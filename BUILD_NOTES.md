# Firmware and model provenance

## Included from `Esp_ECG5000`

Each experiment includes the application firmware source, the host-side `validate_hardware_serial.py`, its frozen TFLite model, a complete run's CSV and summary, and a theoretical-energy report. The Float32 Keras model and architecture record are taken from `NUCLEO-L4R5ZI/DENSE_F32/model/`. The INT8 conversion script is `STM32F746G-DISCO/DENSE_INT8/convert_validate_int8.py`; it uses the frozen Keras model and the 3,200-record training file for representative calibration.

A training script for the Float32 model and the original Float32 TFLite conversion procedure were **not found within `Esp_ECG5000`**. The frozen model files allow reviewers to inspect and verify the deployment results, but retraining the model from this package is not documented. The exact Python environment lock was also not found within this project folder.

## External build dependencies

The experiment build records identify STM32CubeIDE 1.11.2, Arm GNU Toolchain 13.3.1, X-CUBE-AI / ST Edge AI 10.2.0, STM32Cube firmware L4 1.18.2, F7 1.17.4, and H5 1.7.0. The H573 CMSIS-NN configuration records TFLM source commit `bfeb43aa795adad4044bbf2e709062b1b989a98a`.

ST firmware packages, the X-CUBE-AI runtime and generated code, TFLM/CMSIS source trees, machine-specific build scripts, and ELF/BIN outputs are not copied here. This package supports **offline review and recalculation of the reported results**; it has not been verified as a self-contained firmware build. The application source and serial validators document the measured inference window, and the [serial protocol](SERIAL_PROTOCOL.md) documents the host/board exchange.

The archived linker maps for the three CMSIS-NN configurations contain `arm_fully_connected_wrapper_s8` and `arm_softmax_s8`, indicating that these optimized kernels were linked in the corresponding builds.
