# Serial validation protocol

The host selects the board's serial port explicitly and uses 115200 baud, 8N1. Multibyte integers and IEEE-754 Float32 values are little-endian.

A request is 572 bytes:

```text
"ECG1" | request_id uint32 | 140 × float32 input samples | checksum uint32
```

A response is 40 bytes:

```text
"RES2" | request_id uint32 | status uint8 | predicted_class uint8 |
reserved uint16 | inference_time_us uint32 |
5 × float32 class probabilities | checksum uint32
```

The checksum is 32-bit FNV-1a. The firmware measures the inference window with the DWT cycle counter and sends `inference_time_us` in every response. The exact timed operations for each variant are recorded in its `results/experiment_summary.json`: X-CUBE-AI includes buffer preparation and the runtime call; TFLM includes the applicable input/output copies or quantization/dequantization around `Invoke`. UART transfer, protocol checksums, and argmax are outside the DWT window. Host round-trip time is a separate measurement.
