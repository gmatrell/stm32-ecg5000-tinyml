# Scope and open provenance items

1. **Code and data rights.** No software license is included for the project-authored code at this stage. Dataset copies are excluded because the upstream [Zenodo record](https://zenodo.org/records/11186692) does not state redistribution terms in its Rights section.
2. **Training and data provenance.** Within `Esp_ECG5000`, locate or document the authoritative training procedure for the frozen Float32 model, its Float32 TFLite conversion, the creation and source of the 3,200/800/1,000 data split, and the exact Python package versions.
3. **H573 CMSIS-NN build provenance.** The CMSIS-NN experiment's link map contains optimized Fully Connected and Softmax symbols, and its ELF hash differs from the reference TFLM ELF. However, paths inside the CMSIS-NN map and firmware-size text point to the baseline `DENSE_INT8_TFLM` directory. Confirm the exact build and flashed binary before claiming a fully reproducible H573 CMSIS-NN firmware build.

The archived per-inference results and their numerical summaries were checked when this reviewer package was prepared. Firmware compilation and hardware reruns have not been performed from this selected package.


