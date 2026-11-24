# ECG5000 data provenance

The experiments used three local split files: `ECG3200TRAIN.txt`, `ECG800VALIDATION.txt`, and `ECG1000TEST.txt`. They are excluded from this reviewer package; see `Dataset/README.txt` for the upstream download location and partition rationale. All 15 experiment summaries record the same SHA-256 for the 1,000-record test file.

| File | Records | SHA-256 |
| --- | ---: | --- |
| `ECG3200TRAIN.txt` | 3,200 | `519004ed0d3d72aa4a863d8a7281a531ff1984d893fd3df83cb2c4dce8b2acf4` |
| `ECG800VALIDATION.txt` | 800 | `a2ef33dad3dbbbf86558af51d9deeb5a874a384d8a03e1ee5966fdcfdcbba859` |
| `ECG1000TEST.txt` | 1,000 | `261011d52919c2e6da91b135e228ad3c6725b15a9bf5ff5bbdc86b0b85b52bcb` |

The hashes and row counts above identify the three local split files used in the experiments.

ECG5000 is described by the [Time Series Classification archive](https://timeseriesclassification.com/description.php?Dataset=ECG5000), which distributes a 500/4,500 train/test split; see also its [Zenodo record](https://zenodo.org/records/11186692). **No script that creates the 3,200/800/1,000 files is present within `Esp_ECG5000`**, so this package does not claim to reproduce their exact membership from the archive files. The split procedure and source-file provenance require confirmation. The split files are excluded from this repository because the upstream record does not state redistribution terms.
