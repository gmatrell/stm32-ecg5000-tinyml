ECG5000 DATASET

Source and redistribution
-------------------------
The ECG5000 dataset is publicly downloadable from the Time Series Classification archive:
https://timeseriesclassification.com/description.php?Dataset=ECG5000
An archive copy is also available at https://zenodo.org/records/11186692.

Public availability does not itself authorize redistribution. The ECG5000 archive record does not state terms permitting this project to republish the processed records. This repository therefore links to the source and does not include copies of the dataset.

Partition used in the study
---------------------------
The standard ECG5000 distribution has 500 training and 4,500 test records. Our study used a stratified partition of the 5,000 records: 3,200 for training, 800 for validation, and 1,000 for testing, with a reported fixed random seed of 123. The original training subset has only two examples of the minority Class 5. The study partition increases that number to 15 in training while approximately preserving the overall class proportions; the validation and test subsets contain four and five Class 5 examples, respectively.

This partition has no special methodological significance beyond representing the minority class more adequately for training and providing separate validation and test sets. All 15 embedded implementations were evaluated against the same 1,000-record test set using the same frozen neural network. Classification metrics from this nonstandard partition are not directly comparable with results based on the original ECG5000 split.

The split-generation script and source-row indices have not been found within Esp_ECG5000. Therefore, the stated random seed alone is insufficient to reconstruct the exact membership of the three subsets from the downloaded archive files.
