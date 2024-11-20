# V2 Balloon Detection Dataset

This repository contains the V2 Balloon Detection Dataset used for the Object Detection project in ML and CV classes.

## Dataset Structure

v2-balloon-detection-dataset/ ├── annotations/ │ └── balloon-data.csv ├── images/ │ ├── 34020010494_e5cb88e1c4_k.jpg │ ├── 25899693952_7c8b8b9edc_k.jpg │ └── ... └── README.md

## How to Download and Extract

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/SurajNamburi/v2-balloon-detection-dataset.git
   cd v2-balloon-detection-dataset

2. Run the program
python src/object_detection.py --csv data/balloon-data.csv --images data/images/ --model_dir models/ --log logs/experiment_log.txt --C 0.01 --kernel linear

Parameters:
--C: Regularization parameter for SVM. You can experiment with different values (e.g., 0.1, 1, 10) to see how it affects performance.
--kernel: Kernel type for SVM. Options include linear, rbf, poly, etc.
