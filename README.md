# V2 Balloon Detection Dataset

This repository contains the V2 Balloon Detection Dataset used for the Object Detection project in ML and CV classes.

## Dataset Structure

object_detection_project/
├── data/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── balloon-data.csv
├── logs/
│   └── experiment_log.txt
├── src/
│   └── object_detection.py
└── README.md

## How to Download and Extract

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/SurajNamburi/v2-balloon-detection-dataset.git
   cd v2-balloon-detection-dataset

2. **Install Dependencies:**
pip install numpy pandas opencv-python scikit-image scikit-learn joblib 


3. **Run the Program:**
   python src/object_detection.py --csv data/balloon-data.csv --images data/images/ --model_dir models/ --log logs/experiment_log.txt --C 0.01 --kernel linear

Parameters:
--C: Regularization parameter for SVM. You can experiment with different values (e.g., 0.1, 1, 10) to see how it affects performance.
--kernel: Kernel type for SVM. Options include linear, rbf, poly, etc.
