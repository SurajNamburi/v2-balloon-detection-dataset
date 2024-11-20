# V2 Balloon Detection Dataset

This repository contains the V2 Balloon Detection Dataset used for the Object Detection project in ML and CV classes.

## Dataset Structure

object_detection_project/
├── data/
│   ├── images/
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
   ```bash
   pip install numpy pandas opencv-python scikit-image scikit-learn joblib opencv-python requests


4. **Run the Program:**
   ```bash
   python src/object_detection.py --csv data/balloon-data.csv --model_dir models/ --log logs/experiment_log.txt --lr 0.001 --lambda_param 0.01 --n_iters 1000


Parameters:

--csv: Path to the balloon-data.csv file.

--model_dir: Directory where the trained model will be saved.

--log: Path to the log file.

--lr: Learning rate for the SVM classifier.

--lambda_param: Regularization parameter for the SVM classifier.

--n_iters: Number of iterations for SVM training.
