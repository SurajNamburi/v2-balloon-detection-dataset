# Balloon Detection Using Manual HOG Feature Extraction and SVM Classification

This repository contains the V2 Balloon Detection Dataset used for the Object Detection project in ML and CV classes.

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

--csv: Path to balloon-data.csv file.

--model_dir: Directory where trained model saved.

--log: Log file path.

--lr: Learning rate.

--lambda_param: Regularization parameter.

--n_iters: Iterations for SVM training.
