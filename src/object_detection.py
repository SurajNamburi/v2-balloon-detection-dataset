import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import argparse
import logging
import json
import matplotlib.pyplot as plt

def load_data(csv_path, images_dir):
    df = pd.read_csv(csv_path)
    data = []
    labels = []

    logging.info(f'Reading CSV file: {csv_path}')
    logging.info(f'Number of rows in CSV: {len(df)}')

    for index, row in df.iterrows():
        fname = row['fname']
        num_balloons = row['num_balloons']
        image_path = os.path.join(images_dir, fname)
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Image {fname} not found.")
            continue  # Skip if image is not found
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = row['height'], row['width']

        # Parse bbox safely using json.loads
        try:
            bbox = json.loads(row['bbox'].replace("'", '"'))  # Replace single quotes with double quotes for valid JSON
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error for image {fname}: {e}")
            bbox = []  # Treat as no bounding boxes

        # Extract balloon regions
        for box in bbox:
            if not isinstance(box, dict):
                logging.warning(f"Invalid bbox format in image {fname}: {box}")
                continue  # Skip invalid bbox
            # Extract coordinates
            try:
                xmin = int(box['xmin'])
                ymin = int(box['ymin'])
                xmax = int(box['xmax'])
                ymax = int(box['ymax'])
                w = xmax - xmin
                h = ymax - ymin
            except (KeyError, ValueError, TypeError) as e:
                logging.error(f"Error parsing bbox in image {fname}: {box} - {e}")
                continue  # Skip if any key is missing or values are not integers

            # Validate bbox coordinates
            if xmin < 0 or ymin < 0 or w <= 0 or h <= 0 or xmax > width or ymax > height:
                logging.warning(f"Bbox out of bounds or invalid in image {fname}: {box}")
                continue  # Skip invalid bbox

            # Crop and resize the balloon region
            try:
                crop = image[ymin:ymin + h, xmin:xmin + w]
                crop = cv2.resize(crop, (64, 128))  # Resize to fixed size
                data.append(crop)
                labels.append(1)  # Balloon present
            except Exception as e:
                logging.error(f"Error processing bbox {box} in image {fname}: {e}")
                continue  # Skip if cropping/resizing fails

        # Generate negative samples (no balloon)
        try:
            # Number of negative samples per image can be adjusted as needed
            num_neg_samples = max(1, num_balloons)  # At least one negative sample
            for _ in range(num_neg_samples):
                neg_x = np.random.randint(0, max(width - 64, 1))
                neg_y = np.random.randint(0, max(height - 128, 1))
                neg_crop = image[neg_y:neg_y + 128, neg_x:neg_x + 64]
                neg_crop = cv2.resize(neg_crop, (64, 128))
                data.append(neg_crop)
                labels.append(0)  # No balloon
        except Exception as e:
            logging.error(f"Error generating negative sample for image {fname}: {e}")
            continue  # Skip if negative sample generation fails

    logging.info(f'Total positive samples: {labels.count(1)}')
    logging.info(f'Total negative samples: {labels.count(0)}')

    return np.array(data, dtype=np.uint8), np.array(labels, dtype=np.int32)


def extract_hog_features(images):
    hog_features = []
    for idx, image in enumerate(images):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        except cv2.error as e:
            logging.error(f'Error converting image {idx} to grayscale: {e}')
            continue  # Skip this image

        try:
            features = hog(gray,
                          orientations=9,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          block_norm='L2-Hys',
                          visualize=False,
                          feature_vector=True)
            hog_features.append(features)
        except Exception as e:
            logging.error(f'Error extracting HOG features from image {idx}: {e}')
            continue  # Skip this image
    return np.array(hog_features, dtype=np.float32)



def train_svm(X_train, y_train, C=0.01, kernel='linear'):
    clf = SVC(C=C, kernel=kernel, probability=True)
    clf.fit(X_train, y_train)
    return clf

def sliding_window(image, clf, window_size=(64, 128), step_size=16):
    detections = []
    prob_scores = []
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            window = image[y:y + window_size[1], x:x + window_size[0]]
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                continue
            features = extract_hog_features([window])
            pred = clf.predict(features)
            prob = clf.decision_function(features)
            if pred == 1:
                detections.append((x, y, window_size[0], window_size[1]))
                prob_scores.append(prob)
    return detections, prob_scores

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return acc, rmse

def setup_logging(log_file):
    logging.basicConfig(filename=log_file,
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)


def main(args):
    # Setup logging
    setup_logging(args.log)
    logging.info('Starting Object Detection Experiment')

    # Load data
    logging.info('Loading data...')
    data, labels = load_data(args.csv, args.images)
    logging.info(f'Total samples loaded: {len(data)}')
    logging.info(f'Total labels loaded: {len(labels)}')

    if len(data) == 0:
        logging.error('No data loaded. Exiting the program.')
        return

    # Print first 5 samples and labels for verification
    logging.info(f'First 5 labels: {labels[:5]}')
    logging.info(f'First 5 data samples shape: {[img.shape for img in data[:5]]}')

    # Feature extraction
    logging.info('Extracting HOG features...')
    features = extract_hog_features(data)
    logging.info(f'Features shape: {features.shape}')

    if len(features) == 0:
        logging.error('Feature extraction failed. No features extracted. Exiting the program.')
        return

    # Ensure features have the correct dimensions
    if len(features.shape) < 2:
        logging.error('Features do not have the expected 2D shape. Exiting the program.')
        return

    logging.info(f'Feature vector size: {features.shape[1]}')

    # Split data
    logging.info('Splitting data into train and test sets...')
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=labels)
    logging.info(f'Training samples: {len(X_train)}, Testing samples: {len(X_test)}')

    # Train SVM
    logging.info('Training SVM classifier...')
    clf = train_svm(X_train, y_train, C=args.C, kernel=args.kernel)
    logging.info('SVM training completed.')

    # Evaluate
    logging.info('Evaluating the model...')
    acc, rmse = evaluate_model(clf, X_test, y_test)
    logging.info(f'Test Accuracy: {acc * 100:.2f}%')
    logging.info(f'Test RMSE: {rmse:.2f}')

    # Save the model
    model_path = os.path.join(args.model_dir, 'svm_balloon_detector.joblib')
    joblib.dump(clf, model_path)
    logging.info(f'Model saved to {model_path}')

    logging.info('Experiment completed.\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Balloon Object Detection using HOG and SVM')
    parser.add_argument('--csv', type=str, default='../data/balloon-data.csv',
                        help='Path to the balloon-data.csv file')
    parser.add_argument('--images', type=str, default='../data/images/',
                        help='Directory containing images')
    parser.add_argument('--model_dir', type=str, default='../models/',
                        help='Directory to save the trained model')
    parser.add_argument('--log', type=str, default='../logs/experiment_log.txt',
                        help='Path to the log file')
    parser.add_argument('--C', type=float, default=0.01,
                        help='Regularization parameter for SVM')
    parser.add_argument('--kernel', type=str, default='linear',
                        help='Kernel type for SVM')
    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    main(args)