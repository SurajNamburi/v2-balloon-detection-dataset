# src/object_detection.py

import os
import cv2
import numpy as np
import pandas as pd
import json
import logging
import argparse
import requests
from io import BytesIO
import pickle
from hog import extract_hog_features_manual
from svm import LinearSVM


def setup_logging(log_file):
    """
    Configures the logging settings.
    """
    logging.basicConfig(filename=log_file,
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG)  # Set to DEBUG for detailed logs


def load_data(csv_path, base_image_url):
    """
    Loads data by downloading images from GitHub.

    Parameters:
    - csv_path: Path to the balloon-data.csv file.
    - base_image_url: Base URL to the GitHub raw images directory.

    Returns:
    - data: NumPy array of image data.
    - labels: NumPy array of labels.
    """
    df = pd.read_csv(csv_path)
    data = []
    labels = []

    logging.info(f'Reading CSV file: {csv_path}')
    logging.info(f'Number of rows in CSV: {len(df)}')

    for index, row in df.iterrows():
        fname = row['fname']
        num_balloons = row['num_balloons']
        # Construct the raw GitHub URL for the image
        image_url = f"{base_image_url}/{fname}"

        try:
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an error for bad status codes
            image_bytes = BytesIO(response.content)
            image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                logging.warning(f"Failed to decode image from URL: {image_url}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = row['height'], row['width']
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading image {fname} from {image_url}: {e}")
            continue  # Skip to the next image

        # Parse bbox safely using json.loads
        try:
            bbox_str = row['bbox'].replace("'", '"')  # Replace single quotes with double quotes for valid JSON
            bbox = json.loads(bbox_str)
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


def evaluate_model(y_true, y_pred):
    """
    Evaluates the model using accuracy and RMSE.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - accuracy: Classification accuracy.
    - rmse: Root Mean Squared Error.
    """
    accuracy = np.mean(y_true == y_pred)
    # RMSE is not typically used for classification, but included per project requirements
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return accuracy, rmse


def main(args):
    """
    Main function to execute the object detection pipeline.
    """
    # Setup logging
    setup_logging(args.log)
    logging.info('Starting Object Detection Experiment')

    # Define the base GitHub raw image URL
    base_image_url = "https://raw.githubusercontent.com/SurajNamburi/v2-balloon-detection-dataset/main/data/images"

    # Load data
    logging.info('Loading data...')
    data, labels = load_data(args.csv, base_image_url)
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
    features = []
    for idx, img in enumerate(data):
        try:
            hog_feat = extract_hog_features_manual(img)
            features.append(hog_feat)
        except Exception as e:
            logging.error(f'Error extracting HOG features from sample {idx}: {e}')
            continue  # Skip this sample
    features = np.array(features, dtype=np.float32)
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
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=labels)
    logging.info(f'Training samples: {len(X_train)}, Testing samples: {len(X_test)}')

    # Train SVM
    logging.info('Training SVM classifier...')
    svm = LinearSVM(learning_rate=args.lr, lambda_param=args.lambda_param, n_iters=args.n_iters)
    svm.fit(X_train, y_train)
    logging.info('SVM training completed.')

    # Predict on test data
    logging.info('Predicting on test data...')
    predictions = svm.predict(X_test)

    # Evaluate
    logging.info('Evaluating the model...')
    acc, rmse = evaluate_model(y_test, predictions)
    logging.info(f'Test Accuracy: {acc * 100:.2f}%')
    logging.info(f'Test RMSE: {rmse:.2f}')

    # Save the model
    model_path = os.path.join(args.model_dir, 'linear_svm_balloon_detector.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(svm, f)
    logging.info(f'Model saved to {model_path}')

    logging.info('Experiment completed.\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Balloon Object Detection using Manual HOG and SVM')
    parser.add_argument('--csv', type=str, default='../data/balloon-data.csv',
                        help='Path to the balloon-data.csv file')
    parser.add_argument('--model_dir', type=str, default='../models/',
                        help='Directory to save the trained model')
    parser.add_argument('--log', type=str, default='../logs/experiment_log.txt',
                        help='Path to the log file')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for SVM')
    parser.add_argument('--lambda_param', type=float, default=0.01,
                        help='Regularization parameter for SVM')
    parser.add_argument('--n_iters', type=int, default=1000,
                        help='Number of iterations for SVM training')
    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    main(args)

