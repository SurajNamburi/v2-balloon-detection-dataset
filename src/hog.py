import numpy as np
import cv2


def compute_gradients(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=np.float32)

    grad_x = cv2.filter2D(gray.astype(np.float32), -1, sobel_x)
    grad_y = cv2.filter2D(gray.astype(np.float32), -1, sobel_y)

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    direction[direction < 0] += 180

    return magnitude, direction


def compute_cell_histograms(magnitude, direction, cell_size=8, bins=9):
    num_cells_y = magnitude.shape[0] // cell_size
    num_cells_x = magnitude.shape[1] // cell_size
    hist_tensor = np.zeros((num_cells_y, num_cells_x, bins))
    bin_edges = np.linspace(0, 180, bins + 1)

    for i in range(num_cells_y):
        for j in range(num_cells_x):
            cell_mag = magnitude[i * cell_size:(i + 1) * cell_size,
                       j * cell_size:(j + 1) * cell_size]
            cell_dir = direction[i * cell_size:(i + 1) * cell_size,
                       j * cell_size:(j + 1) * cell_size]
            cell_mag = cell_mag.flatten()
            cell_dir = cell_dir.flatten()
            hist, _ = np.histogram(cell_dir, bins=bin_edges, weights=cell_mag)
            hist_tensor[i, j, :] = hist

    return hist_tensor


def normalize_blocks(hist_tensor, block_size=2, epsilon=1e-5):
    num_cells_y, num_cells_x, num_bins = hist_tensor.shape
    blocks_y = num_cells_y - block_size + 1
    blocks_x = num_cells_x - block_size + 1
    normalized_blocks = []

    for i in range(blocks_y):
        for j in range(blocks_x):
            block = hist_tensor[i:i + block_size, j:j + block_size, :].flatten()
            norm = np.linalg.norm(block, ord=2)
            normalized = block / (norm + epsilon)
            normalized = np.minimum(normalized, 0.2)
            normalized = normalized / (np.linalg.norm(normalized, ord=2) + epsilon)
            normalized_blocks.append(normalized)

    return np.array(normalized_blocks)


def extract_hog_features_manual(image, cell_size=8, block_size=2, bins=9):
    magnitude, direction = compute_gradients(image)
    hist_tensor = compute_cell_histograms(magnitude, direction, cell_size, bins)
    normalized_blocks = normalize_blocks(hist_tensor, block_size, epsilon=1e-5)
    feature_vector = normalized_blocks.flatten()
    return feature_vector
