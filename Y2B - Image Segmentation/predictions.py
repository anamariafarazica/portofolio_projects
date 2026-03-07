"""
Vision Module: Root Segmentation Pipeline
Handles U-Net inference for root, shoot, and seed segmentation.
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import cv2
import numpy as np
import keras.backend as K
from keras.models import load_model


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def f1(y_true, y_pred):
    """Custom F1 metric for model loading."""
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return TP / (Positives + K.epsilon())
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return TP / (Pred_Positives + K.epsilon())
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def load_unet_model(model_path):
    """Load trained U-Net model."""
    return load_model(model_path, custom_objects={"f1": f1}, compile=False)


# ============================================================================
# PETRI DISH DETECTION & CROPPING
# ============================================================================

def segment_petri_dish(image):
    """Detect petri dish region using Otsu thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary


def find_petri_dish_contour(binary_image):
    """Find largest valid petri dish contour."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / (binary_image.shape[0] * binary_image.shape[1])
    if area_ratio < 0.1:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    aspect = float(w) / h if h != 0 else 0
    return largest if 0.6 <= aspect <= 1.4 else None


def crop_to_contour(image, contour, padding=20):
    """Crop image to petri dish bounding box with padding."""
    x, y, w, h = cv2.boundingRect(contour)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2*padding)
    h = min(image.shape[0] - y, h + 2*padding)
    size = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    nx, ny = cx - size // 2, cy - size // 2
    nx = max(0, min(nx, image.shape[1] - size))
    ny = max(0, min(ny, image.shape[0] - size))
    size = min(size, image.shape[1] - nx, image.shape[0] - ny)
    return image[ny:ny+size, nx:nx+size], (nx, ny, size, size)


# ============================================================================
# PATCH-BASED INFERENCE
# ============================================================================

def calculate_padded_size(size, patch_size=256):
    """Calculate target size for patch-based processing."""
    return int(np.ceil(size / patch_size) * patch_size)


def get_padding_values(current, target):
    """Calculate symmetric padding values."""
    pad_h = target[0] - current[0]
    pad_w = target[1] - current[1]
    return pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2


def pad_image(image, target_shape):
    """Pad image to target shape."""
    top, bottom, left, right = get_padding_values(image.shape[:2], target_shape)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)


def patchify_image(image, patch_size=256):
    """Split image into patches."""
    h, w = image.shape[:2]
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return patches


def unpatchify_image(patches, original_shape, patch_size=256):
    """Reconstruct image from patches."""
    h, w = original_shape[:2]
    reconstructed = np.zeros((h, w), dtype=patches[0].dtype)
    idx = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            if i + patch_size <= h and j + patch_size <= w:
                reconstructed[i:i+patch_size, j:j+patch_size] = patches[idx]
                idx += 1
    return reconstructed


# ============================================================================
# MAIN PREDICTION PIPELINE
# ============================================================================

def predict_masks(image_path, model, patch_size=256, threshold=0.5):
    """
    Complete prediction pipeline for a single image.
    
    Args:
        image_path: Path to input image
        model: Loaded U-Net model
        patch_size: Size of patches for inference
        threshold: Confidence threshold for predictions
    
    Returns:
        dict: {
            'root_mask': np.array,
            'shoot_mask': np.array,
            'seed_mask': np.array,
            'original': np.array,
            'crop_coords': tuple (x, y, w, h)
        }
    """
    # Load image
    image = cv2.imread(str(image_path), 0)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    original_h, original_w = image.shape[:2]
    
    # Detect and crop petri dish
    image_bgr = cv2.imread(str(image_path))
    binary_image = segment_petri_dish(image_bgr)
    contour = find_petri_dish_contour(binary_image)
    
    if contour is None:
        raise ValueError("No petri dish detected in image")
    
    petri_dish, crop_coords = crop_to_contour(image_bgr, contour, padding=20)
    crop_x, crop_y, crop_w, crop_h = crop_coords
    
    # Pad to patch size
    target_height = calculate_padded_size(petri_dish.shape[0], patch_size)
    target_width = calculate_padded_size(petri_dish.shape[1], patch_size)
    petri_dish_padded = pad_image(petri_dish, (target_height, target_width))
    
    # Patchify and predict
    patches = patchify_image(petri_dish_padded, patch_size=patch_size)
    
    predicted_patches_root = []
    predicted_patches_seed = []
    predicted_patches_shoot = []
    
    for patch in patches:
        patch_input = patch.astype(np.float32) / 255.0
        patch_input = np.expand_dims(patch_input, axis=0)
        prediction = model.predict(patch_input, verbose=0)
        
        predicted_patches_root.append((prediction[0, :, :, 0] > threshold).astype(np.uint8) * 255)
        predicted_patches_seed.append((prediction[0, :, :, 1] > threshold).astype(np.uint8) * 255)
        predicted_patches_shoot.append((prediction[0, :, :, 2] > threshold).astype(np.uint8) * 255)
    
    # Unpatchify
    predicted_mask_root = unpatchify_image(predicted_patches_root, petri_dish_padded.shape, patch_size)
    predicted_mask_seed = unpatchify_image(predicted_patches_seed, petri_dish_padded.shape, patch_size)
    predicted_mask_shoot = unpatchify_image(predicted_patches_shoot, petri_dish_padded.shape, patch_size)
    
    # Remove padding
    padded_h, padded_w = petri_dish_padded.shape[:2]
    cropped_h, cropped_w = petri_dish.shape[:2]
    top, bottom, left, right = get_padding_values((cropped_h, cropped_w), (padded_h, padded_w))
    
    predicted_mask_root_unpadded = predicted_mask_root[top:top+cropped_h, left:left+cropped_w]
    predicted_mask_shoot_unpadded = predicted_mask_shoot[top:top+cropped_h, left:left+cropped_w]
    predicted_mask_seed_unpadded = predicted_mask_seed[top:top+cropped_h, left:left+cropped_w]
    
    # Place back in original coordinates
    predicted_mask_root_corrected = np.zeros((original_h, original_w), dtype=np.uint8)
    predicted_mask_shoot_corrected = np.zeros((original_h, original_w), dtype=np.uint8)
    predicted_mask_seed_corrected = np.zeros((original_h, original_w), dtype=np.uint8)
    
    predicted_mask_root_corrected[crop_y:crop_y+cropped_h, crop_x:crop_x+cropped_w] = predicted_mask_root_unpadded
    predicted_mask_shoot_corrected[crop_y:crop_y+cropped_h, crop_x:crop_x+cropped_w] = predicted_mask_shoot_unpadded
    predicted_mask_seed_corrected[crop_y:crop_y+cropped_h, crop_x:crop_x+cropped_w] = predicted_mask_seed_unpadded
    
    return {
        'root_mask': predicted_mask_root_corrected,
        'shoot_mask': predicted_mask_shoot_corrected,
        'seed_mask': predicted_mask_seed_corrected,
        'original': image,
        'crop_coords': crop_coords
    }