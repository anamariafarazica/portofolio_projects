"""
Robot Controller Module: PID-based Inoculation
Transforms pixel coordinates to robot workspace and executes precision positioning.
"""

import numpy as np
import cv2
import time


# ============================================================================
# COORDINATE TRANSFORMATION
# ============================================================================

def transform_pixel_to_robot(pixel_x, pixel_y, dish_bbox, original_width, original_height):
    """
    Transform pixel coordinates to robot workspace coordinates.
    Uses dish-relative normalization for universal mapping.
    
    Args:
        pixel_x: X coordinate in pixel space
        pixel_y: Y coordinate in pixel space
        dish_bbox: (x, y, w, h) bounding box of detected petri dish
        original_width: Original image width
        original_height: Original image height
    
    Returns:
        tuple: (robot_x, robot_y, robot_z) in meters
    """
    # Simulation constants
    DISH_CENTER_X = 0.113
    DISH_CENTER_Y = 0.087
    DISH_RADIUS = 0.075
    PLATE_Z = 0.095
    
    if dish_bbox is None:
        print("  ⚠ No dish detected, assuming centered position")
        dish_center_x = original_width / 2
        dish_center_y = original_height / 2
        dish_radius_pixels = min(original_width, original_height) * 0.35
    else:
        x, y, w, h = dish_bbox
        dish_center_x = x + w / 2
        dish_center_y = y + h / 2
        dish_radius_pixels = (w + h) / 4
    
    # Normalize pixel coordinates relative to dish center
    rel_x = (pixel_x - dish_center_x) / dish_radius_pixels
    rel_y = (pixel_y - dish_center_y) / dish_radius_pixels
    
    # Map to robot coordinates
    robot_x = DISH_CENTER_X + rel_y * DISH_RADIUS
    robot_y = DISH_CENTER_Y + rel_x * DISH_RADIUS
    robot_z = PLATE_Z
    
    return robot_x, robot_y, robot_z


def adjust_unreachable_position(x, y, z, max_y=0.175):
    """
    Adjust position to nearest reachable point if outside workspace.
    
    Args:
        x, y, z: Target coordinates in meters
        max_y: Maximum reachable Y coordinate
    
    Returns:
        tuple: (adjusted_x, adjusted_y, adjusted_z, was_adjusted)
    """
    adjusted = False
    original_x, original_y, original_z = x, y, z
    
    if y > max_y:
        y = max_y
        adjusted = True
    
    if x < 0.040:
        x = 0.040
        adjusted = True
    elif x > 0.190:
        x = 0.190
        adjusted = True
    
    if z < 0.080:
        z = 0.080
        adjusted = True
    elif z > 0.110:
        z = 0.110
        adjusted = True
    
    if adjusted:
        print(f"  ⚠ Position adjusted:")
        print(f"    X: {original_x:.4f} → {x:.4f}")
        print(f"    Y: {original_y:.4f} → {y:.4f}")
        print(f"    Z: {original_z:.4f} → {z:.4f}")
    
    return x, y, z, adjusted


# ============================================================================
# TEXTURE PREPARATION
# ============================================================================

def to_grayscale(image):
    """Convert image to grayscale if needed."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def segment_petri_dish(image):
    """Segment petri dish using Otsu thresholding and morphological operations."""
    gray_image = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary


def find_petri_dish_contour(binary_image):
    """Find the largest circular contour (petri dish) in binary image."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    image_area = binary_image.shape[0] * binary_image.shape[1]
    contour_area = cv2.contourArea(largest_contour)
    
    if contour_area < 0.1 * image_area:
        return None
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    
    if 0.6 <= aspect_ratio <= 1.4:
        return largest_contour
    return None


def crop_to_contour(image, contour, padding=10):
    """Crop image to contour bounding box with padding, making it square."""
    x, y, w, h = cv2.boundingRect(contour)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2*padding)
    h = min(image.shape[0] - y, h + 2*padding)
    
    size = max(w, h)
    center_x = x + w // 2
    center_y = y + h // 2
    
    new_x = center_x - size // 2
    new_y = center_y - size // 2
    new_x = max(0, min(new_x, image.shape[1] - size))
    new_y = max(0, min(new_y, image.shape[0] - size))
    size = min(size, image.shape[1] - new_x, image.shape[0] - new_y)
    
    cropped_image = image[new_y:new_y+size, new_x:new_x+size]
    return cropped_image


def prepare_plate_texture(image_path, output_path):
    """
    Prepare plant image as texture for simulation.
    Flips image to match simulation coordinate system.
    
    Args:
        image_path: Path to original plant image
        output_path: Path to save processed texture
    
    Returns:
        tuple: (original_width, original_height, dish_bbox)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    original_height, original_width = img.shape[:2]
    
    # Detect dish BEFORE flipping (CV predictions are on original image)
    binary = segment_petri_dish(img)
    contour = find_petri_dish_contour(binary)
    
    if contour is None:
        print("  ⚠ Could not detect petri dish, using full image")
        dish_bbox = None
    else:
        x, y, w, h = cv2.boundingRect(contour)
        dish_bbox = (x, y, w, h)
        print(f"  ✓ Detected dish at ({x}, {y}) with size {w}×{h}")
    
    # Flip for texture (simulation visual matches coordinate system)
    img = cv2.flip(img, 1)
    
    # Crop after flipping
    if contour is not None:
        # Mirror the contour coordinates
        flipped_contour = contour.copy()
        flipped_contour[:, :, 0] = original_width - flipped_contour[:, :, 0]
        cropped_dish = crop_to_contour(img, flipped_contour, padding=10)
    else:
        cropped_dish = img
    
    plate_size = 277
    dish_resized = cv2.resize(cropped_dish, (plate_size, plate_size))
    dish_rgba = cv2.cvtColor(dish_resized, cv2.COLOR_BGR2BGRA)
    
    texture_size = 1100
    background = np.zeros((texture_size, texture_size, 4), dtype=np.uint8)
    plate_start_x = texture_size - plate_size
    plate_start_y = texture_size - plate_size
    background[plate_start_y:texture_size, plate_start_x:texture_size] = dish_rgba
    
    cv2.imwrite(output_path, background)
    
    return original_width, original_height, dish_bbox


# ============================================================================
# ROBOT CONTROL
# ============================================================================

def move_to_target_with_pid(sim, pid_x, pid_y, pid_z, target, tolerance=0.0005, max_iter=2000):
    """
    Move robot to target position using PID control.
    
    Args:
        sim: Simulation instance
        pid_x, pid_y, pid_z: PID controllers for each axis
        target: [x, y, z] target position in meters
        tolerance: Position error tolerance in meters
        max_iter: Maximum iterations
    
    Returns:
        tuple: (success, final_error, iterations)
    """
    dt = 0.01
    
    pid_x.reset()
    pid_y.reset()
    pid_z.reset()
    
    prev_error = float('inf')
    error_history = []
    
    for iteration in range(max_iter):
        obs = sim.run([[0, 0, 0, 0]], num_steps=1)
        current_pos = obs['robotId_1']['robot_position']
        current_x, current_y, current_z = current_pos
        
        error = np.sqrt((target[0] - current_x)**2 + 
                       (target[1] - current_y)**2 + 
                       (target[2] - current_z)**2)
        
        error_history.append(error)
        
        # Reset integral if error increases
        if error > prev_error:
            pid_x.integral = 0
            pid_y.integral = 0
            pid_z.integral = 0
        
        prev_error = error
        
        if error < tolerance:
            print(f"    Final error: {error*1000:.3f}mm")
            print(f"    Min error reached: {min(error_history)*1000:.3f}mm")
            return True, error, iteration
        
        vel_x = np.clip(pid_x.compute(target[0], current_x, dt), -1, 1)
        vel_y = np.clip(pid_y.compute(target[1], current_y, dt), -1, 1)
        vel_z = np.clip(pid_z.compute(target[2], current_z, dt), -1, 1)
        
        sim.run([[vel_x, vel_y, vel_z, 0]], num_steps=1)
    
    return False, error, max_iter


def drop_at_position(sim, drop_duration=1, wait_after=50):
    """
    Release drop and wait for it to fall and contact plate.
    
    Args:
        sim: Simulation instance
        drop_duration: Number of steps to release drop
        wait_after: Number of steps to wait after release
    """
    for _ in range(drop_duration):
        sim.run([[0, 0, 0, 1]], num_steps=1)
    for _ in range(wait_after):
        sim.run([[0, 0, 0, 0]], num_steps=1)
    print("  ✓ Drop completed")


# ============================================================================
# MAIN INTEGRATION PIPELINE
# ============================================================================

def inoculate_root_tips(sim, pid_controllers, root_tips_data, dish_bbox, image_width, image_height):
    """
    Main integration: Process root tips and execute inoculation.
    
    Args:
        sim: Simulation instance
        pid_controllers: tuple of (pid_x, pid_y, pid_z) controllers
        root_tips_data: Dictionary with plant_id -> {'root_tip': (x, y), ...}
        dish_bbox: (x, y, w, h) bounding box of petri dish
        image_width: Original image width
        image_height: Original image height
    
    Returns:
        list: Results for each plant with success status and metrics
    """
    pid_x, pid_y, pid_z = pid_controllers
    results = []
    
    print(f"\nProcessing {len(root_tips_data)} root tips...")
    
    for plant_id, plant_data in root_tips_data.items():
        tip_pixel = plant_data['root_tip']
        
        if tip_pixel is None:
            print(f"\n--- Plant {plant_id} ---")
            print(f"  ⚠ No root tip detected, skipping")
            results.append({
                'plant_id': plant_id,
                'success': False,
                'reason': 'no_root_tip'
            })
            continue
        
        pixel_x, pixel_y = tip_pixel[0], tip_pixel[1]
        
        print(f"\n--- Plant {plant_id} ---")
        print(f"  Root tip (pixels): ({pixel_x}, {pixel_y})")
        
        # Transform coordinates
        x_robot, y_robot, z_robot = transform_pixel_to_robot(
            pixel_x, pixel_y, dish_bbox, image_width, image_height
        )
        print(f"  Target (robot): [{x_robot:.4f}, {y_robot:.4f}, {z_robot:.4f}]")
        
        # Adjust if outside workspace
        x_robot, y_robot, z_robot, was_adjusted = adjust_unreachable_position(
            x_robot, y_robot, z_robot
        )
        target = [x_robot, y_robot, z_robot]
        
        # Move to target
        print(f"  Moving to target...")
        success, final_error, iterations = move_to_target_with_pid(
            sim, pid_x, pid_y, pid_z, target
        )
        
        if success:
            print(f"  ✓ Reached target in {iterations} iterations (error: {final_error:.6f}m)")
            drop_at_position(sim)
            results.append({
                'plant_id': plant_id,
                'success': True,
                'iterations': iterations,
                'error': final_error,
                'adjusted': was_adjusted
            })
        else:
            print(f"  ✗ Failed to reach target (final error: {final_error:.6f}m)")
            results.append({
                'plant_id': plant_id,
                'success': False,
                'iterations': iterations,
                'error': final_error,
                'adjusted': was_adjusted
            })
        
        time.sleep(0.5)
    
    return results


def print_inoculation_summary(results):
    """Print summary statistics of inoculation results."""
    print("\n" + "="*70)
    print("INOCULATION SUMMARY")
    print("="*70)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"  Successful inoculations: {len(successful)}/{len(results)}")
    print(f"  Failed inoculations: {len(failed)}/{len(results)}")
    
    if successful:
        avg_iterations = np.mean([r['iterations'] for r in successful])
        avg_error = np.mean([r['error'] for r in successful])
        print(f"\n  Average iterations: {avg_iterations:.1f}")
        print(f"  Average error: {avg_error*1000:.3f}mm")
        print(f"  Position adjustments: {sum(1 for r in successful if r.get('adjusted', False))}")
    
    if failed:
        print(f"\n  Failed plants:")
        for r in failed:
            reason = r.get('reason', 'tolerance_not_reached')
            print(f"    - Plant {r['plant_id']}: {reason}")
    