"""
Primary Root Extraction Module: Root Tip Detection
Extracts longest path from attachment point to root tip.
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize, binary_dilation, disk
from skimage.measure import label, regionprops
from collections import deque


# ============================================================================
# SKELETON PREPROCESSING
# ============================================================================

def clean_and_skeletonize(root_mask, gap_closing_radius=8):
    """
    Clean root mask and extract skeleton.
    Dilates to close small gaps before skeletonization.
    
    Args:
        root_mask: Binary root mask
        gap_closing_radius: Radius for gap closing dilation
    
    Returns:
        Binary skeleton array
    """
    root_binary = root_mask > 127
    root_dilated = binary_dilation(root_binary, disk(gap_closing_radius))
    skeleton = skeletonize(root_dilated)
    return skeleton


# ============================================================================
# ATTACHMENT POINT DETECTION
# ============================================================================

def find_attachment_point(skeleton):
    """
    Find root attachment point (top of main root structure).
    Uses connected component with greatest vertical extent.
    
    Args:
        skeleton: Binary skeleton image
    
    Returns:
        tuple: (x, y) coordinates of attachment point, or None if empty
    """
    skeleton_coords = np.where(skeleton > 0)
    
    if len(skeleton_coords[0]) == 0:
        return None
    
    # Find component with greatest vertical extent
    labeled = label(skeleton)
    
    best_component = None
    max_height = 0
    
    for region in regionprops(labeled):
        min_y, min_x, max_y, max_x = region.bbox
        height = max_y - min_y
        
        if height > max_height:
            max_height = height
            best_component = region.label
    
    if best_component is not None:
        component_mask = labeled == best_component
        component_coords = np.where(component_mask)
        
        # Find topmost point with best vertical continuity
        topmost_y = component_coords[0].min()
        top_candidates = component_coords[1][component_coords[0] == topmost_y]
        
        best_x = int(np.median(top_candidates))
        best_continuity = 0
        
        for x_candidate in top_candidates:
            # Count pixels directly below this point
            below_mask = (component_coords[0] > topmost_y) & \
                        (np.abs(component_coords[1] - x_candidate) < 10)
            continuity = below_mask.sum()
            
            if continuity > best_continuity:
                best_continuity = continuity
                best_x = x_candidate
        
        return (int(best_x), int(topmost_y))
    
    # Fallback: use topmost median point
    topmost_y = skeleton_coords[0].min()
    topmost_x = int(np.median(skeleton_coords[1][skeleton_coords[0] == topmost_y]))
    return (int(topmost_x), int(topmost_y))


# ============================================================================
# LONGEST PATH EXTRACTION
# ============================================================================

def get_neighbors(y, x, shape):
    """Get 8-connected neighbors within image bounds."""
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                neighbors.append((ny, nx))
    return neighbors


def find_longest_path_bfs(skeleton, start_point):
    """
    Find longest path from start point using BFS with distance accumulation.
    
    Args:
        skeleton: Binary skeleton image
        start_point: (x, y) starting coordinates
    
    Returns:
        tuple: (path, length)
            - path: list of (y, x) coordinates from start to farthest point
            - length: accumulated Euclidean distance along path
    """
    if start_point is None:
        return None, None
    
    start_x, start_y = start_point
    
    visited = np.zeros_like(skeleton, dtype=bool)
    distances = np.zeros_like(skeleton, dtype=float)
    parent = {}
    
    queue = deque([(start_y, start_x, 0)])
    visited[start_y, start_x] = True
    parent[(start_y, start_x)] = None
    
    max_dist = 0
    farthest_point = (start_y, start_x)
    
    while queue:
        y, x, dist = queue.popleft()
        distances[y, x] = dist
        
        if dist > max_dist:
            max_dist = dist
            farthest_point = (y, x)
        
        for ny, nx in get_neighbors(y, x, skeleton.shape):
            if skeleton[ny, nx] > 0 and not visited[ny, nx]:
                visited[ny, nx] = True
                step_dist = np.sqrt((ny - y)**2 + (nx - x)**2)
                queue.append((ny, nx, dist + step_dist))
                parent[(ny, nx)] = (y, x)
    
    # Reconstruct path
    path = []
    current = farthest_point
    while current is not None:
        path.append(current)
        current = parent.get(current)
    
    path.reverse()
    
    return path, max_dist


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def extract_primary_root(root_mask, gap_closing_radius=8):
    """
    Complete pipeline to extract primary root from individual plant mask.
    
    Args:
        root_mask: Binary mask of individual plant root
        gap_closing_radius: Radius for gap closing before skeletonization
    
    Returns:
        dict: {
            'skeleton': Binary skeleton array,
            'attachment_point': (x, y) attachment coordinates,
            'root_tip': (x, y) root tip coordinates,
            'path': List of (y, x) coordinates along primary root,
            'length_pixels': Float length of primary root,
            'primary_root_mask': Binary mask of primary root path
        }
        Returns None if root_mask is empty or processing fails
    """
    if root_mask is None or root_mask.sum() == 0:
        return None
    
    # 1. Clean and skeletonize
    skeleton = clean_and_skeletonize(root_mask, gap_closing_radius)
    
    if skeleton.sum() == 0:
        return None
    
    # 2. Find attachment point
    attachment_pt = find_attachment_point(skeleton)
    
    if attachment_pt is None:
        return None
    
    # 3. Find longest path from attachment point
    path, length = find_longest_path_bfs(skeleton, attachment_pt)
    
    if path is None or len(path) == 0:
        return None
    
    # 4. Extract root tip (end of path)
    root_tip = path[-1]
    root_tip_coords = (int(root_tip[1]), int(root_tip[0]))  # (x, y)
    
    # 5. Create primary root mask
    primary_root_mask = np.zeros_like(skeleton, dtype=np.uint8)
    for y, x in path:
        primary_root_mask[y, x] = 255
    
    return {
        'skeleton': (skeleton.astype(np.uint8) * 255),
        'attachment_point': attachment_pt,
        'root_tip': root_tip_coords,
        'path': path,
        'length_pixels': float(length),
        'primary_root_mask': primary_root_mask,
        'num_points': len(path)
    }


def process_multiple_plants(plant_masks, gap_closing_radius=8):
    """
    Process multiple plant masks and extract primary roots.
    
    Args:
        plant_masks: List of binary masks for individual plants
        gap_closing_radius: Radius for gap closing
    
    Returns:
        list: List of dictionaries with primary root data for each plant
              (None for plants that couldn't be processed)
    """
    results = []
    
    for plant_idx, plant_mask in enumerate(plant_masks):
        result = extract_primary_root(plant_mask, gap_closing_radius)
        results.append(result)
    
    return results