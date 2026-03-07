"""
Instance Segmentation Module: Individual Plant Separation
Uses watershed algorithm to segment individual plant roots.
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize, remove_small_objects, binary_dilation, disk
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from skimage.segmentation import watershed


# ============================================================================
# PREPROCESSING UTILITIES
# ============================================================================

def binarize(mask, thresh=127):
    """Convert mask to binary format."""
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    _, binary = cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)
    return binary


def preprocess_root_mask(root_mask):
    """
    Clean and prepare root mask for segmentation.
    - Morphological closing to connect gaps
    - Remove small objects
    - Remove top 10% of image (noise region)
    """
    root_bin = binarize(root_mask)
    
    # Vertical closing to connect root segments
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    root_closed = cv2.morphologyEx(root_bin, cv2.MORPH_CLOSE, vertical_kernel)
    
    # Remove small noise
    root_denoised = remove_small_objects(root_closed > 0, min_size=50)
    
    # Remove top 10% (typically noisy region)
    img_height = root_denoised.shape[0]
    top_threshold = int(img_height * 0.10)
    root_denoised[:top_threshold, :] = False
    
    return root_denoised


def create_plant_head_mask(seed_mask, shoot_mask, filter_size=1000, y_tolerance=200):
    """
    Create combined plant head mask from seed and shoot predictions.
    Filters out small regions and those far from median Y position.
    """
    seed_bin = binarize(seed_mask)
    shoot_bin = binarize(shoot_mask)
    
    # Combine and dilate
    plant_head = (seed_bin > 0) | (shoot_bin > 0)
    plant_head = binary_dilation(plant_head, disk(25))
    
    # Filter by Y-coordinate clustering
    plant_head_coords = np.where(plant_head > 0)
    if len(plant_head_coords[0]) == 0:
        return plant_head
    
    lbl_heads = label(plant_head)
    heads_filtered = np.zeros_like(plant_head, dtype=bool)
    
    # Find median Y of large heads
    large_heads = [r.centroid[0] for r in regionprops(lbl_heads) if r.area > 5000]
    if large_heads:
        median_y = np.median(large_heads)
    else:
        median_y = np.median(plant_head_coords[0])
    
    y_min = max(0, int(median_y - y_tolerance))
    y_max = int(median_y + y_tolerance)
    
    # Keep only heads within Y range or large heads
    for r in regionprops(lbl_heads):
        cy, cx = r.centroid
        if r.area > filter_size or (y_min <= cy <= y_max):
            heads_filtered[lbl_heads == r.label] = True
    
    return heads_filtered


# ============================================================================
# ROOT-HEAD CONNECTION
# ============================================================================

def find_main_root_under_head(root_mask, head_mask, search_width=80, search_depth=300):
    """
    Find root segments connected to plant heads.
    Uses dilated search to bridge small gaps.
    """
    labeled_heads = label(head_mask)
    
    # Dilate roots to bridge small gaps
    root_dilated = binary_dilation(root_mask, disk(25))
    labeled_roots_dilated = label(root_dilated)
    labeled_roots_original = label(root_mask)
    
    connected_roots = np.zeros_like(root_mask, dtype=bool)
    
    # For each head, search below for connected root
    head_regions = sorted(regionprops(labeled_heads), key=lambda r: r.area, reverse=True)
    
    for head_region in head_regions:
        head_component = (labeled_heads == head_region.label)
        head_coords = np.where(head_component)
        bottom_y = head_coords[0].max()
        center_x = int(np.mean(head_coords[1]))
        
        # Define search region below head
        x_min = max(0, center_x - search_width)
        x_max = min(root_mask.shape[1], center_x + search_width)
        y_min = bottom_y
        y_max = min(root_mask.shape[0], bottom_y + search_depth)
        
        search_region = root_dilated[y_min:y_max, x_min:x_max]
        
        if search_region.sum() == 0:
            continue
        
        # Find topmost root point in search region
        root_coords_in_region = np.where(search_region > 0)
        if len(root_coords_in_region[0]) == 0:
            continue
        
        topmost_y_local = root_coords_in_region[0].min()
        topmost_indices = np.where(root_coords_in_region[0] == topmost_y_local)[0]
        topmost_x_local = int(np.median(root_coords_in_region[1][topmost_indices]))
        
        # Convert to global coordinates
        topmost_y_global = y_min + topmost_y_local
        topmost_x_global = x_min + topmost_x_local
        
        # Get dilated component and map back to original
        root_label_at_point = labeled_roots_dilated[topmost_y_global, topmost_x_global]
        
        if root_label_at_point > 0:
            dilated_component = (labeled_roots_dilated == root_label_at_point)
            
            # Find corresponding original components
            for orig_region in regionprops(labeled_roots_original):
                orig_component = (labeled_roots_original == orig_region.label)
                if np.any(orig_component & dilated_component):
                    connected_roots |= orig_component
    
    return connected_roots


def get_head_y_range(plant_head):
    """Get Y-coordinate range of plant heads for filtering."""
    if plant_head.sum() == 0:
        return None
    
    head_coords = np.where(plant_head > 0)
    lbl_heads = label(plant_head)
    
    # Use large heads to determine median Y
    large_heads = [r.centroid[0] for r in regionprops(lbl_heads) if r.area > 5000]
    if large_heads:
        median_y = np.median(large_heads)
    else:
        median_y = np.median(head_coords[0])
    
    y_tolerance = 200
    y_min = max(0, int(median_y - y_tolerance))
    y_max = int(median_y + y_tolerance)
    
    return (y_min, y_max)


def add_orphan_roots_by_strip(root_connected, root_denoised, plant_head, n_plants, img_width):
    """
    Add isolated root segments in vertical strips that lack plant heads.
    """
    head_y_range = get_head_y_range(plant_head)
    strip_width = img_width // n_plants
    
    for i in range(n_plants):
        x0 = i * strip_width
        x1 = (i + 1) * strip_width if i < n_plants - 1 else img_width
        
        strip_head = plant_head[:, x0:x1]
        strip_root = root_denoised[:, x0:x1]
        
        # If no head but root exists, check if in Y range
        if strip_head.sum() < 50 and strip_root.sum() > 0:
            if head_y_range is not None:
                y_min, y_max = head_y_range
                strip_root_in_range = strip_root[y_min:y_max, :]
                
                if strip_root_in_range.sum() > 50:
                    root_connected[:, x0:x1] |= strip_root
            else:
                if strip_root.sum() > 50:
                    root_connected[:, x0:x1] |= strip_root
    
    return root_connected


def filter_disconnected_components(root_connected, plant_head, head_y_range=None):
    """
    Remove root components far from plant heads.
    """
    if plant_head.sum() == 0:
        return root_connected
    
    head_distance = ndi.distance_transform_edt(~plant_head)
    lbl_connected = label(root_connected)
    
    for region in regionprops(lbl_connected):
        comp_mask = (lbl_connected == region.label)
        min_dist_to_head = head_distance[comp_mask].min()
        
        # Apply stricter filtering outside Y range
        if head_y_range is not None:
            y_min, y_max = head_y_range
            region_coords = np.where(comp_mask)
            region_y_center = np.mean(region_coords[0])
            in_y_range = y_min <= region_y_center <= y_max
            
            if not in_y_range:
                if min_dist_to_head > 200 or (region.area < 200 and min_dist_to_head > 100):
                    root_connected[comp_mask] = False
        else:
            if min_dist_to_head > 200 or (region.area < 200 and min_dist_to_head > 100):
                root_connected[comp_mask] = False
    
    return root_connected


# ============================================================================
# WATERSHED SEGMENTATION
# ============================================================================

def create_watershed_markers(root_mask_dilated, plant_head, n_plants, img_width):
    """
    Create marker seeds for watershed segmentation.
    One marker per vertical strip, prioritizing largest root component.
    """
    strip_width = img_width // n_plants
    markers = np.zeros_like(root_mask_dilated, dtype=np.int32)
    
    for i in range(n_plants):
        x0 = i * strip_width
        x1 = (i + 1) * strip_width if i < n_plants - 1 else img_width
        
        seed_strip = plant_head[:, x0:x1]
        strip_root = root_mask_dilated[:, x0:x1]
        marker_placed = False
        
        # If head exists, use largest root component center
        if seed_strip.sum() > 50 and strip_root.sum() > 0:
            lbl_strip_roots = label(strip_root)
            
            if lbl_strip_roots.max() > 0:
                largest_comp = max(regionprops(lbl_strip_roots), key=lambda r: r.area)
                comp_center_y = int(largest_comp.centroid[0])
                comp_center_x = int(largest_comp.centroid[1]) + x0
                
                markers[comp_center_y, comp_center_x] = i + 1
                marker_placed = True
        
        # Otherwise use topmost root point
        if not marker_placed and strip_root.sum() > 0:
            root_coords = np.where(strip_root > 0)
            topmost_y = root_coords[0].min()
            topmost_indices = np.where(root_coords[0] == topmost_y)[0]
            topmost_x = int(np.median(root_coords[1][topmost_indices])) + x0
            
            markers[topmost_y, topmost_x] = i + 1
    
    # Dilate markers for robustness
    markers_dilated = np.zeros_like(markers)
    for i in range(1, n_plants + 1):
        marker_mask = (markers == i)
        if marker_mask.sum() > 0:
            dilated = binary_dilation(marker_mask, disk(5))
            markers_dilated[dilated] = i
    
    return markers_dilated


def segment_individual_plants(root_mask_cleaned, root_mask_dilated, markers):
    """
    Perform watershed segmentation to separate individual plants.
    Maps dilated segmentation back to cleaned root mask.
    """
    # Watershed on dilated roots
    distance = ndi.distance_transform_edt(root_mask_dilated)
    segmented_dilated = watershed(-distance, markers, mask=root_mask_dilated)
    
    n_plants = markers.max()
    individual_roots = []
    
    # Map back to cleaned roots using local neighborhood voting
    for i in range(1, n_plants + 1):
        root_mask = np.zeros_like(root_mask_cleaned, dtype=bool)
        
        for y, x in zip(*np.where(root_mask_cleaned)):
            # Check 21x21 neighborhood
            y_min, y_max = max(0, y-10), min(root_mask_cleaned.shape[0], y+11)
            x_min, x_max = max(0, x-10), min(root_mask_cleaned.shape[1], x+11)
            neighborhood = segmented_dilated[y_min:y_max, x_min:x_max]
            
            if neighborhood.size > 0:
                labels, counts = np.unique(neighborhood[neighborhood > 0], return_counts=True)
                if len(labels) > 0:
                    most_common = labels[np.argmax(counts)]
                    if most_common == i:
                        root_mask[y, x] = True
        
        # Remove small noise
        root_mask = remove_small_objects(root_mask, min_size=20)
        root_mask_uint8 = (root_mask.astype(np.uint8) * 255)
        individual_roots.append(root_mask_uint8)
    
    return individual_roots


def assign_unassigned_roots(individual_roots, root_cleaned, n_plants, img_width):
    """
    Assign leftover unassigned root pixels to appropriate plants by strip.
    """
    strip_width = img_width // n_plants
    
    for i in range(len(individual_roots)):
        # If plant has very little root material
        if individual_roots[i].sum() < 50000:
            # Find unassigned pixels
            assigned_mask = np.zeros_like(root_cleaned, dtype=bool)
            for mask in individual_roots:
                assigned_mask |= (mask > 0)
            
            unassigned = root_cleaned & ~assigned_mask
            
            # If significant unassigned material exists
            if unassigned.sum() > individual_roots[i].sum() * 1.5:
                x0 = i * strip_width
                x1 = (i + 1) * strip_width if i < n_plants - 1 else img_width
                
                strip_unassigned = unassigned[:, x0:x1]
                
                if strip_unassigned.sum() > 100:
                    new_mask = np.zeros_like(root_cleaned, dtype=bool)
                    new_mask[:, x0:x1] = strip_unassigned
                    individual_roots[i] = (new_mask.astype(np.uint8) * 255)
    
    return individual_roots


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_skeleton_overlay(individual_roots, background_image, colors=None):
    """
    Create colored overlay visualization with skeletonized roots.
    
    Args:
        individual_roots: List of binary masks for each plant
        background_image: Grayscale background image
        colors: List of RGB tuples (default: standard 5 colors)
    
    Returns:
        RGB overlay image
    """
    if colors is None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    
    # Skeletonize each root
    individual_roots_skeleton = []
    for root_mask in individual_roots:
        root_binary = root_mask > 0
        skeleton = skeletonize(root_binary)
        skeleton_thickened = binary_dilation(skeleton, disk(4))
        skeleton_uint8 = (skeleton_thickened.astype(np.uint8) * 255)
        individual_roots_skeleton.append(skeleton_uint8)
    
    # Create overlay
    overlay = cv2.cvtColor(background_image, cv2.COLOR_GRAY2RGB)
    for i, root_skeleton in enumerate(individual_roots_skeleton):
        mask_bool = root_skeleton > 0
        overlay[mask_bool] = colors[i % len(colors)]
    
    return overlay


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def segment_individual_plants_pipeline(root_mask, seed_mask, shoot_mask, n_plants=5):
    """
    Complete instance segmentation pipeline.
    
    Args:
        root_mask: Binary root segmentation mask
        seed_mask: Binary seed segmentation mask
        shoot_mask: Binary shoot segmentation mask
        n_plants: Expected number of plants (default: 5)
    
    Returns:
        list: Individual binary masks for each plant (n_plants masks)
    """
    img_height, img_width = root_mask.shape
    
    # 1. Preprocess masks
    root_denoised = preprocess_root_mask(root_mask)
    plant_head = create_plant_head_mask(seed_mask, shoot_mask)
    
    # 2. Connect roots to heads
    root_connected = find_main_root_under_head(
        root_denoised, plant_head, 
        search_width=80, search_depth=300
    )
    
    # 3. Add orphan roots in strips without heads
    root_connected = add_orphan_roots_by_strip(
        root_connected, root_denoised, plant_head, n_plants, img_width
    )
    
    # 4. Filter disconnected components
    head_y_range = get_head_y_range(plant_head)
    root_connected = filter_disconnected_components(
        root_connected, plant_head, head_y_range
    )
    
    # 5. Fallback if too little connected material
    if root_connected.sum() < 1000:
        root_connected = root_denoised.copy()
    
    # 6. Dilate for watershed
    root_connected_dilated = binary_dilation(root_connected, disk(20))
    
    # 7. Create watershed markers
    markers = create_watershed_markers(
        root_connected_dilated, plant_head, n_plants, img_width
    )
    
    # 8. Watershed segmentation
    individual_roots = segment_individual_plants(
        root_connected, root_connected_dilated, markers
    )
    
    # 9. Assign leftover unassigned roots
    individual_roots = assign_unassigned_roots(
        individual_roots, root_connected, n_plants, img_width
    )
    
    return individual_roots