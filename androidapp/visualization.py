"""
Visualization utilities for bar path plotting
"""

import os
import tempfile
from typing import Tuple, Optional

import cv2
import numpy as np


# Constants
CANVAS_SIZE = 800
PADDING = 100
GRID_SPACING = 60
BG_COLOR = (25, 25, 30)
GRID_COLOR = (45, 45, 50)


def create_path_visualization(
    down_phase: Optional[Tuple[np.ndarray, np.ndarray]],
    up_phase: Optional[Tuple[np.ndarray, np.ndarray]],
    video_dimensions: Tuple[int, int],
    output_path: Optional[str] = None
) -> str:
    """
    Create a modern visualization of the averaged bar path
    
    Args:
        down_phase: Tuple of (x_coords, y_coords) for downward movement
        up_phase: Tuple of (x_coords, y_coords) for upward movement
        video_dimensions: (width, height) of the original video
        output_path: Optional path to save the image, otherwise creates temp file
    
    Returns:
        Path to the saved visualization image
    """
    if down_phase is None and up_phase is None:
        return _create_empty_visualization(output_path)
    
    # Collect all points
    all_points = _collect_points(down_phase, up_phase)
    
    if not all_points:
        return _create_empty_visualization(output_path)
    
    # Create canvas
    img = _create_canvas()
    
    # Calculate transformation
    bounds = _get_bounds(all_points)
    transform = _create_transform(bounds)
    
    # Draw path
    if down_phase:
        _draw_phase(img, down_phase, transform, is_down=True)
    if up_phase:
        _draw_phase(img, up_phase, transform, is_down=False)
    
    # Add legend and title
    _add_legend(img)
    _add_title(img)
    
    # Save
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".png")
    
    cv2.imwrite(output_path, img)
    return output_path


def _create_empty_visualization(output_path: Optional[str] = None) -> str:
    """Create an empty visualization placeholder"""
    img = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), BG_COLOR, dtype=np.uint8)
    
    # Add gradient overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (CANVAS_SIZE, CANVAS_SIZE), (45, 45, 55), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    
    # Add text
    cv2.putText(
        img, "No path data available",
        (CANVAS_SIZE // 2 - 150, CANVAS_SIZE // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (180, 180, 190),
        2,
        cv2.LINE_AA
    )
    
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".png")
    
    cv2.imwrite(output_path, img)
    return output_path


def _collect_points(
    down_phase: Optional[Tuple[np.ndarray, np.ndarray]],
    up_phase: Optional[Tuple[np.ndarray, np.ndarray]]
) -> list:
    """Collect all points from phases"""
    all_points = []
    
    if down_phase is not None:
        xs_down, ys_down = down_phase
        all_points.extend(zip(xs_down, ys_down))
    
    if up_phase is not None:
        xs_up, ys_up = up_phase
        all_points.extend(zip(xs_up, ys_up))
    
    return all_points


def _create_canvas() -> np.ndarray:
    """Create the visualization canvas with gradient and grid"""
    img = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), BG_COLOR, dtype=np.uint8)
    
    # Create subtle gradient background
    for y in range(CANVAS_SIZE):
        alpha = y / CANVAS_SIZE
        color = int(25 + alpha * 10)
        img[y, :] = (color, color, color + 5)
    
    # Draw subtle grid
    for i in range(0, CANVAS_SIZE, GRID_SPACING):
        cv2.line(img, (i, 0), (i, CANVAS_SIZE), GRID_COLOR, 1, cv2.LINE_AA)
        cv2.line(img, (0, i), (CANVAS_SIZE, i), GRID_COLOR, 1, cv2.LINE_AA)
    
    return img


def _get_bounds(points: list) -> dict:
    """Get bounding box of points"""
    all_x = [p[0] for p in points]
    all_y = [p[1] for p in points]
    
    return {
        'min_x': min(all_x),
        'max_x': max(all_x),
        'min_y': min(all_y),
        'max_y': max(all_y),
        'width': max(all_x) - min(all_x),
        'height': max(all_y) - min(all_y)
    }


def _create_transform(bounds: dict):
    """Create transformation function for coordinates"""
    available_size = CANVAS_SIZE - 2 * PADDING
    scale = available_size / max(bounds['width'], bounds['height']) if max(bounds['width'], bounds['height']) > 0 else 1
    
    scaled_width = bounds['width'] * scale
    scaled_height = bounds['height'] * scale
    offset_x = (CANVAS_SIZE - scaled_width) / 2
    offset_y = (CANVAS_SIZE - scaled_height) / 2
    
    def transform(x, y):
        norm_x = (x - bounds['min_x']) * scale
        norm_y = (y - bounds['min_y']) * scale
        return int(norm_x + offset_x), int(norm_y + offset_y)
    
    return transform


def _draw_phase(
    img: np.ndarray, 
    phase: Tuple[np.ndarray, np.ndarray], 
    transform, 
    is_down: bool
):
    """Draw a phase (downward or upward) on the canvas"""
    xs, ys = phase
    points = np.array([transform(x, y) for x, y in zip(xs, ys)], dtype=np.int32)
    
    # Draw path with gradient effect
    for i in range(len(points) - 1):
        progress = i / (len(points) - 1)
        
        if is_down:
            # Green gradient for downward
            color = (int(100 + progress * 50), int(255 - progress * 80), 0)
        else:
            # Red/Pink gradient for upward
            color = (int(255 - progress * 80), 0, int(100 + progress * 50))
        
        cv2.line(img, tuple(points[i]), tuple(points[i + 1]), color, 5, cv2.LINE_AA)


def _add_legend(img: np.ndarray):
    """Add legend to the visualization"""
    legend_x, legend_y = 20, 30
    legend_width, legend_height = 200, 100
    
    # Semi-transparent dark card
    overlay = img.copy()
    cv2.rectangle(
        overlay, 
        (legend_x, legend_y), 
        (legend_x + legend_width, legend_y + legend_height),
        (35, 35, 40), 
        -1
    )
    cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)
    
    # Border
    cv2.rectangle(
        img, 
        (legend_x, legend_y), 
        (legend_x + legend_width, legend_y + legend_height),
        (60, 60, 70), 
        2, 
        cv2.LINE_AA
    )
    
    # Legend items
    item_y = legend_y + 30
    
    # Downward
    cv2.circle(img, (legend_x + 20, item_y), 8, (0, 255, 100), -1, cv2.LINE_AA)
    cv2.circle(img, (legend_x + 20, item_y), 10, (100, 255, 150), 2, cv2.LINE_AA)
    cv2.putText(
        img, "Downward Phase", (legend_x + 40, item_y + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA
    )
    
    # Upward
    item_y += 35
    cv2.circle(img, (legend_x + 20, item_y), 8, (150, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (legend_x + 20, item_y), 10, (200, 100, 255), 2, cv2.LINE_AA)
    cv2.putText(
        img, "Upward Phase", (legend_x + 40, item_y + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 255), 1, cv2.LINE_AA
    )


def _add_title(img: np.ndarray):
    """Add title to the visualization"""
    title_y = CANVAS_SIZE - 25
    title = "AVERAGED BAR PATH"
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)[0]
    title_x = (CANVAS_SIZE - text_size[0]) // 2
    
    # Shadow
    cv2.putText(
        img, title, (title_x + 2, title_y + 2),
        cv2.FONT_HERSHEY_DUPLEX, 0.9, (10, 10, 15), 3, cv2.LINE_AA
    )
    
    # Text
    cv2.putText(
        img, title, (title_x, title_y),
        cv2.FONT_HERSHEY_DUPLEX, 0.9, (200, 200, 210), 2, cv2.LINE_AA
    )
