import numpy as np
from typing import Tuple

from cgmml.common.depthmap_toolkit.depthmap import Depthmap, is_google_tango_resolution
from cgmml.common.depthmap_toolkit.visualisation import render_plot


def predict_height(depthmap_file: str, rgb_file: str, calibration_file: str) -> Tuple[float, float]:

    # Check if it is captured by a new device
    dmap = Depthmap.create_from_zip_absolute(depthmap_file, 0, calibration_file)
    angle = dmap.get_angle_between_camera_and_floor()
    if is_google_tango_resolution(dmap.width, dmap.height):
        raise Exception('Skipping because it is not a new device data')

    # Check if the child is fully visible
    floor: float = dmap.get_floor_level()
    mask = dmap.segment_child(floor)
    if not dmap.is_child_fully_visible(mask):
        raise Exception('Skipping because the child is not fully visible')

    # Calculate height
    highest_point = dmap.get_highest_point(mask)
    height_in_cm = (highest_point[1] - floor) * 100.0
    return height_in_cm, angle


def render_prediction_plots(depthmap_file: str, rgb_file: str, calibration_file: str) -> np.array:
    dmap = Depthmap.create_from_zip_absolute(depthmap_file, rgb_file, calibration_file)
    return render_plot(dmap)
