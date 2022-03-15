import numpy as np

from cgmml.common.depthmap_toolkit.depthmap import Depthmap
from cgmml.common.depthmap_toolkit.visualisation import render_plot_debug


def predict_height(depthmap_file: str, rgb_file: str, calibration_file: str) -> float:

    # Check if the child is fully visible
    dmap = Depthmap.create_from_zip_absolute(depthmap_file, 0, calibration_file)
    floor = dmap.get_floor_level()
    mask = dmap.segment_child(floor)
    if not dmap.is_child_fully_visible(mask):
        raise Exception('Skipping because the child is not fully visible')

    # Calculate height
    highest_point = dmap.get_highest_point(mask)
    height_in_cm = (highest_point[1] - floor) * 100.0
    return height_in_cm


def render_prediction_plots(depthmap_file: str, rgb_file: str, calibration_file: str) -> np.array:
    dmap = Depthmap.create_from_zip_absolute(depthmap_file, rgb_file, calibration_file)
    return render_plot_debug(dmap)
