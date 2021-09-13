from typing import Tuple

import numpy as np
from PIL import Image

from cgmml.common.depthmap_toolkit.depthmap import Depthmap, is_google_tango_resolution
from cgmml.common.depthmap_toolkit.depthmap_utils import calculate_boundary, vector_length
from cgmml.common.background_segmentation.deeplab.deeplab_model import get_deeplab_model, render, PERSON_SEGMENTATION

DEEPLAB_MODEL = get_deeplab_model()


def predict_height(depthmap_file: str, rgb_file: str, calibration_file: str) -> Tuple[float, float]:

    # Check if it is captured by a new device
    dmap = Depthmap.create_from_zip_absolute(depthmap_file, 0, calibration_file)
    angle = dmap.get_angle_between_camera_and_floor()
    if is_google_tango_resolution(dmap.width, dmap.height):
        raise Exception('Skipping because it is not a new device data')

    # Run segmentation
    im = Image.open(rgb_file).rotate(90, expand=True)
    resized_im, seg_map = DEEPLAB_MODEL.run(im)
    seg_map[seg_map != PERSON_SEGMENTATION] = 0

    # Check if the child's head is fully visible
    boundary = calculate_boundary(seg_map)
    if boundary[0] <= 0 | boundary[2] >= seg_map.shape[0] - 1:
        raise Exception('Skipping because the child is not fully visible')

    # Upscale depthmap
    depth = np.median(dmap.depthmap_arr[dmap.depthmap_arr != 0])
    dmap.resize(seg_map.shape[0], seg_map.shape[1])
    dmap.depthmap_arr[:, :] = depth

    # Get highest and lowest point
    points_3d_arr = dmap.convert_2d_to_3d_oriented()
    x_array = np.copy(points_3d_arr[0, :, :])
    x_array[seg_map != PERSON_SEGMENTATION] = -np.inf
    idx_child_point = np.unravel_index(np.argmax(x_array, axis=None), x_array.shape)
    highest = points_3d_arr[:, idx_child_point[0], idx_child_point[1]]
    x_array[seg_map != PERSON_SEGMENTATION] = np.inf
    idx_child_point = np.unravel_index(np.argmin(x_array, axis=None), x_array.shape)
    lowest = points_3d_arr[:, idx_child_point[0], idx_child_point[1]]

    # Calculate height
    length = vector_length(highest - lowest)
    height_in_cm = length * 100.0
    return height_in_cm, angle


def render_prediction_plots(depthmap_file: str, rgb_file: str, calibration_file: str) -> np.array:
    return render(DEEPLAB_MODEL, rgb_file, 90)
