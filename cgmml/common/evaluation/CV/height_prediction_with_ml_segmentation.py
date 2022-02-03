import math

import numpy as np
from PIL import Image

from cgmml.common.depthmap_toolkit.constants import MASK_CHILD
from cgmml.common.depthmap_toolkit.depthmap import Depthmap, is_google_tango_resolution
from cgmml.common.depthmap_toolkit.depthmap_utils import calculate_boundary
from cgmml.common.background_segmentation.deeplab.deeplab_model import get_deeplab_model, render, PERSON_SEGMENTATION

DEEPLAB_MODEL = get_deeplab_model()
HEIGHT_SCALE_FACTOR = 0.05
HEIGHT_OFFSET_IN_CM = -0.25


def predict_height(depthmap_file: str, rgb_file: str, calibration_file: str) -> float:

    # Check if it is captured by a new device
    dmap = Depthmap.create_from_zip_absolute(depthmap_file, 0, calibration_file)
    angle = dmap.get_angle_between_camera_and_floor()
    if is_google_tango_resolution(dmap.width, dmap.height):
        raise Exception('Skipping because it is not a new device data')

    # Run segmentation
    im = Image.open(rgb_file).rotate(-90, expand=True)
    resized_im, seg_map = DEEPLAB_MODEL.run(im)
    seg_map[seg_map != PERSON_SEGMENTATION] = 0

    # Check if the child's head is fully visible
    boundary = calculate_boundary(seg_map)
    if boundary[0] <= 0:
        raise Exception('Skipping because the child\'s head is not fully visible')

    # Upscale depthmap
    floor = dmap.get_floor_level()
    mask = dmap.detect_floor(floor)
    depth = dmap.get_distance_of_child_from_camera(mask)
    dmap.resize(seg_map.shape[0], seg_map.shape[1])
    dmap.depthmap_arr[:, :] = depth

    # Calculate height
    seg_map[seg_map == PERSON_SEGMENTATION] = MASK_CHILD
    highest = dmap.get_highest_point(seg_map)[1]
    factor = 1.0 + math.sin(math.radians(angle)) * HEIGHT_SCALE_FACTOR
    height_in_cm = factor * (highest - floor) * 100.0 + HEIGHT_OFFSET_IN_CM
    return height_in_cm


def render_prediction_plots(depthmap_file: str, rgb_file: str, calibration_file: str) -> np.array:
    return render(DEEPLAB_MODEL, rgb_file, -90)
