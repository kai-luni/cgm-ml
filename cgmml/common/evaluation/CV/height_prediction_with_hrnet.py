from typing import Tuple

import numpy as np
from PIL import Image
from skimage.draw import line_aa

from cgmml.common.depthmap_toolkit.depthmap import Depthmap, is_google_tango_resolution
from cgmml.common.depthmap_toolkit.depthmap_utils import vector_length
from cgmml.models.HRNET.inference import get_hrnet_model


BONES = [[0, 11, 13, 15], [0, 12, 14, 16]]
HRNET_MODEL = get_hrnet_model('inference-config-hrnet.yaml')
PREDICTION_OFFSET_IN_CM = 12.0


def predict_height(depthmap_file: str, rgb_file: str, calibration_file: str) -> Tuple[float, float]:

    # Check if it is captured by a new device
    dmap = Depthmap.create_from_zip_absolute(depthmap_file, 0, calibration_file)
    angle = dmap.get_angle_between_camera_and_floor()
    if is_google_tango_resolution(dmap.width, dmap.height):
        raise Exception('Skipping because it is not a new device data')

    # Check how many persons were detected
    result = HRNET_MODEL.result_on_artifact_level(rgb_file, '0')
    if result['no_of_body_pose_detected'] < 1:
        raise Exception('Skipping because there is no child detected')
    if result['no_of_body_pose_detected'] > 1:
        raise Exception('Skipping because there are more persons detected')

    # Prepare data
    im = Image.open(rgb_file).rotate(-90, expand=True)
    pose = result['pose_result'][0]
    points = pose['key_points_coordinate']
    dmap.resize(im.size[1], im.size[0])
    points_3d_arr = dmap.convert_2d_to_3d_oriented(should_smooth=False)

    # Calculate distances between bones
    heights = []
    for side, part in enumerate(BONES):
        height = 0
        for bone in range(len(part) - 1):
            index1 = part[bone]
            index2 = part[bone + 1]
            x1 = int(list(points[index1].values())[0]['x'])
            y1 = im.size[0] - int(list(points[index1].values())[0]['y'])
            x2 = int(list(points[index2].values())[0]['x'])
            y2 = im.size[0] - int(list(points[index2].values())[0]['y'])
            if dmap.depthmap_arr[x1, y1] == 0:
                height -= np.inf
            if dmap.depthmap_arr[x2, y2] == 0:
                height -= np.inf
            point1 = points_3d_arr[:, x1, y1]
            point2 = points_3d_arr[:, x2, y2]
            height += vector_length(point1 - point2)
        heights.append(height)

    height_in_cm = np.max(heights) * 100.0 + PREDICTION_OFFSET_IN_CM
    return height_in_cm, angle


def render_prediction_plots(depthmap_file: str, rgb_file: str, calibration_file: str) -> np.array:

    # Get image
    image = Image.open(rgb_file).rotate(90, expand=True)
    size = image.size
    image = image.convert("RGB")
    image = np.asarray(image, dtype=np.float32) / 255
    image = image[:, :, :3]

    # Get pose
    result = HRNET_MODEL.result_on_artifact_level(rgb_file, '0')
    if result['no_of_body_pose_detected'] != 1:
        return image
    pose = result['pose_result'][0]
    points = pose['key_points_coordinate']

    # Draw pose estimation
    for side, part in enumerate(BONES):
        for bone in range(len(part) - 1):
            index1 = part[bone]
            index2 = part[bone + 1]
            x1 = size[1] - int(list(points[index1].values())[0]['x'])
            y1 = int(list(points[index1].values())[0]['y'])
            x2 = size[1] - int(list(points[index2].values())[0]['x'])
            y2 = int(list(points[index2].values())[0]['y'])
            rr, cc, val = line_aa(x1, y1, x2, y2)
            image[rr, cc, 0] = 1

    return image
