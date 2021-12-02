from joblib import load
from typing import Tuple

import numpy as np

from cgmml.common.depthmap_toolkit.depthmap import is_google_tango_resolution
from cgmml.models.HRNET.body_pose import BodyPose
from cgmml.models.HRNET.hrnet3d import JOINT_INDEX_NOSE
from cgmml.models.Pose3dPoints.Pose3dPoints_height.m2021q4_randomforest.src.constants import REPO_DIR
from cgmml.models.Pose3dPoints.Pose3dPoints_height.m2021q4_randomforest.src.train_util import get_features_from_fpath
from cgmml.models.Pose3dPoints.Pose3dPoints_height.m2021q4_randomforest.src.config_train import CONFIG_TRAIN

PREDICTION_OFFSET_IN_CM_STANDING = 11.0
PREDICTION_OFFSET_IN_CM_LYING = 11.25
TEMP_FILE = 'temp.obj'

DATA_DIR = REPO_DIR / 'data'
MODEL_NAME_LYING = '2021q4-points3d-rf-height-28k-200and201'
MODEL_NAME_STANDING = '2021q4-points3d-rf-height-27k-100and101and102'
MODEL_PATH_LYING = DATA_DIR / f'models/pose3dpointsModel/{MODEL_NAME_LYING}.joblib'
MODEL_PATH_STANDING = DATA_DIR / f'models/pose3dpointsModel/{MODEL_NAME_STANDING}.joblib'
MODEL_STANDING = load(MODEL_PATH_STANDING)
MODEL_LYING = load(MODEL_PATH_LYING)


def predict_height_common(depthmap_file: str, rgb_file: str, calibration_file: str, standing: bool) -> object:

    # Check if it is captured by a new device
    body = BodyPose.create_from_rgbd(depthmap_file, rgb_file, calibration_file)
    if is_google_tango_resolution(body.dmap.width, body.dmap.height):
        raise Exception('Skipping because it is not a new device data')

    # Check how many persons were detected
    person_count = body.get_person_count()
    if person_count < 1:
        raise Exception('Skipping because there is no child detected')
    if person_count > 1:
        raise Exception('Skipping because there are more persons detected')

    # Check if child has correct pose
    if (not standing) and body.is_standing():
        raise Exception('Skipping because the child is standing')
    if standing and (not body.is_standing()):
        raise Exception('Skipping because the child is not standing')

    return body


def predict_height_cv_lying(depthmap_file: str, rgb_file: str, calibration_file: str) -> Tuple[float, float]:
    body = predict_height_common(depthmap_file, rgb_file, calibration_file, False)
    angle = body.dmap.get_angle_between_camera_and_floor()
    height_in_cm = body.get_person_length() * 100.0 + PREDICTION_OFFSET_IN_CM_LYING
    return height_in_cm, angle


def predict_height_cv_standing(depthmap_file: str, rgb_file: str, calibration_file: str) -> Tuple[float, float]:
    body = predict_height_common(depthmap_file, rgb_file, calibration_file, True)
    angle = body.dmap.get_angle_between_camera_and_floor()
    nose_height = body.get_person_joints()[JOINT_INDEX_NOSE][1]
    height_in_cm = nose_height * 100.0 + PREDICTION_OFFSET_IN_CM_STANDING
    return height_in_cm, angle


def predict_height_ml(depthmap_file: str, rgb_file: str, calibration_file: str, standing: bool) -> Tuple[float, float]:

    body = predict_height_common(depthmap_file, rgb_file, calibration_file, standing)
    body.export_object(TEMP_FILE)

    if standing:
        model = MODEL_STANDING
    else:
        model = MODEL_LYING

    child_features = get_features_from_fpath(TEMP_FILE, config_train=CONFIG_TRAIN)
    feats = np.array(list(child_features.values()))
    height_in_cm = model.predict([feats])[0]

    angle = body.dmap.get_angle_between_camera_and_floor()
    return height_in_cm, angle


def predict_height_ml_lying(depthmap_file: str, rgb_file: str, calibration_file: str) -> Tuple[float, float]:
    return predict_height_ml(depthmap_file, rgb_file, calibration_file, False)


def predict_height_ml_standing(depthmap_file: str, rgb_file: str, calibration_file: str) -> Tuple[float, float]:
    return predict_height_ml(depthmap_file, rgb_file, calibration_file, True)


def render_prediction_plots(depthmap_file: str, rgb_file: str, calibration_file: str) -> np.array:
    body = BodyPose.create_from_rgbd(depthmap_file, rgb_file, calibration_file)
    return body.debug_render()
