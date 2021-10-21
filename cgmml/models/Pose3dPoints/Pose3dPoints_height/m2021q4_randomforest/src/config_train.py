from bunch import Bunch

from cgmml.models.Pose3dPoints.Pose3dPoints_height.m2021q4_randomforest.src.constants import SKELETON_IMPORTANT

DATASET_MODE_DOWNLOAD = "dataset_mode_download"
DATASET_MODE_MOUNT = "dataset_mode_mount"

CONFIG_TRAIN = Bunch(dict(  # Hyperparameters

    SCAN_TYPES_TO_USE=['200', '201'],  # '202', '101', '102' is empty,

    # bone features
    SKELETON=SKELETON_IMPORTANT,
    USE_3DVECTOR_FEATURES=True,  # needed
    USE_MULTIBONE_SUM=True,  # needed in RF

    # joint features
    USE_CONFIDENCE_FEATURES=False,  # not needed in RF
    USE_VALIDITY_FEATURES=False,  # not needed in RF
))
