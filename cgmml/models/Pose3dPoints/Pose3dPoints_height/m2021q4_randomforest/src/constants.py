from pathlib import Path

import numpy as np

from cgmml.models.HRNET.hrnet3d import (
    JOINT_INDEX_NOSE, JOINT_INDEX_LEFT_EYE, JOINT_INDEX_RIGHT_EYE, JOINT_INDEX_LEFT_EAR, JOINT_INDEX_RIGHT_EAR,
    JOINT_INDEX_LEFT_SHOULDER, JOINT_INDEX_RIGHT_SHOULDER, JOINT_INDEX_LEFT_ELBOW, JOINT_INDEX_RIGHT_ELBOW,
    JOINT_INDEX_LEFT_WRIST, JOINT_INDEX_RIGHT_WRIST, JOINT_INDEX_LEFT_HIP, JOINT_INDEX_RIGHT_HIP,
    JOINT_INDEX_LEFT_KNEE, JOINT_INDEX_RIGHT_KNEE, JOINT_INDEX_LEFT_ANKLE, JOINT_INDEX_RIGHT_ANKLE)

REPO_DIR = Path(__file__).parents[6].absolute()
DATA_DIR_ONLINE_RUN = Path("/tmp/data/")
MODEL_CKPT_FILENAME = "best_model.ckpt"

SKELETON_CGM = np.array([
    [JOINT_INDEX_RIGHT_KNEE, JOINT_INDEX_RIGHT_ANKLE],
    [JOINT_INDEX_LEFT_KNEE, JOINT_INDEX_LEFT_ANKLE],
    [JOINT_INDEX_RIGHT_HIP, JOINT_INDEX_RIGHT_KNEE],
    [JOINT_INDEX_LEFT_HIP, JOINT_INDEX_LEFT_KNEE],
    [JOINT_INDEX_LEFT_HIP, JOINT_INDEX_RIGHT_HIP],
    [JOINT_INDEX_RIGHT_SHOULDER, JOINT_INDEX_RIGHT_HIP],
    [JOINT_INDEX_LEFT_SHOULDER, JOINT_INDEX_LEFT_HIP],
    [JOINT_INDEX_LEFT_SHOULDER, JOINT_INDEX_RIGHT_SHOULDER],
    [JOINT_INDEX_RIGHT_SHOULDER, JOINT_INDEX_RIGHT_ELBOW],
    [JOINT_INDEX_RIGHT_ELBOW, JOINT_INDEX_RIGHT_WRIST],
    [JOINT_INDEX_LEFT_SHOULDER, JOINT_INDEX_LEFT_ELBOW],
    [JOINT_INDEX_LEFT_ELBOW, JOINT_INDEX_LEFT_WRIST],
    [JOINT_INDEX_NOSE, JOINT_INDEX_LEFT_SHOULDER],
    [JOINT_INDEX_NOSE, JOINT_INDEX_RIGHT_SHOULDER],
    [JOINT_INDEX_NOSE, JOINT_INDEX_RIGHT_EYE],
    [JOINT_INDEX_NOSE, JOINT_INDEX_LEFT_EYE],
    [JOINT_INDEX_LEFT_EYE, JOINT_INDEX_RIGHT_EYE],
    [JOINT_INDEX_LEFT_EYE, JOINT_INDEX_LEFT_EAR],
    [JOINT_INDEX_RIGHT_EYE, JOINT_INDEX_RIGHT_EAR],
])

SKELETON_IMPORTANT = np.array([
    [JOINT_INDEX_RIGHT_KNEE, JOINT_INDEX_RIGHT_ANKLE],
    [JOINT_INDEX_LEFT_KNEE, JOINT_INDEX_LEFT_ANKLE],
    [JOINT_INDEX_RIGHT_HIP, JOINT_INDEX_RIGHT_KNEE],
    [JOINT_INDEX_LEFT_HIP, JOINT_INDEX_LEFT_KNEE],
    [JOINT_INDEX_RIGHT_SHOULDER, JOINT_INDEX_RIGHT_HIP],
    [JOINT_INDEX_LEFT_SHOULDER, JOINT_INDEX_LEFT_HIP],
    [JOINT_INDEX_LEFT_SHOULDER, JOINT_INDEX_RIGHT_SHOULDER],
    [JOINT_INDEX_NOSE, JOINT_INDEX_LEFT_SHOULDER],
    [JOINT_INDEX_NOSE, JOINT_INDEX_RIGHT_SHOULDER],
    [JOINT_INDEX_NOSE, JOINT_INDEX_RIGHT_EYE],
    [JOINT_INDEX_NOSE, JOINT_INDEX_LEFT_EYE],
])

MULTIBONE_STRUCTURES = [
    [JOINT_INDEX_RIGHT_SHOULDER, JOINT_INDEX_RIGHT_HIP, JOINT_INDEX_RIGHT_KNEE, JOINT_INDEX_RIGHT_ANKLE],
    [JOINT_INDEX_LEFT_SHOULDER, JOINT_INDEX_LEFT_HIP, JOINT_INDEX_LEFT_KNEE, JOINT_INDEX_LEFT_ANKLE],
]
