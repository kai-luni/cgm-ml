import argparse
from joblib import load
import numpy as np

from constants import REPO_DIR
from train_util import get_features_from_fpath
from config_train import CONFIG_TRAIN


DATA_DIR = REPO_DIR / 'data'
MODEL_NAME = '2021q4-points3d-rf-height-28k-200and201'
MODEL_PATH = DATA_DIR / f'models/pose3dpointsModel/{MODEL_NAME}.joblib'
MODEL = None


def load_model():
    return load(MODEL_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-file", help="OBJ file path")

    args = parser.parse_args()

    if not MODEL:
        MODEL = load_model()

    obj_file_path = REPO_DIR / args.obj_file
    assert obj_file_path.exists()
    child_features = get_features_from_fpath(obj_file_path, config_train=CONFIG_TRAIN)
    feats = np.array(list(child_features.values()))

    prediction = MODEL.predict([feats])[0]
    print(prediction)
