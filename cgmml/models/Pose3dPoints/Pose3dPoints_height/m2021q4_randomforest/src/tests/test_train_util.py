from pathlib import Path

import pandas as pd

from cgmml.models.Pose3dPoints.Pose3dPoints_height.m2021q4_randomforest.src.train_util import (
    Joint, get_features_from_fpath, get_artifact_id_from_path, get_path_from_artifact_id,
    get_joints_from_fpath, prepare_df)

from cgmml.models.Pose3dPoints.Pose3dPoints_height.m2021q4_randomforest.src.config_train import CONFIG_TRAIN

THIS_DIR = Path(__file__).resolve().parent
DATASET_DIR = THIS_DIR / 'test_data'
OBJ_FILE_PATHS = list(sorted(DATASET_DIR.glob('*.obj')))


def test_get_features_from_fpath_creates_distvec_features():
    child_features = get_features_from_fpath(OBJ_FILE_PATHS[0], CONFIG_TRAIN)
    assert [f for f in child_features.keys() if f.startswith('distvec_')]


def test_get_artifact_id_from_path():
    assert '0bb5dec2-8d84-11eb-95f0-8bab827b14ec' == get_artifact_id_from_path(OBJ_FILE_PATHS[0])


def test_get_path_from_artifact_id():
    artifact_path = get_path_from_artifact_id('0bb5dec2-8d84-11eb-95f0-8bab827b14ec', DATASET_DIR)
    assert artifact_path == OBJ_FILE_PATHS[0]


def test_get_joints_from_fpath():
    joints = get_joints_from_fpath(OBJ_FILE_PATHS[0])
    assert len(joints) == 17
    assert isinstance(joints[0], Joint)


def test_prepare_df():
    data = {'height': [60.5, 64.0, 61., 79.2],
            'predicted': [60.4, 64.0, 64., 81.3],
            'scantype': ['100', '101', '200', '200']}
    index = ['artifact_id_1', 'artifact_id_2', 'artifact_id_3', 'artifact_id_4']
    df = pd.DataFrame(data, index=index)
    df_grouped = prepare_df(df)
    assert df_grouped.index.names == ['qrcode', 'scantype']
    df_filtered = df_grouped.iloc[df_grouped.index.get_level_values('scantype') == '200']
    assert df_filtered.shape[0] == 2


if __name__ == "__main__":
    test_prepare_df()
