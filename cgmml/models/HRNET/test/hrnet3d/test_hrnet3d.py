import numpy as np
from pathlib import Path

from cgmml.common.depthmap_toolkit.depthmap import Depthmap
from cgmml.models.HRNET.hrnet3d import convert_2dskeleton_to_3d, get_person_standing_confidence, get_person_lengths

THIS_DIR = Path(__file__).resolve().parent
DEPTHMAP_FPATH = THIS_DIR / 'child/b3cad5a4-de28-4a44-ac2f-e06213519457.depth'
CALIBRATION_FILE = THIS_DIR / 'child/camera_calibration_p30pro_EU.txt'

CONFIDENCES = [0.9013791680335999, 0.9763078689575195, 0.9742285013198853, 0.9315378069877625, 0.9380636215209961,
               0.8982335925102234, 0.9402387738227844, 0.8676837086677551, 0.9122104048728943, 0.8321260809898376,
               1.0177103281021118, 0.7978922724723816, 0.8439000248908997, 0.840006947517395, 0.8910136818885803,
               0.8315653204917908, 0.8750773072242737]
JOINTS = [[163, 219], [149, 234], [149, 219], [149, 267], [154, 190], [221, 287], [231, 171], [313, 330], [327, 157],
          [332, 272], [404, 166], [380, 243], [395, 214], [477, 267], [481, 210], [554, 277], [578, 205]]


def get_joints3d():
    dmap = Depthmap.create_from_zip_absolute(DEPTHMAP_FPATH, 0, CALIBRATION_FILE)
    floor = dmap.get_floor_level()
    dmap.resize(640, 480)
    return convert_2dskeleton_to_3d(dmap, floor, JOINTS, CONFIDENCES)


def test_convert_2dskeleton_to_3d():
    lengths = get_person_lengths(get_joints3d())
    assert abs(lengths[0] - lengths[1]) < 0.25


def test_get_person_standing_confidence():
    confidence = get_person_standing_confidence(get_joints3d())
    assert confidence > 0.75
    assert confidence < 1.25


def test_get_person_lengths():
    lengths = get_person_lengths(get_joints3d())
    expected = np.array([0.900051, 0.899511])
    np.testing.assert_array_almost_equal(expected, lengths)


if __name__ == "__main__":
    test_convert_2dskeleton_to_3d()
    test_get_person_standing_confidence()
    test_get_person_lengths()
