import os

from bunch import Bunch

CONFIG_NAME = os.path.splitext(os.path.basename(__file__))[0]

RESULT_CONFIG = Bunch(dict(
    # Error margin on various ranges
    ACCURACIES=[.2, .4, .6, 1., 1.2, 2., 2.5, 3., 4., 5., 6.],  # 0.2cm, 0.4cm, 0.6cm, 1cm, ...
))


DATA_CONFIG = Bunch(dict(
    NAME='pose3dpoints',  # Name of evaluation dataset

    TARGET_NAMES=['height'],
    CODES=['100', '101', '102', '200', '201', '202']
))
