import numpy as np
import logging
import sys
from collections import defaultdict

from cgmml.common.evaluation.CV.csv_utils import read_csv, write_csv
from cgmml.models.HRNET.body_pose import BodyPose
from cgmml.models.HRNET.hrnet3d import get_person_standing_confidence

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

METADATA_SCAN_ID = 0
METADATA_DEPTHMAP = 3
METADATA_RGB = 4
METADATA_SCAN_TYPE = 9


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print('You did not enter raw data path, metadata file name or scan_prefix')
        print('E.g.: python standing_lying_evaluation.py rawdata_dir metadata_file 10')
        sys.exit(1)

    # Get paths
    path = sys.argv[1]
    calibration_file = '../../depthmap_toolkit/camera_calibration_p30pro_EU.txt'
    metadata_file = path + '/' + sys.argv[2]
    scan_prefix = sys.argv[3]

    # Get metadata
    metadata = defaultdict(list)
    for data in read_csv(metadata_file):
        if data[METADATA_SCAN_TYPE].startswith(scan_prefix):
            key = data[METADATA_SCAN_ID]
            metadata[key].append(data)

    # Evaluate standing-lying between one scan
    output = []
    keys = metadata.keys()
    for key_index, key in enumerate(keys):
        confidence = []
        for artifact in range(len(metadata[key])):
            data = metadata[key][artifact]

            try:
                depthmap_file = (path + data[METADATA_DEPTHMAP]).replace('"', '')
                rgb_file = (path + data[METADATA_RGB]).replace('"', '')
                body = BodyPose.create_from_rgbd(depthmap_file, rgb_file, calibration_file)
                joints = body.get_person_joints()
                confidence.append(get_person_standing_confidence(joints))
            except Exception:
                continue

        if len(confidence) > 0:
            value = np.mean(confidence)
            logger.info('Scan %d/%d is %.3f standing', key_index + 1, len(keys), value)
            data.append(value)
            output.append(data)
        else:
            logger.info('Scan %d/%d has no valid data', key_index + 1, len(keys))
    write_csv('standing_lying_evaluation.csv', output)
