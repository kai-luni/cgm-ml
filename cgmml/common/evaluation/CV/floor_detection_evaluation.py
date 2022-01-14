import numpy as np
import logging
import sys
from collections import defaultdict

from cgmml.common.depthmap_toolkit.depthmap import Depthmap
from cgmml.common.evaluation.CV.csv_utils import read_csv, write_csv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

METADATA_SCAN_ID = 0
METADATA_DEPTHMAP = 3
METADATA_SCAN_TYPE = 9


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print('You did not enter raw data path or metadata file name')
        print('E.g.: python floor_detection_evaluation.py rawdata_dir metadata_file')
        sys.exit(1)

    # Get paths
    path = sys.argv[1]
    calibration_file = '../../depthmap_toolkit/camera_calibration_p30pro_EU.txt'
    metadata_file = path + '/' + sys.argv[2]

    # Get metadata of standing children
    metadata = defaultdict(list)
    for data in read_csv(metadata_file):
        if data[METADATA_SCAN_TYPE].startswith('10'):
            key = data[METADATA_SCAN_ID]
            metadata[key].append(data)

    # Evaluate floor levels difference between one scan
    sum = 0
    count = 0
    output = []
    keys = metadata.keys()
    for key_index, key in enumerate(keys):
        floors = []
        for artifact in range(len(metadata[key])):
            data = metadata[key][artifact]

            try:
                depthmap_file = (path + data[METADATA_DEPTHMAP]).replace('"', '')
                dmap = Depthmap.create_from_zip_absolute(depthmap_file, 0, calibration_file)
                floors.append(dmap.get_floor_level())
            except Exception:
                continue

        if len(floors) > 1:
            diff = (np.max(floors) - np.min(floors)) * 100
            data.append(diff)
            output.append(data)
            count += 1
            sum += diff
            text = 'Scan %d/%d has floor level diversity %.3fcm, average error is %.3fcm'
            logger.info(text, key_index + 1, len(keys), diff, sum / count)
        else:
            logger.info('Scan %d/%d has no floor level detected', key_index + 1, len(keys))
    write_csv('floor_evaluation.csv', output)
