import logging
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

from cgmml.common.depthmap_toolkit.depthmap import Depthmap
from cgmml.common.evaluation.CV.csv_utils import read_csv, write_csv
from cgmml.common.background_segmentation.deeplab.deeplab_model import get_deeplab_model, PERSON_SEGMENTATION

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

DEEPLAB_MODEL = get_deeplab_model()

METADATA_SCAN_ID = 0
METADATA_DEPTHMAP = 3
METADATA_RGB = 4
METADATA_SCAN_TYPE = 9

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print('You did not enter raw data path, metadata file name or scan type')
        print('E.g.: python depth_sensor_validity_detection.py rawdata_dir metadata_file 10')
        sys.exit(1)

    # Get paths
    path = sys.argv[1]
    calibration_file = '../../depthmap_toolkit/camera_calibration_p30pro_EU.txt'
    metadata_file = path + '/' + sys.argv[2]

    # Get metadata
    scan_type = sys.argv[3]
    metadata = defaultdict(list)
    for data in read_csv(metadata_file):
        if data[METADATA_SCAN_TYPE].startswith(scan_type):
            key = data[METADATA_SCAN_ID]
            metadata[key].append(data)

    # Evaluate floor levels difference between one scan
    output = []
    keys = metadata.keys()
    for key_index, key in enumerate(keys):
        valid = []
        for artifact in range(len(metadata[key])):
            data = metadata[key][artifact]

            try:
                # Run segmentation
                rgb_file = (path + data[METADATA_RGB]).replace('"', '')
                im = Image.open(rgb_file).rotate(-90, expand=True)
                resized_im, seg_map = DEEPLAB_MODEL.run(im)
                seg_map[seg_map != PERSON_SEGMENTATION] = 0

                # Get upscaled depthmap
                depthmap_file = (path + data[METADATA_DEPTHMAP]).replace('"', '')
                dmap = Depthmap.create_from_zip_absolute(depthmap_file, 0, calibration_file)
                dmap.resize(seg_map.shape[0], seg_map.shape[1])

                # Count validity
                cond1 = seg_map == PERSON_SEGMENTATION
                cond2 = dmap.depthmap_arr > 0.1
                count_child_px = len(dmap.depthmap_arr[cond1])
                count_valid_px = len(dmap.depthmap_arr[cond1 & cond2])
                valid.append(count_valid_px / count_child_px)
            except Exception:
                continue

        value = 0
        if len(valid) > 0:
            value = np.mean(valid)
        data.append(value)
        output.append(data)
        logger.info('Scan %d/%d has depthmap validity %.3fcm', key_index + 1, len(keys), value)
    write_csv('depth_validation.csv', output)
