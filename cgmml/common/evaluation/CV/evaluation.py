import numpy as np
import logging
import sys
from typing import Tuple

from csv_utils import read_csv, write_csv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

ERROR_THRESHOLDS = [0.2, 0.4, 0.6, 1.0, 1.2, 1.4, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
MAXIMUM_CHILD_HEIGHT_IN_CM = 130
MINIMUM_CHILD_HEIGHT_IN_CM = 45

METADATA_SCAN_ID = 0
METADATA_ORDER = 1
METADATA_ARTIFACT_ID = 2
METADATA_DEPTHMAP = 3
METADATA_RGB = 4
METADATA_MANUAL_HEIGHT = 5
METADATA_MANUAL_WEIGHT = 6
METADATA_MANUAL_MUAC = 7
METADATA_SCAN_VERSION = 8
METADATA_SCAN_TYPE = 9
METADATA_MANUAL_DATE = 10
METADATA_SCAN_DATE = 11
METADATA_HEIGHT = 12
METADATA_ERROR = 13
METADATA_ANGLE = 14


def filter_metadata(indata: list) -> list:
    size = len(indata)
    output = []
    for index in range(size):

        # Check if the scan version is correct
        data = indata[index]
        if not data[METADATA_SCAN_VERSION].startswith('v0.9'):
            continue

        # Check if it is a standing child
        if not data[METADATA_SCAN_TYPE].startswith('10'):
            continue

        # Check if it is a first frame of the scan
        if int(data[METADATA_ORDER]) != 1:
            continue

        output.append(data)
    return output


def generate_report(indata: list, info: str) -> list:

    # Generate report format
    output = [
        ['Scan type     ', 0.2, 0.4, 0.6, 1.0, 1.2, 1.4, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
        ['Standing front', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ['Standing 360  ', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ['Standing back ', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    count = ['', 0, 0, 0]

    for row in indata:

        # Get scan type
        line = 0
        scan_type = int(row[METADATA_SCAN_TYPE])
        if scan_type == 100:
            line = 1
        if scan_type == 101:
            line = 2
        if scan_type == 102:
            line = 3
        count[line] += 1

        # Update error categories
        error = float(row[METADATA_ERROR])
        for i, error_thresh in enumerate(ERROR_THRESHOLDS, start=1):
            if error <= error_thresh:
                output[line][i] += 1

    # Show result in percents
    for row in range(1, 4):
        for column in range(1, 13):
            output[row][column] = str(100. * output[row][column] / float(max(1, count[row]))) + '%'
    output.append([info])
    return output


def log_rejection(output: list, data: list, reason: str):
    logger.info(reason)
    data.append(reason)
    output.append(data)


def log_report(data: list):
    output = '\n'
    first_row = True
    for row in data:
        first = True
        line = ''
        for value in row:
            if len(line) > 0:
                line = line + ', '
            if first:
                line = line + str(value)
                first = False
            else:
                number = str(value).replace('%', '')
                if len(number) >= 8:
                    number = number[0: 7]
                while len(number) < 8:
                    number = ' ' + number
                if first_row:
                    number = ' ' + number
                else:
                    number = number + '%'
                line = line + number

        output = output + line + '\n'
        first_row = False
    logger.info(output)


def get_path(root: str, data: np.array, metadata: int) -> str:
    path = root + '/' + data[metadata]
    return path.replace('"', '')


def process_height_prediction(depthmap_file: str, rgb_file: str, calibration_file: str) -> Tuple[float, float]:

    # Check if the input was not rejected
    try:
        height, angle = predict_height(depthmap_file, rgb_file, calibration_file)
    except Exception as exc:
        raise Exception(exc)

    # Filter heights less than MINIMUM_CHILD_HEIGHT_IN_CM
    if height < MINIMUM_CHILD_HEIGHT_IN_CM:
        raise Exception(f'Skipping because the height is less than {MINIMUM_CHILD_HEIGHT_IN_CM}cm')

    # Filter heights more than MAXIMUM_CHILD_HEIGHT_IN_CM
    if height > MAXIMUM_CHILD_HEIGHT_IN_CM:
        raise Exception(f'Skipping because the height is more than {MAXIMUM_CHILD_HEIGHT_IN_CM}cm')

    return height, angle


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print('You did not enter raw data path, metadata file name or method name')
        print('E.g.: python evaluation.py rawdata_dir metadata_file depthmap_toolkit')
        print('Available methods are depthmap_toolkit and ml_segmentation')
        sys.exit(1)

    if sys.argv[3] == 'depthmap_toolkit':
        from height_prediction_depthmap_toolkit import predict_height
    elif sys.argv[3] == 'ml_segmentation':
        from height_prediction_with_ml_segmentation import predict_height
    else:
        raise Exception('Unimplemented method')

    calibration_file = '../../depthmap_toolkit/camera_calibration_p30pro_EU.txt'
    path = sys.argv[1]
    metadata_file = path + '/' + sys.argv[2]

    avg_err = 0
    output = []
    rejections = []
    indata = filter_metadata(read_csv(metadata_file))
    size = len(indata)
    for index in range(size):
        logger.info('Processing %d/%d', index + 1, size)
        data = indata[index]

        # Process prediction
        try:
            depthmap_file = get_path(path, data, METADATA_DEPTHMAP)
            rgb_file = get_path(path, data, METADATA_RGB)
            height, angle = process_height_prediction(depthmap_file, rgb_file, calibration_file)
        except Exception as exc:
            log_rejection(rejections, data, exc)
            continue

        # Update output
        error = abs(height - float(indata[index][METADATA_MANUAL_HEIGHT]))
        logger.info('Height=%fcm, error=%fcm', height, error)
        data.append(height)
        data.append(error)
        data.append(angle)
        output.append(data)
        avg_err += error
        info = 'Average error=' + str(avg_err / float(len(output))) + 'cm'
        log_report(generate_report(output, info))

    write_csv('output.csv', output)
    write_csv('rejections.csv', rejections)
    write_csv('report.csv', generate_report(output, info))
