import math
import numpy as np
import logging
import sys
from collections import defaultdict

from cgmml.common.evaluation.CV.csv_utils import read_csv, write_csv

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


def check_height_prediction(height: float):
    """Validates the height result validity and raises an exception if not

    Args:
        height: Height of a child in centimeters
    """

    # Filter invalid values
    if math.isnan(height):
        raise Exception('Skipping because the height result is not a number')

    # Filter heights less than MINIMUM_CHILD_HEIGHT_IN_CM
    if height < MINIMUM_CHILD_HEIGHT_IN_CM:
        raise Exception(f'Skipping because the height is less than {MINIMUM_CHILD_HEIGHT_IN_CM}cm')

    # Filter heights more than MAXIMUM_CHILD_HEIGHT_IN_CM
    if height > MAXIMUM_CHILD_HEIGHT_IN_CM:
        raise Exception(f'Skipping because the height is more than {MAXIMUM_CHILD_HEIGHT_IN_CM}cm')


def filter_metadata(metadata: list, is_standing: bool, one_artifact_per_scan: bool) -> dict:
    """Prepare metadata for processing, group them by scan id and filter them by scan version and type

    Args:
        metadata: Unprocessed metadata loaded from CSV file
        is_standing: True to load only standing children metadata, False to load only lying children metadata
        one_artifact_per_scan: True to return one artifact per scan (faster), False to return all artifacts (slower)

    Returns:
        metadata: dictionary of metadata grouped by scan id
    """

    size = len(metadata)
    output = defaultdict(list)
    for index in range(size):

        # Check if the scan version is correct
        data = metadata[index]
        if not data[METADATA_SCAN_VERSION].startswith('v0.9'):
            continue

        # Check if it is a standing child
        if is_standing and (not data[METADATA_SCAN_TYPE].startswith('10')):
            continue

        # Check if it is a lying child
        if (not is_standing) and (not data[METADATA_SCAN_TYPE].startswith('20')):
            continue

        # Check if it is a first frame of the scan
        if one_artifact_per_scan and (int(data[METADATA_ORDER]) != 1):
            continue

        key = data[METADATA_SCAN_ID]
        output[key].append(data)
    return output


def generate_report(metadata: list, info: str, is_standing: bool) -> list:
    """Create a report from processed metadata extended by information about measure error

    Args:
        metadata: metadata loaded from CSV file extended by information about height, angle and measure error
        info: Footer of the report containg additional information
        is_standing: True to load only standing children metadata, False to load only lying children metadata

    Returns:
        formatted array which can be exported into CSV or written into console
    """

    # Generate report format
    type1 = 'Standing front'
    type2 = 'Standing 360  '
    type3 = 'Standing back '
    if not is_standing:
        type1 = 'Lying front   '
        type2 = 'Lying 360     '
        type3 = 'Lying back    '
    output = [
        ['Scan type     ', 0.2, 0.4, 0.6, 1.0, 1.2, 1.4, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
        [type1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [type2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [type3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    count = ['', 0, 0, 0]

    for row in metadata:

        # Get scan type
        scan_type = int(row[METADATA_SCAN_TYPE])
        line = scan_type % 100 + 1
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


def log_report(data: list):
    """Log into console human readable report

    Args:
        data: formatted report array created by generate_report
    """

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


def run_evaluation(path: str, metadata_file: str, calibration_file: str, method: str, one_artifact_per_scan: bool):
    """Runs evaluation process and save results into CSV files

    Args:
        path: Path where CSV report should be stored
        metadata_file: Path to the CSV file with RAW dataset metadata preprocessed by rgbd_match.py script
        calibration_file: Path to lens calibration file of the device
        method: Method for estimation, available methods are depthmap_toolkit, ml_segmentation, ml_segmentation_lying
        one_artifact_per_scan: True to return one artifact per scan (faster), False to return all artifacts (slower)
    """

    is_standing = True
    if method == 'depthmap_toolkit':
        from height_prediction_depthmap_toolkit import predict_height
    elif method == 'ml_segmentation':
        from height_prediction_with_ml_segmentation import predict_height
    elif method == 'ml_segmentation_lying':
        from height_prediction_with_ml_segmentation_lying import predict_height
        is_standing = False
    else:
        raise Exception('Unimplemented method')

    metadata = filter_metadata(read_csv(metadata_file), is_standing, one_artifact_per_scan)

    sum_err = 0
    output = []
    rejections = []
    keys = metadata.keys()
    for key_index, key in enumerate(keys):
        logger.info('Processing %d/%d', key_index + 1, len(keys))

        angles = []
        heights = []
        last_fail = 0
        for artifact in range(len(metadata[key])):
            data = metadata[key][artifact]

            # Process prediction
            try:
                depthmap_file = (path + data[METADATA_DEPTHMAP]).replace('"', '')
                rgb_file = (path + data[METADATA_RGB]).replace('"', '')
                height, angle = predict_height(depthmap_file, rgb_file, calibration_file)
                check_height_prediction(height)
                heights.append(height)
                angles.append(angle)
            except Exception as exc:
                last_fail = str(exc)
                continue

        info = update_output(angles, heights, last_fail, data, output, rejections, is_standing, sum_err)
        log_report(generate_report(output, info, is_standing))

    write_csv('output.csv', output)
    write_csv('rejections.csv', rejections)
    write_csv('report.csv', generate_report(output, info, is_standing))


def update_output(
        angles: np.array,
        heights: np.array,
        last_fail: str,
        data: list,
        output: list,
        rejections: list,
        is_standing: bool,
        sum_err: float) -> str:
    """Update output about processed and rejected scans and update evaluation error

    Args:
        angles: angles between floor and camera of current scan artifacts
        heights: height in centimeters of current scan artifacts
        last_fail: last reason for rejections of an artifact
        data: metadata of the last processed artifact
        output: array where to add metadata about processed scans
        rejections: array where to add metadata about rejected scans
        is_standing: True to load only standing children metadata, False to load only lying children metadata
        sum_err: sum of errors of height prediction
    Returns:
        formatted string about average error of the evaluation
    """

    if len(heights) == 0:
        logger.info(last_fail)
        data.append(last_fail)
        rejections.append(data)
        return ''
    angle = np.median(angles)
    height = np.median(heights)
    error = abs(height - float(data[METADATA_MANUAL_HEIGHT]))
    logger.info('Height=%fcm, error=%fcm', height, error)
    data.append(height)
    data.append(error)
    data.append(angle)
    output.append(data)
    sum_err += error
    avg_err = sum_err / len(output)
    return f'Average error={avg_err}cm'


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print('You did not enter raw data path, metadata file name or method name')
        print('E.g.: python evaluation.py rawdata_dir metadata_file depthmap_toolkit')
        print('Available methods are depthmap_toolkit, ml_segmentation and ml_segmentation_lying')
        sys.exit(1)

    method = sys.argv[3]
    path = sys.argv[1]
    calibration_file = '../../depthmap_toolkit/camera_calibration_p30pro_EU.txt'
    metadata_file = path + '/' + sys.argv[2]
    one_artifact_per_scan = False

    run_evaluation(path, metadata_file, calibration_file, method, one_artifact_per_scan)
