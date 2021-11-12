import cv2
import os
import sys
from tqdm import tqdm
from pathlib import Path
from shutil import copyfile
import numpy as np

from cgmml.common.evaluation.CV.csv_utils import read_csv
from cgmml.models.HRNET.body_pose import BodyPose

METADATA_ARTIFACT_ID = 2
METADATA_DEPTHMAP = 3
METADATA_RGB = 4
METADATA_SCAN_VERSION = 8
METADATA_SCAN_TYPE = 9


def mkdir(path: str):
    try:
        os.mkdir(path)
    except OSError:
        print('The directory already exists')


def mkdirs(path: str):
    try:
        os.makedirs(Path(path).resolve().parents[0])
    except OSError:
        print('The directory already exists')


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print('You did not enter raw data path, metadata file name')
        print('E.g.: python hrnet_dataset.py rawdata_dir metadata_file')
        sys.exit(1)

    path = sys.argv[1]
    calibration_file = '../../common/depthmap_toolkit/camera_calibration_p30pro_EU.txt'
    metadata_file = path + '/' + sys.argv[2]
    output_path = 'dataset/'
    raw_path = 'raw_dataset/'

    mkdir(output_path)
    mkdir(raw_path)

    metadata = read_csv(metadata_file)
    metadata_filtered = []
    for row in metadata:

        # Check if the scan version is correct
        if not row[METADATA_SCAN_VERSION].startswith('v0.8'):
            metadata_filtered.append(row)

    size = len(metadata_filtered)
    for index in tqdm(range(size)):
        row = metadata_filtered[index]

        # Get data paths
        depthmap_file = (path + row[METADATA_DEPTHMAP]).replace('"', '')
        rgb_file = (path + row[METADATA_RGB]).replace('"', '')
        json_file = (path + row[METADATA_RGB]).replace('"', '') + '-hrnet.json'
        output_file = f'{output_path}/{row[METADATA_ARTIFACT_ID]}.obj'.replace('"', '')
        if os.path.isfile(output_file):
            continue

        try:
            # Process data
            body = BodyPose.create_from_rgbd(depthmap_file, rgb_file, calibration_file)
            if (body.get_person_count() != 1):
                continue
            body.export_object(output_file)

            # Create anonymized raw dataset
            image = body.get_child_image(True, False) * 255
            image = image[..., ::-1].copy()
            image = np.rot90(image, 1)
            json_raw = (raw_path + row[METADATA_RGB]).replace('"', '') + '-hrnet.json'
            depthmap_raw = (raw_path + row[METADATA_DEPTHMAP]).replace('"', '')
            rgb_raw = (raw_path + row[METADATA_RGB]).replace('"', '')
            mkdirs(depthmap_raw)
            mkdirs(rgb_raw)
            cv2.imwrite(rgb_raw + '.jpg', image)
            os.rename(rgb_raw + '.jpg', rgb_raw)
            copyfile(depthmap_file, depthmap_raw)
            copyfile(json_file, json_raw)

        except Exception:
            print(f'Data for {depthmap_file} does not seem to be valid')
