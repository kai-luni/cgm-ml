import os
import shutil

import sys
import logging
import zipfile

import numpy as np

from typing import List
from cgmml.common.depthmap_toolkit.depthmap import parse_calibration
from cgmml.common.depthmap_toolkit.depthmap_utils import parse_numbers

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(pathname)s: line %(lineno)d'))
logger.addHandler(handler)

ENCODING = 'charmap'


def convert_3d_to_2d(intrinsics: list, x: float, y: float, depth: float, width: int, height: int) -> list:
    """Convert point in meters into point in pixels

    Args:
        intrinsics of sensor: Tells if this is ToF sensor or RGB sensor
        x: X-pos in m
        y: Y-pos in m
        depth: distance from sensor to object at (x, y)
        width
        height

    Returns:
        tx, ty, depth
    """
    fx = intrinsics[0] * float(width)
    fy = intrinsics[1] * float(height)
    cx = intrinsics[2] * float(width)
    cy = intrinsics[3] * float(height)
    tx = x * fx / depth + cx
    ty = y * fy / depth + cy
    return [tx, ty, depth]


def parse_pcd(filepath: str) -> List[List[float]]:
    with open(filepath, 'r') as f:
        data = []
        while True:
            line = str(f.readline())
            if line.startswith('DATA'):
                break

        while True:
            line = str(f.readline())
            if not line:
                break
            else:
                values = parse_numbers(line)
                data.append(values)
    return data


def pcd2depth(calibration: np.array, pcd_fpath: str, width: int, height: int) -> np.ndarray:
    # Convert to depthmap
    points = parse_pcd(pcd_fpath)
    output = np.zeros((width, height, 3))
    for p in points:
        v = convert_3d_to_2d(calibration[1], p[0], p[1], p[2], width, height)
        x = int(width - v[0] - 1)
        y = int(height - v[1] - 1)
        if x >= 0 and y >= 0 and x < width and y < height:
            output[x, y, 0] = p[3]
            output[x, y, 2] = p[2]
    return output


def write_depthmap(output_depth_fpath: str, depthmap: np.ndarray, width: int, height: int):
    with open('data', 'wb') as f:
        header_str = str(width) + 'x' + str(height) + '_0.001_255\n'
        f.write(header_str.encode(ENCODING))
        for y in range(height):
            for x in range(width):
                depth = int(depthmap[x, y, 2] * 1000)
                confidence = int(depthmap[x, y, 0] * 255)
                depth_byte = chr(int(depth / 256)).encode(ENCODING)
                depth_byte2 = chr(depth % 256).encode(ENCODING)
                confidence_byte = chr(confidence).encode(ENCODING)
                f.write(depth_byte)
                f.write(depth_byte2)
                f.write(confidence_byte)
    # Zip data
    with zipfile.ZipFile(output_depth_fpath, "w", zipfile.ZIP_DEFLATED) as f:
        f.write('data', 'data')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('You did not enter pcd_dir folder and calibration file path')
        print('E.g.: python convertpcd2depth.py pcd_dir calibration_file')
        sys.exit(1)

    pcd_dir = sys.argv[1]
    calibration_file = sys.argv[2]

    calibration = parse_calibration(calibration_file)

    depth_filenames = []
    for (dirpath, dirnames, filenames) in os.walk(pcd_dir):
        depth_filenames.extend(filenames)
    depth_filenames.sort()
    try:
        shutil.rmtree('output')
    except BaseException:
        print('no previous data to delete')
    os.makedirs('output/depth')

    # works for lenovo
    width = int(240 * 0.75)
    height = int(180 * 0.75)

    for filename in depth_filenames:
        input_filename = f'{pcd_dir}/{filename}'
        output_filename = f'output/depth/{filename}.depth'
        depthmap = pcd2depth(calibration, input_filename, width, height)
        write_depthmap(output_filename, depthmap, width, height)
    logger.info('Data exported into folder output')
