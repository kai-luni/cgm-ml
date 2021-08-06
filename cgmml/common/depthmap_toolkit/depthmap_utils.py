from typing import List

import numpy as np

IDENTITY_MATRIX_4D = [1., 0., 0., 0.,
                      0., 1., 0., 0.,
                      0., 0., 1., 0.,
                      0., 0., 0., 1.]


def matrix_calculate(position: List[float], rotation: List[float]) -> List[float]:
    """Calculate a matrix image->world from device position and rotation"""

    output = IDENTITY_MATRIX_4D

    sqw = rotation[3] * rotation[3]
    sqx = rotation[0] * rotation[0]
    sqy = rotation[1] * rotation[1]
    sqz = rotation[2] * rotation[2]

    invs = 1 / (sqx + sqy + sqz + sqw)
    output[0] = (sqx - sqy - sqz + sqw) * invs
    output[5] = (-sqx + sqy - sqz + sqw) * invs
    output[10] = (-sqx - sqy + sqz + sqw) * invs

    tmp1 = rotation[0] * rotation[1]
    tmp2 = rotation[2] * rotation[3]
    output[1] = 2.0 * (tmp1 + tmp2) * invs
    output[4] = 2.0 * (tmp1 - tmp2) * invs

    tmp1 = rotation[0] * rotation[2]
    tmp2 = rotation[1] * rotation[3]
    output[2] = 2.0 * (tmp1 - tmp2) * invs
    output[8] = 2.0 * (tmp1 + tmp2) * invs

    tmp1 = rotation[1] * rotation[2]
    tmp2 = rotation[0] * rotation[3]
    output[6] = 2.0 * (tmp1 + tmp2) * invs
    output[9] = 2.0 * (tmp1 - tmp2) * invs

    output[12] = -position[0]
    output[13] = -position[1]
    output[14] = -position[2]
    return output


def matrix_transform_point(point: np.ndarray, device_pose_arr: np.ndarray) -> np.ndarray:
    """Transformation of point by device pose matrix

    point(np.array of float): 3D point
    device_pose: flattened 4x4 matrix

    Returns:
        3D point(np.array of float)
    """
    point_4d = np.append(point, 1.)
    output = np.matmul(device_pose_arr, point_4d)
    output[0:2] = output[0:2] / abs(output[3])
    return output[0:-1]


def parse_numbers(line: str) -> List[float]:
    """Parse line of numbers

    Args:
        line: Example: "0.6786797 0.90489584 0.49585155 0.5035042"

    Return:
        numbers: [0.6786797, 0.90489584, 0.49585155, 0.5035042]
    """
    return [float(value) for value in line.split(' ')]
