import numpy as np

from cgmml.common.depthmap_toolkit.depthmap_utils import vector_length

JOINT_INDEX_NOSE = 0
JOINT_INDEX_LEFT_EYE = 1
JOINT_INDEX_RIGHT_EYE = 2
JOINT_INDEX_LEFT_EAR = 3
JOINT_INDEX_RIGHT_EAR = 4
JOINT_INDEX_LEFT_SHOULDER = 5
JOINT_INDEX_RIGHT_SHOULDER = 6
JOINT_INDEX_LEFT_ELBOW = 7
JOINT_INDEX_RIGHT_ELBOW = 8
JOINT_INDEX_LEFT_WRIST = 9
JOINT_INDEX_RIGHT_WRIST = 10
JOINT_INDEX_LEFT_HIP = 11
JOINT_INDEX_RIGHT_HIP = 12
JOINT_INDEX_LEFT_KNEE = 13
JOINT_INDEX_RIGHT_KNEE = 14
JOINT_INDEX_LEFT_ANKLE = 15
JOINT_INDEX_RIGHT_ANKLE = 16
BONES = [[JOINT_INDEX_LEFT_EYE, JOINT_INDEX_LEFT_HIP, JOINT_INDEX_LEFT_KNEE, JOINT_INDEX_LEFT_ANKLE],
         [JOINT_INDEX_RIGHT_EYE, JOINT_INDEX_RIGHT_HIP, JOINT_INDEX_RIGHT_KNEE, JOINT_INDEX_RIGHT_ANKLE]]

DEPTH_PENALTY_FACTOR = 10
MINIMAL_DEPTH = 0.2


def are_joints_valid(joints: list) -> bool:
    for joint in joints:
        for value in joint:
            if np.isnan(value):
                return False
    return True


def convert_2dskeleton_to_3d(dmap: object, floor: float, joints: list, confidences: list) -> list:

    # Create cache for searching nearest valid point
    width = dmap.width
    height = dmap.height
    xbig = np.expand_dims(np.array(range(width)), -1).repeat(height, axis=1)
    ybig = np.expand_dims(np.array(range(height)), 0).repeat(width, axis=0)
    xbig[dmap.depthmap_arr < MINIMAL_DEPTH] = width * height
    ybig[dmap.depthmap_arr < MINIMAL_DEPTH] = width * height
    points_3d_arr = dmap.convert_2d_to_3d_oriented(should_smooth=False)

    output = []
    for confidence, joint in zip(confidences, joints):
        x = int(joint[0])
        y = int(joint[1])

        # Find the closest 3D-point in array (be careful not to find a bg pixel) to the joint
        distance = (abs(xbig - x) + abs(ybig - y) + dmap.depthmap_arr * DEPTH_PENALTY_FACTOR).astype(int)
        idx = np.unravel_index(np.argmin(distance, axis=None), [width, height])
        point_3d = points_3d_arr[:, idx[0], idx[1]]

        # Distance in depthmap to the HRNET joint in pixels
        distance_in_px = abs(idx[0] - x) + abs(idx[1] - y)

        # Normalize 3D-point relative to the floor
        point_3d[1] -= floor

        # Add confidence of the joint
        point_3d = np.append(point_3d, [confidence, distance_in_px])
        output.append(point_3d)

    if not are_joints_valid(output):
        raise Exception('The depth data seems to be not valid')
    return output


def get_person_standing_confidence(joints: list) -> float:
    eyes_center = (joints[JOINT_INDEX_LEFT_EYE][1] + joints[JOINT_INDEX_RIGHT_EYE][1]) / 2.0
    length = max(np.max(get_person_lengths(joints)), 0.000001)
    return (eyes_center / length)


def get_person_lengths(joints: list) -> list:
    heights = []
    for part in BONES:
        height = 0
        for index1, index2 in zip(part[:-1], part[1:]):
            height += vector_length(joints[index1] - joints[index2])
        heights.append(height)
    return heights


def write_skeleton_into_obj(filepath: str, joints: list):
    with open(filepath, 'w') as f:

        # Write header
        f.write('# joint coordinates: v x y z  # confidence distance_in_px\n')
        f.write('# connection of joints: l index1 index2 (indexing starting from 1)\n')

        # Write coordinates
        for joint in joints:
            f.write(f'v {joint[0]} {joint[1]} {joint[2]}  # {joint[3]} {joint[4]}\n')

        # Write indices of connected joints
        f.write(f'l {JOINT_INDEX_RIGHT_KNEE + 1} {JOINT_INDEX_RIGHT_ANKLE + 1}\n')
        f.write(f'l {JOINT_INDEX_LEFT_KNEE + 1} {JOINT_INDEX_LEFT_ANKLE + 1}\n')
        f.write(f'l {JOINT_INDEX_RIGHT_HIP + 1} {JOINT_INDEX_RIGHT_KNEE + 1}\n')
        f.write(f'l {JOINT_INDEX_LEFT_HIP + 1} {JOINT_INDEX_LEFT_KNEE + 1}\n')
        f.write(f'l {JOINT_INDEX_LEFT_HIP + 1} {JOINT_INDEX_RIGHT_HIP + 1}\n')
        f.write(f'l {JOINT_INDEX_RIGHT_SHOULDER + 1} {JOINT_INDEX_RIGHT_HIP + 1}\n')
        f.write(f'l {JOINT_INDEX_LEFT_SHOULDER + 1} {JOINT_INDEX_LEFT_HIP + 1}\n')
        f.write(f'l {JOINT_INDEX_LEFT_SHOULDER + 1} {JOINT_INDEX_RIGHT_SHOULDER + 1}\n')
        f.write(f'l {JOINT_INDEX_RIGHT_SHOULDER + 1} {JOINT_INDEX_RIGHT_ELBOW + 1}\n')
        f.write(f'l {JOINT_INDEX_RIGHT_ELBOW + 1} {JOINT_INDEX_RIGHT_WRIST + 1}\n')
        f.write(f'l {JOINT_INDEX_LEFT_SHOULDER + 1} {JOINT_INDEX_LEFT_ELBOW + 1}\n')
        f.write(f'l {JOINT_INDEX_LEFT_ELBOW + 1} {JOINT_INDEX_LEFT_WRIST + 1}\n')
        f.write(f'l {JOINT_INDEX_NOSE + 1} {JOINT_INDEX_LEFT_SHOULDER + 1}\n')
        f.write(f'l {JOINT_INDEX_NOSE + 1} {JOINT_INDEX_RIGHT_SHOULDER + 1}\n')
        f.write(f'l {JOINT_INDEX_NOSE + 1} {JOINT_INDEX_RIGHT_EYE + 1}\n')
        f.write(f'l {JOINT_INDEX_NOSE + 1} {JOINT_INDEX_LEFT_EYE + 1}\n')
        f.write(f'l {JOINT_INDEX_LEFT_EYE + 1} {JOINT_INDEX_RIGHT_EYE + 1}\n')
