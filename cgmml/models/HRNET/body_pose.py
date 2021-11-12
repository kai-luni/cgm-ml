import cv2
import json
import numpy as np
from skimage.draw import line_aa

from cgmml.common.depthmap_toolkit.depthmap import Depthmap
from cgmml.common.depthmap_toolkit.exporter import export_obj
from cgmml.common.depthmap_toolkit.visualisation import blur_face, CHILD_HEAD_HEIGHT_IN_METERS
from cgmml.models.HRNET.inference import get_hrnet_model
from cgmml.models.HRNET.hrnet3d import convert_2dskeleton_to_3d, get_person_standing_confidence, get_person_lengths
from cgmml.models.HRNET.hrnet3d import write_skeleton_into_obj, BONES, JOINT_INDEX_LEFT_EYE, JOINT_INDEX_RIGHT_EYE

HRNET_MODEL = get_hrnet_model()
STANDING_CLASSIFY_FACTOR = 0.5


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class BodyPose:
    """Body pose in 3D space

    Instance variables:
        dmap (Depthmap): Depthmap object scaled to the image size
        floor (float): Floor level in oriented 3d space
        rgb (Image): RGB data with the same coordinates as depthmap
        rgb_fpath (str): Path to RGB file (e.g. to the jpg)
        persons_coordinates (dict): Data received from the HRNET model
    """

    def __init__(self, dmap: Depthmap, rgb_fpath: str):
        """Create object from depthmap and rgb file path"""
        self.floor = dmap.get_floor_level()
        self.rgb = cv2.imread(str(rgb_fpath))
        dim = (640, int(self.rgb.shape[0] / self.rgb.shape[1] * 640.0))
        self.rgb = cv2.resize(self.rgb, dim, cv2.INTER_AREA)
        dmap.resize(self.rgb.shape[1], self.rgb.shape[0])

        self.dmap = dmap
        self.rgb_fpath = rgb_fpath

        cache_fpath = f'{rgb_fpath}-hrnet.json'
        try:
            with open(cache_fpath) as json_file:
                self.persons_coordinates = json.load(json_file)
        except OSError:
            self.persons_coordinates = HRNET_MODEL.result_on_artifact_level_from_image(self.rgb, rgb_fpath, '0')
            with open(cache_fpath, 'w') as json_file:
                json.dump(self.persons_coordinates, json_file, cls=NumpyEncoder)

    @classmethod
    def create_from_rgbd(cls,
                         depthmap_fpath: str,
                         rgb_fpath: str,
                         calibration_fpath: str) -> 'BodyPose':
        dmap = Depthmap.create_from_zip_absolute(depthmap_fpath, 0, calibration_fpath)
        return cls(dmap, rgb_fpath)

    def debug_render(self) -> np.array:

        # Check if person is detected and get anonymized image
        if self.get_person_count() != 1:
            return self.get_child_image(False, False)
        image = self.get_child_image(True, False)

        # Draw pose estimation
        pose = self.persons_coordinates['pose_result'][0]
        joints = pose['key_points_coordinate']
        for part in BONES:
            for index1, index2 in zip(part[:-1], part[1:]):
                x1 = int(list(joints[index1].values())[0]['x'])
                y1 = self.rgb.shape[0] - int(list(joints[index1].values())[0]['y']) - 1
                x2 = int(list(joints[index2].values())[0]['x'])
                y2 = self.rgb.shape[0] - int(list(joints[index2].values())[0]['y']) - 1
                rr, cc, val = line_aa(x1, y1, x2, y2)
                image[rr, cc, 0] = 1

        # Reorient the image
        if not self.is_standing():
            image = np.flip(np.flip(image, 1), 0)
        return image

    def export_object(self, filepath: str, use_skeleton=True):

        # export 3d model
        if not use_skeleton:
            export_obj(filepath, self.dmap, self.floor, triangulate=True)
            return

        # export 3d skeleton
        joints = self.get_person_joints()
        write_skeleton_into_obj(filepath, joints)

    def get_child_image(self, should_anonymize: bool, should_reorient: bool) -> np.array:
        # Convert Image array to np array
        bgr = np.asarray(self.rgb, dtype=np.float32) / 255
        image = bgr[..., ::-1].copy()
        image = np.rot90(image, -1)

        # Blur face
        if not should_anonymize:
            return image
        eyes_center = self.get_person_joints()[JOINT_INDEX_LEFT_EYE][0:3]
        eyes_center += self.get_person_joints()[JOINT_INDEX_RIGHT_EYE][0:3]
        eyes_center /= 2.0
        eyes_center[1] += self.floor
        image = blur_face(image, eyes_center, self.dmap, CHILD_HEAD_HEIGHT_IN_METERS / 2.0)

        # Reorient the image
        if should_reorient and (not self.is_standing()):
            image = np.flip(np.flip(image, 1), 0)
        return image

    def get_person_count(self) -> int:
        return self.persons_coordinates['no_of_body_pose_detected']

    def get_person_length(self) -> float:
        joints = self.get_person_joints()
        heights = get_person_lengths(joints)
        return max(np.max(heights), 0.000001)

    def get_person_joints(self) -> list:
        assert self.get_person_count() == 1

        joints = []
        confidences = []
        pose = self.persons_coordinates['pose_result'][0]
        for confidence, joint in zip(pose['key_points_prob'], pose['key_points_coordinate']):
            confidence = float(list(confidence.values())[0]['score'])
            confidences.append(confidence)

            x = int(list(joint.values())[0]['x'])
            y = self.rgb.shape[0] - int(list(joint.values())[0]['y']) - 1
            joints.append([x, y])

        return convert_2dskeleton_to_3d(self.dmap, self.floor, joints, confidences)

    def is_standing(self) -> bool:
        joints = self.get_person_joints()
        return get_person_standing_confidence(joints) > STANDING_CLASSIFY_FACTOR
