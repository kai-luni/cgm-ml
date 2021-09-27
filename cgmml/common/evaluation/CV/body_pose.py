import numpy as np
from PIL import Image
from skimage.draw import line_aa

from cgmml.common.depthmap_toolkit.depthmap import Depthmap
from cgmml.common.depthmap_toolkit.depthmap_utils import vector_length
from cgmml.common.depthmap_toolkit.exporter import export_obj
from cgmml.common.depthmap_toolkit.visualisation import blur_face, CHILD_HEAD_HEIGHT_IN_METERS
from cgmml.models.HRNET.inference import get_hrnet_model

BONE_INDEX_NOSE = 0
BONES = [[0, 11, 13, 15], [0, 12, 14, 16]]
HRNET_MODEL = get_hrnet_model('inference-config-hrnet.yaml')
MINIMAL_DEPTH = 0.2
STANDING_CLASSIFY_FACTOR = 0.5


class BodyPose:
    """Body pose in 3D space

    Args:
        dmap (Depthmap): Depthmap object scaled to the image size
        floor (float): Floor level in oriented 3d space
        rgb (Image): RGB data with the same coordinates as depthmap
        rgb_fpath (str): Path to RGB file (e.g. to the jpg)
        json (dict): Data received from the HRNET model
    """

    def __init__(self, dmap: object, rgb_fpath: str):
        """Constructor

        Create object from depthmap and rgb file path
        """
        self.floor = dmap.get_floor_level()
        self.rgb = Image.open(rgb_fpath).rotate(-90, expand=True)
        dmap.resize(self.rgb.size[1], self.rgb.size[0])

        self.dmap = dmap
        self.rgb_fpath = rgb_fpath
        self.json = HRNET_MODEL.result_on_artifact_level(rgb_fpath, '0')

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
        pose = self.json['pose_result'][0]
        points = pose['key_points_coordinate']
        for side, part in enumerate(BONES):
            for bone in range(len(part) - 1):
                index1 = part[bone]
                index2 = part[bone + 1]
                x1 = int(list(points[index1].values())[0]['x'])
                y1 = self.rgb.size[0] - int(list(points[index1].values())[0]['y'])
                x2 = int(list(points[index2].values())[0]['x'])
                y2 = self.rgb.size[0] - int(list(points[index2].values())[0]['y'])
                rr, cc, val = line_aa(x1, y1, x2, y2)
                image[rr, cc, 0] = 1

        # Reorient the image
        if not self.is_standing():
            image = np.flip(np.flip(image, 1), 0)
        return image

    def export_object(self, filepath: str, as_model=False):
        if as_model:
            export_obj(filepath, self.dmap, self.floor, triangulate=True)
        else:
            points = self.get_person_points()
            with open(filepath, 'w') as f:
                for point in points:
                    f.write(f'v {point[0]} {point[1]} {point[2]}\n')
                f.write('l 15 17\n')
                f.write('l 14 16\n')
                f.write('l 13 15\n')
                f.write('l 12 14\n')
                f.write('l 12 13\n')
                f.write('l 7 13\n')
                f.write('l 6 12\n')
                f.write('l 6 7\n')
                f.write('l 7 9\n')
                f.write('l 9 11\n')
                f.write('l 6 8\n')
                f.write('l 8 10\n')
                f.write('l 1 6\n')
                f.write('l 1 7\n')
                f.write('l 1 3\n')
                f.write('l 1 2\n')
                f.write('l 2 3\n')
                f.write('l 2 4\n')
                f.write('l 3 5\n')

    def get_child_image(self, anonymize: bool, reorient: bool) -> np.array:

        # Convert Image array to np array
        image = self.rgb.convert("RGB")
        image = np.asarray(image, dtype=np.float32) / 255
        image = image[:, :, :3]

        # Blur face
        if anonymize:
            highest_point = self.get_person_points()[BONE_INDEX_NOSE]
            highest_point[1] += self.floor
            image = blur_face(image, highest_point, self.dmap, CHILD_HEAD_HEIGHT_IN_METERS / 2.0)
        else:
            return image

        # Reorient the image
        if reorient and (not self.is_standing()):
            image = np.flip(np.flip(image, 1), 0)
        return image

    def get_person_count(self) -> int:
        return self.json['no_of_body_pose_detected']

    def get_person_length(self) -> float:
        heights = []
        points = self.get_person_points()
        for side, part in enumerate(BONES):
            height = 0
            for bone in range(len(part) - 1):
                point1 = points[part[bone]]
                point2 = points[part[bone + 1]]
                height += vector_length(point1 - point2)
            heights.append(height)
        return max(np.max(heights), 0.000001)

    def get_person_points(self) -> list:
        assert(self.get_person_count() == 1)
        pose = self.json['pose_result'][0]
        points = pose['key_points_coordinate']
        points_3d_arr = self.dmap.convert_2d_to_3d_oriented(should_smooth=False)

        # Create cache for searching nearest valid point
        width = self.dmap.width
        height = self.dmap.height
        xbig = np.expand_dims(np.array(range(width)), -1).repeat(height, axis=1)
        ybig = np.expand_dims(np.array(range(height)), 0).repeat(width, axis=0)
        xbig[self.dmap.depthmap_arr < MINIMAL_DEPTH] = width * height
        ybig[self.dmap.depthmap_arr < MINIMAL_DEPTH] = width * height

        output = []
        for point in points:
            x = int(list(point.values())[0]['x'])
            y = self.rgb.size[0] - int(list(point.values())[0]['y'])
            if self.dmap.depthmap_arr[x, y] == 0:
                distance = (abs(xbig - x) + abs(ybig - y) + self.dmap.depthmap_arr * 1000).astype(int)
                idx = np.unravel_index(np.argmin(distance, axis=None), [width, height])
                point_3d = points_3d_arr[:, idx[0], idx[1]]
                point_3d[1] -= self.floor
                output.append(point_3d)
            else:
                point_3d = points_3d_arr[:, x, y]
                point_3d[1] -= self.floor
                output.append(point_3d)
        return output

    def is_standing(self) -> bool:
        height = self.get_person_points()[BONE_INDEX_NOSE][1]
        length = self.get_person_length()
        return height / length > STANDING_CLASSIFY_FACTOR
