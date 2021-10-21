from pathlib import Path
import matplotlib.pyplot as plt

from cgmml.models.HRNET.body_pose import BodyPose

THIS_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":

    depthmap_file = THIS_DIR / 'child/b3cad5a4-de28-4a44-ac2f-e06213519457.depth'
    rgb_file = THIS_DIR / 'child/a69d7b60-8342-409a-be2f-ede7e0f4941e.jpg'
    calibration_file = THIS_DIR / 'child/camera_calibration_p30pro_EU.txt'

    body = BodyPose.create_from_rgbd(depthmap_file, rgb_file, calibration_file)
    body.export_object('output_skeleton.obj')
    plt.imshow(body.debug_render())
    plt.show()
