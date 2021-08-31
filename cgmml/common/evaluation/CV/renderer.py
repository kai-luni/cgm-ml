import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import sys

from csv_utils import read_csv

METADATA_DEPTHMAP = 3
METADATA_RGB = 4

REPO_DIR = Path(__file__).parents[4]
EXPORT_DIR = REPO_DIR / 'data' / 'render'

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print('You did not enter raw data path, metadata file name or method name')
        print('E.g.: python renderer.py rawdata_dir metadata_path depthmap_toolkit')
        print('Available methods are depthmap_toolkit and ml_segmentation')
        sys.exit(1)

    if sys.argv[3] == 'depthmap_toolkit':
        from height_prediction_depthmap_toolkit import render_prediction_plots
    elif sys.argv[3] == 'ml_segmentation':
        from height_prediction_with_ml_segmentation import render_prediction_plots
    else:
        raise Exception('Unimplemented method')

    calibration_file = '../../depthmap_toolkit/camera_calibration_p30pro_EU.txt'
    path = sys.argv[1]
    metadata_file = sys.argv[2]

    # Load metadata
    indata = read_csv(metadata_file)
    size = len(indata)

    # Re-create export folder
    try:
        shutil.rmtree(EXPORT_DIR)
    except BaseException:
        print('no previous data to delete')
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # For every depthmap show visualisation
    output = []
    processed = {}
    for index in range(size):
        data = indata[index]

        # Get filenames
        depthmap_file = path + data[METADATA_DEPTHMAP]
        depthmap_file = depthmap_file.replace('"', '')
        rgb_file = path + data[METADATA_RGB]
        rgb_file = rgb_file.replace('"', '')

        # Render data
        file = str(EXPORT_DIR) + '/' + str(index + 1) + '.png'
        plt.imsave(file, render_prediction_plots(depthmap_file, rgb_file, calibration_file))
