from height_prediction_with_hrnet import predict_height_cv_standing as predict_height_hrnet
from height_prediction_with_ml_segmentation import predict_height as predict_height_segmentation

MAX_DIFFERENCE = 0.5


def predict_height(depthmap_file: str, rgb_file: str, calibration_file: str) -> float:
    height_hrnet = predict_height_hrnet(depthmap_file, rgb_file, calibration_file)
    height_segmentation = predict_height_segmentation(depthmap_file, rgb_file, calibration_file)

    if abs(height_hrnet - height_segmentation) > MAX_DIFFERENCE:
        raise Exception('The height double check didn\'t succeed')

    height = (height_segmentation + height_hrnet) / 2.0
    return height
