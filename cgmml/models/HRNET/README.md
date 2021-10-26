# Pose estimation (HRNet)

HRNET2D detects persons in RGB image and calculates simplified skeleton coordinates in pixels.
Our HRNet3D uses depthmap to convert the information into metrical space.

![Figure_1](https://user-images.githubusercontent.com/6472545/137865074-45e22ca0-fa68-4d41-a4a3-97bc79dc205f.png)

## Resources

This code is written with the help of `https://github.com/leoxiaobin/deep-high-resolution-net.pytorch`

## Core functionality
* `inference.py` - generates from RGB image 2D coordinates of body skeleton in pixels (device dependent)
* `hrnet3d.py` - uses depthmap and output from `inference.py` to provide 3D coordinates of body skeleton in meters (device independent)
* `body_pose.py` - integration of HRNet3D functions (3D coordinates export, face blurring, standing/lying clasification)
* `test/hrnet3d/demo.py` - minimalistic code example how to use `body_pose.py`

## Caveats

The projection finds the child's closest point to the camera, so the 3D points are slightly different. Example: For a standing front-facing child, when finding the knee, the 3D knee point will be on the front skin of the knee, not inside the knee. Despite this problem, the approach still yields quite a good approximation.

## AzureML

### List of registered models in AzureML workspace:

1. [pose_hrnet_w48_384x288](https://ml.azure.com/model/pose_hrnet_w48_384x288:1/details?wsid=/subscriptions/9b5bbfae-d5d1-4aae-a2ca-75159c0c887d/resourceGroups/cgm-ml-prod-we-rg/providers/Microsoft.MachineLearningServices/workspaces/cgm-ml-prod-we-azml&tid=3a27c573-ec1a-4734-9cd3-3208af51794b)
2. [pose_hrnet_w32_384x288](https://ml.azure.com/model/pose_hrnet_w32_384x288:1/details?wsid=/subscriptions/9b5bbfae-d5d1-4aae-a2ca-75159c0c887d/resourceGroups/cgm-ml-prod-we-rg/providers/Microsoft.MachineLearningServices/workspaces/cgm-ml-prod-we-azml&tid=3a27c573-ec1a-4734-9cd3-3208af51794b)

### Steps to use/run the pose prediction on sample scans
1. Make sure you have sample scans at `data/path/version_wise_jpg` path
2. Run below script to generate csv file with all the results and time taken to execute the pose prediction.
   `python inference.py`
