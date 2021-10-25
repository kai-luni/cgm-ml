# Background/Child segmentation

## Introduction

In digital image processing and computer vision, image segmentation is the process of partitioning a digital image into multiple segments.
In our case we assign each pixel of the image to a class which is either `background` or `child`.

## Setup

We use a DeepLab pretrained model to do the segmentation.

Download the tar-ed model files from
<https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md>
(e.g. http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz)
and put it into the `cgm-ml/data/models/background_segmentation` directory.

## Usage

Make sure that the image is rotated correctly, this will ensure best segmentation results.

```python
from PIL import Image
from deeplab import DeepLabModel
model = DeepLabModel(path_to_model_file)
im = Image.open(sample_rgb_fpath)
resized_image, segentation_map = MODEL.run(im)
```
