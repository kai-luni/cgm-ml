import os
import urllib3
import tensorflow.compat.v1 as tf
from cgmml.common.background_segmentation.deeplab.deeplab import DeepLabModel

CHUNK_SIZE = 16 * 1024 * 1024
MODEL_MIRROR = 'http://download.tensorflow.org/models/'
MODEL_FILE = 'deeplabv3_pascal_trainval_2018_01_04.tar.gz'
PERSON_SEGMENTATION = 15


def download_file(url, path):
    stream = urllib3.PoolManager().request('GET', url, preload_content=False)
    with open(path, 'wb') as output:
        while True:
            data = stream.read(CHUNK_SIZE)
            if not data:
                break
            output.write(data)
    stream.release_conn()


def get_deeplab_model():
    if not os.path.isfile(MODEL_FILE):
        download_file(MODEL_MIRROR + MODEL_FILE, MODEL_FILE)
    tf.disable_v2_behavior()
    return DeepLabModel(MODEL_FILE)
