from yacs.config import CfgNode as CN


_C = CN()

_C.GPUS = (0,)
_C.WORKERS = 4

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'pose_hrnet'
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.EXTRA = CN(new_allowed=True)


# testing
_C.TEST = CN()

# size of images for each device
# Test Model Epoch
_C.TEST.POST_PROCESS = False


# nms
_C.TEST.MODEL_FILE = ''


def update_config(cfg, args):

    cfg.defrost()
    cfg.merge_from_file(args)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
