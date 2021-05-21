import os

from bunch import Bunch

CONFIG_NAME = os.path.splitext(os.path.basename(__file__))[0]

# Details of model used for evaluation
MODEL_CONFIG = Bunch(dict(
    EXPERIMENT_NAME='q1-ensemble-warmup',
    RUN_IDS=['q1-ensemble-warmup_1620826225_2d4eccea', 'q1-ensemble-warmup_1620888668_aa21c375',
             'q1-ensemble-warmup_1620888698_f28ae559', 'q1-ensemble-warmup_1620888734_5dab77c0',
             'q1-ensemble-warmup_1620888764_734f0aeb', 'q1-ensemble-warmup_1620888793_17bdc5e6',
             'q1-ensemble-warmup_1620888821_b4345da5', 'q1-ensemble-warmup_1620888848_24d582da',
             'q1-ensemble-warmup_1620888873_3b898e53', 'q1-ensemble-warmup_1620888897_c4aa922e',
             'q1-ensemble-warmup_1620888920_d9b18646', 'q1-ensemble-warmup_1620888947_29ac29af',
             'q1-ensemble-warmup_1620888968_5aaa8501', 'q1-ensemble-warmup_1620889003_ca6ffb1a',
             'q1-ensemble-warmup_1620889029_86f10712', 'q1-ensemble-warmup_1620889059_a9b48454'],
    INPUT_LOCATION='outputs',
    NAME='best_model.ckpt',
))


EVAL_CONFIG = Bunch(dict(
    # Name of evaluation
    NAME='q3-depthmap-plaincnn-height-95k-run_03',

    # Experiment in Azure ML which will be used for evaluation
    EXPERIMENT_NAME="QA-pipeline",
    CLUSTER_NAME="gpu-cluster",

    # Used for Debug the QA pipeline
    DEBUG_RUN=False,

    # Will run eval on specified # of scan instead of full dataset
    DEBUG_NUMBER_OF_SCAN=5,

    SPLIT_SEED=0,
))

# Details of Evaluation Dataset
DATA_CONFIG = Bunch(dict(
    NAME='anon-realtime-testdata',  # Name of evaluation dataset

    IMAGE_TARGET_HEIGHT=240,
    IMAGE_TARGET_WIDTH=180,

    BATCH_SIZE=512,  # Batch size for evaluation
    NORMALIZATION_VALUE=7.5,

    TARGET_INDEXES=[0, 5],  # 0 is height, 1 is weight.
    CODES=['100', '101', '102', '200', '201', '202']
))


# Result configuration for result generation after evaluation is done
RESULT_CONFIG = Bunch(dict(
    # Error margin on various ranges
    # EVALUATION_ACCURACIES = [.2, .4, .8, 1.2, 2., 2.5, 3., 4., 5., 6.]
    ACCURACIES=[.2, .4, .6, 1, 1.2, 2., 2.5, 3., 4., 5., 6.],  # 0.2cm, 0.4cm, 0.6cm, 1cm, ...
    ACCURACY_MAIN_THRESH=1.0,
    COLUMNS=['qrcode', 'artifact', 'scantype', 'GT', 'predicted'],

    USE_UNCERTAINTY=True,  # Flag to enable model uncertainty calculation
    NUM_DROPOUT_PREDICTIONS=16,
    DROPOUT_STRENGTH=1,  # 1.0 means like original model
    UNCERTAINTY_THRESHOLD_IN_CM=4.,
    TARGET_INDEXES=[0],
    # path of csv file in the experiment which final result is stored
    SAVE_PATH=f'./outputs/{CONFIG_NAME}',
))
