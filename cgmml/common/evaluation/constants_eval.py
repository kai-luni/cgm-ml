# Error margin on various ranges
EVALUATION_ACCURACIES = [.2, .4, .6, 1., 1.2, 2., 2.5, 3., 4., 5., 6.]
MODEL_CKPT_FILENAME = "best_model.ckpt"

DAYS_IN_YEAR = 365

HEIGHT_NAME = 'height'
WEIGHT_NAME = 'weight'
MUAC_NAME = 'muac'
AGE_NAME = 'age'
SEX_NAME = 'sex'
GOODBAD_NAME = 'goodbad'

SEX_DICT = {'female': 0., 'male': 1.}
GOODBAD_DICT = {'bad': 0., 'good': 1., 'delete': 2.}

COLUMN_NAME_AGE = 'GT_age'
COLUMN_NAME_SEX = 'GT_sex'
COLUMN_NAME_GOODBAD = 'GT_goodbad'
CODE_TO_SCANTYPE = {
    '100': '_standingfront',
    '101': '_standing360',
    '102': '_standingback',
    '200': '_lyingfront',
    '201': '_lyingrot',
    '202': '_lyingback',
}
