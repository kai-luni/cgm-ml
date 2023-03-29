from utils import (
    create_output_file_name,
    get_sl_score,
    perform_hypothesis
)
import traceback
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# import matplotlib.pyplot as plt

if __name__ == '__main__':

    # params
    # Analysis : age_less_than_2, age_greater_than_2
    # Factors to filter : [(None, None, None), (pose_confidence, < , .8), (no of pose = 0)]
    # Diagnosis : (WFH, (Manual weight, manual height), (scan weight, scan height))

    # Path of data
    parent_path = '../data/'

    age_lt_2 = False
    # output_path = '../data/age_gt_2_v1/'
    # output_path = '../data/age_gt_2_v2/'
    output_path = '../data/age_gt_2_v3/'

    # age_lt_2 = True
    # output_path = '../age_lt_2_v1/'
    # output_path = '../age_lt_2_v2/'
    # output_path = '../age_lt_2_v3/'

    # input_df_path = parent_path +  "input_df.csv"
    input_df_path = parent_path + "scan_with_mm_and_results.csv"

    target_col = ('manual_weight', 'manual_height')
    # pred_col = ('manual_weight', 'scan_height')
    pred_col = ('scan_weight', 'scan_height')

    diagnosis_info = ("WFH", target_col, pred_col)

    filteration_criteria_list = [
        ('pose_score', 'gt', 0.81),
        ('pose_score', 'gt', 0.86),
        ('pose_score', 'gt', 0.91),
        ('pose_score', 'gt', 0.94),
        ('sl_score', 'gt', 0.81),
        ('sl_score', 'gt', 0.86),
        ('sl_score', 'gt', 0.91),
        ('sl_score', 'gt', 0.94),
        ('no_of_person_using_pose', 'eq', 1),
        ('no_of_faces_detected', 'eq', 1),
        (None, None, None)
    ]

    scan_types = ['all', 'front', 'back', '360']

    input_df = pd.read_csv(input_df_path, index_col=0)
    # input_df = input_df[:1000]

    # print("input_df dtypes ", input_df.dtypes)
    input_df['sl_score'] = input_df.progress_apply(lambda x: get_sl_score(x.standing, x.scan_type_id), axis=1)
    # print("input_df dtypes ", input_df.dtypes)
    # print("input_df['sl_score'] ", input_df['sl_score'][:10])

    scan_type_map = {
        'front': [100, 200],
        'back': [101, 201],
        '360': [102, 202],
        'all': [100, 101, 102, 200, 201, 202]
    }

    for scan_type in scan_types:
        input_df = input_df[input_df['scan_type_id'].isin(scan_type_map[scan_type])]
        for filteration_criteria in filteration_criteria_list:
            try:
                print("-------------------------------------------------------------------")
                out_file_name = create_output_file_name(scan_type, age_lt_2, filteration_criteria, target_col, pred_col)
                scan_df_out_path = output_path + out_file_name + '.csv'
                kpi_file_path = output_path + out_file_name + '.json'
                if len(input_df) == 0:
                    print("Not able to perform Hypothesis")
                else:
                    perform_hypothesis(
                        input_df,
                        scan_df_out_path,
                        kpi_file_path,
                        age_lt_2,
                        filteration_criteria,
                        diagnosis_info
                    )
                print("---------------------------------------------------------------------")
            except Exception as e:
                print(e)
                traceback.print_exc()

    # TODO support of front back 360, scan types,
    # TODO support of without filter
