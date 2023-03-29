from cgmzscore.src.main import z_score_wfh
from sklearn.metrics import confusion_matrix
from pycm import ConfusionMatrix
import traceback
import json
import numpy as np
# from operator import le
# from re import S
import pandas as pd
# from tqdm import tqdm
from tqdm import tqdm, tqdm_pandas
tqdm_pandas(tqdm())


age_is_2 = 730


scan_type_map = {
    'front': [100, 200],
    'back': [101, 201],
    '360': [102, 202],
    'all': [100, 101, 102, 200, 201, 202]
}


def get_zscore_func(diagnosis_type):
    z_score_func = None
    if diagnosis_type == "WFH":
        z_score_func = z_score_wfh
    else:
        print("Diagnosis Not supported")

    return z_score_func


def calculate_zscore_on_row(z_score_func, row, weight_col, height_col):
    # print(row)
    try:
        score = z_score_func(
            weight=str(row[weight_col]),
            age_in_days=str(row['age']), sex=str(row['sex'][0].upper()),
            height=str(row[height_col])
        )
    except Exception as e:
        traceback.print_exc()
        print("Error :", e)
        score = np.NaN
        print("cound not produce z-score from here")

    return score


# def calculate_zscore(df, diagnosis_type, weight_col, height_col):
#     columns = []
#     columns.append("age_in_days")
#     columns.append("sex")
#     columns.append(height_col)
#     columns.append(weight_col)

#     score_array = []
#     if diagnosis_type == "WFH":
#         z_score_func = z_score_wfh
#     else:
#         print("Diagnosis Not supported")
#         return score_array

#     df1 = df[columns]


#     for tupel in tqdm(df1.itertuples(), total = len(df1)):
#         row = tupel[1:] # remove tupel index for aligning columns with tupels
#         iAge = str(row[columns.index("age_in_days")])
#         sSex = str(row[columns.index("sex")][0].upper())
#         wweight = str(row[columns.index(weight_col)])
#         iHeight = str(row[columns.index(height_col)])
#         #iWeight = int(row[columns.index("target_weight")])
#         try:
#             score = z_score_func(weight=wweight, age_in_days=iAge, sex=sSex, height=iHeight)
#             # print('done')
#         except:
#             score = np.NaN
#             print("count not produce z-score from here")
#         score_array.append(score)
#     return score_array
#     #z_score_wfh(weight="7.853",age_in_days='16',sex='M',height='73')

def diagnosis(wfl):
    if wfl < -3:
        diagnosis = "SAM"
    elif -3 <= wfl < -2:
        diagnosis = "MAM"
    else:
        diagnosis = "Healthy"
    return diagnosis


def calculate_components(data):
    TN1, FP1, FP2, FN1, TP1, FP3, FN2, FN3, TP2 = data.ravel()
    TP = TP1 + TP2
    FN = FN1 + FN2 + FN3
    TN = TN1
    FP = FP1 + FP2 + FP3

    return TP, FN, TN, FP


def calculate_components_mod(confusion_matrix):
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    print("TP, FN, TN, FP ")
    print(TP, FN, TN, FP)

    return TP, FN, TN, FP


def calculate_components_mod_mod(data):
    print(data.ravel())
    A1, A2, A3, B1, B2, B3, C1, C2, C3 = data.ravel()
    TP = B2 + B3 + C2 + C3
    TN = A1
    FP = A2 + A3
    FN = B1 + C1

    return TP, FN, TN, FP


def calculate_medical_kpi(target_diagnosis, predicted_diagnosis):
    # order of labels matters
    data = confusion_matrix(target_diagnosis, predicted_diagnosis, labels=['Healthy', 'MAM', 'SAM'])
    print(type(data))
    print(data.shape)
    print("confusion_matrix data ", data)
    # TP, FN, TN, FP = calculate_components(data)
    # TP, FN, TN, FP = calculate_components_mod(data)
    TP, FN, TN, FP = calculate_components_mod_mod(data)

    cm = ConfusionMatrix(np.array(target_diagnosis), np.array(predicted_diagnosis), classes=['SAM', 'MAM', 'Healthy'])

    print("overall_stat")
    print(cm.overall_stat)

    # sensitivity, recall, hit rate, or true positive rate (TPR)
    # specificity, selectivity or true negative rate (TNR)
    # precision or positive predictive value (PPV)
    # negative predictive value (NPV)
    # miss rate or false negative rate (FNR)
    # fall-out or false positive rate (FPR)
    # false discovery rate (FDR)
    # false omission rate (FOR)
    # Positive likelihood ratio (LR+)
    # Negative likelihood ratio (LR-)
    # threat score (TS) or critical success index (CSI)
    # F1 score

    sensitivity = float(TP / (TP + FN))
    specificity = float(TN / (TN + FP))

    print("TP ", TP, " TN ", TN, " FP ", FP, " FN ", FN)
    print("Sensitivity ", sensitivity)
    print("Specificity ", specificity)

    medical_kpis = {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": float(TP / (TP + FP)),
        "negative_predictive_value": float(TN / (TN + FN)),
        "miss_rate": float(FN / (TP + FN)),
        "fall_out": float(FP / FP + TN),
        "false_discovery_rate": float(FP / (FP + TP)),
        "false_omission_rate": float(FN / (FN + TN)),
        "positive_likelihood_ratio": sensitivity / (100 - specificity),
        "negative_likelihood_ratio": (100 - sensitivity) / specificity,
        "treat_score": float(TP / (TP + FN + FP)),
        "f1_score": float(TP / (TP + (FP + FN) / 2))
    }

    return TP, FP, TN, FN, medical_kpis


def filter_bad_age(row):
    age = row['age'].split(' ')
    # print("age", age)
    if len(age) == 3 and age[1] == 'days':
        return True
    else:
        # print("age", age)
        return False


def get_scan_type(scan_type_code):

    scan_type = 'front'
    return scan_type


def get_standing_laying(scan_type_code):
    sl = 'standing'
    return sl


def get_scan_row(scan_id, df):
    # TODO check if all  manual_weight and manual_height are same for each row
    # TODO check if age and sex are same for each row

    # print(df_group['age'])
    # print(df_group['sex'])

    scan_height_mean = df['scan_height'].mean()
    scan_weight_mean = df['scan_weight'].mean()

    manual_weight = df['manual_weight'].mean()
    manual_height = df['manual_height'].mean()
    # age = df['age_in_days'].iloc[0]
    age = df['age'].iloc[0]
    scan_type_code = df['scan_type_id'].iloc[0]

    sex = df['sex'].iloc[0]
    scan_row = {
        'scan_id': scan_id,
        'age': age,
        'sex': sex,
        'scan_type': get_scan_type(scan_type_code),
        'sl': get_standing_laying(scan_type_code),
        'manual_height': manual_height,
        'manual_weight': manual_weight,
        'scan_height': scan_height_mean,
        'scan_weight': scan_weight_mean
    }
    return scan_row


def get_mae(df_group):
    scan_height_mean = df_group['scan_height'].mean()
    scan_weight_mean = df_group['scan_weight'].mean()

    manual_height_mean = df_group['manual_height'].mean()
    manual_weight_mean = df_group['manual_weight'].mean()

    mae_height = abs(manual_height_mean - scan_height_mean)
    mae_weight = abs(manual_weight_mean - scan_weight_mean)

    return mae_height, mae_weight


def prepare_scan_df_after_filteration(df, age_lt_2, filter_column, filter_operator, filter_threshold):

    # df = df[df.apply(filter_bad_age, axis=1)]
    # df['age_in_days'] = df['age'].map(lambda x : int(x.split(' ')[0]))
    # Filter dataset by age criteria
    if age_lt_2:
        df = df[df['age'] <= age_is_2]
    else:
        df = df[df['age'] > age_is_2]

    df_scan_grouped = df.groupby(['scan_id'], as_index=False)

    total_scan_across_dataset = len(df_scan_grouped)
    print("total_scan_across_dataset ", total_scan_across_dataset)
    no_of_complete_scan_filtered = 0
    total_artifact_across_dataset = 0
    total_artifact_filtered_across_dataset = 0
    scan_df = pd.DataFrame()

    for scan_id, df_group in tqdm(df_scan_grouped):
        no_of_artifacts_before_filter = len(df_group)
        mae_height_b4, mae_weight_b4 = get_mae(df_group)

        if filter_operator == 'lt':
            df_group = df_group[df_group[filter_column] < filter_threshold]
        elif filter_operator == 'gt':
            df_group = df_group[df_group[filter_column] > filter_threshold]
        elif filter_operator == 'eq':
            df_group = df_group[df_group[filter_column] == filter_threshold]
        elif filter_operator is None:
            # print("filter_operator  is None. No filterator performed")
            pass
        else:
            print("filter_operator ", filter_operator)
            print("filter not compatible. No filterator performed")

        no_of_artifact_after_filter = len(df_group)
        no_of_artifact_filtered = no_of_artifacts_before_filter - no_of_artifact_after_filter
        total_artifact_across_dataset = total_artifact_across_dataset + no_of_artifacts_before_filter
        total_artifact_filtered_across_dataset = total_artifact_filtered_across_dataset + no_of_artifact_filtered

        if no_of_artifact_after_filter == 0:
            print("Complete scan is filtered")
            no_of_complete_scan_filtered = no_of_complete_scan_filtered + 1
        else:
            mae_height_after, mae_weight_after = get_mae(df_group)

            scan_row = get_scan_row(scan_id, df_group)
            scan_row['mae_height_b4'] = mae_height_b4
            scan_row['mae_weight_b4'] = mae_weight_b4
            scan_row['mae_height_after'] = mae_height_after
            scan_row['mae_weight_after'] = mae_weight_after
            scan_row['total_artifacts'] = no_of_artifacts_before_filter
            scan_row['no_of_artifact_filtered'] = no_of_artifact_filtered
            scan_row['no_of_artifact_after_filter'] = no_of_artifact_after_filter

            # print(scan_row)
            # print("--------------------------------")

            scan_df = scan_df.append(scan_row, ignore_index=True)
        # print(final_df)
        # i = i +1
        # if i == 10:
        #     break

    # print(scan_df['manual_weight'])
    print(scan_df.columns)

    print("-----------------------------------")
    total_no_of_artifact_per_scan_filtered = total_artifact_filtered_across_dataset / total_scan_across_dataset

    filteration_kpis = {
        "total_scan_across_dataset": total_scan_across_dataset,
        "no_of_complete_scan_filtered": no_of_complete_scan_filtered,
        "total_artifact_across_dataset": total_artifact_across_dataset,
        "total_artifact_filtered_across_dataset": total_artifact_filtered_across_dataset,
        "total_no_of_artifact_per_scan_filtered": total_no_of_artifact_per_scan_filtered
    }

    return scan_df, filteration_kpis


def prepare_zscore_and_diagnosis(scan_df, diagnosis_info, save_df_path=None):
    diagnosis_type, target_col, pred_col = diagnosis_info

    target_weight_col, target_height_col = target_col
    pred_weight_col, pred_height_col = pred_col

    z_score_func = get_zscore_func(diagnosis_type)

    target, predicted = None, None
    print(scan_df.columns)

    print("target_weight_col ", target_weight_col)
    print("target_height_col ", target_height_col)
    print("pred_weight_col ", pred_weight_col)
    print("pred_height_col ", pred_height_col)

    if z_score_func:
        scan_df['pred_wfl_zscore'] = scan_df.progress_apply(
            lambda row: calculate_zscore_on_row(
                z_score_func, row, pred_weight_col, pred_height_col), axis=1)
        scan_df['target_wfl_zscore'] = scan_df.progress_apply(
            lambda row: calculate_zscore_on_row(
                z_score_func, row, target_weight_col, target_height_col), axis=1)

        scan_df['pred_diagnosis'] = scan_df.progress_apply(lambda x: diagnosis(x['pred_wfl_zscore']), axis=1)
        predicted = scan_df['pred_diagnosis']
        scan_df['target_diagnosis'] = scan_df.progress_apply(lambda x: diagnosis(x['target_wfl_zscore']), axis=1)
        target = scan_df['target_diagnosis']

    if save_df_path:
        scan_df.to_csv(save_df_path, index=False)

    # scan_df['pred_wfl_zscore'] = calculate_zscore(
    #     scan_df, diagnosis_type,
    #     pred_weight_col, pred_height_col
    #     )
    # scan_df['pred_diagnosis'] = scan_df.apply(lambda x: diagnosis(x['pred_wfl_zscore']), axis=1)
    # predicted = scan_df['pred_diagnosis']

    # scan_df['target_wfl_zscore'] = calculate_zscore(
    #     scan_df, diagnosis_type,
    #     target_weight_col, target_height_col
    #     )
    # scan_df['target_diagnosis'] = scan_df.apply(lambda x: diagnosis(x['target_wfl_zscore']), axis=1)
    # target = scan_df['target_diagnosis']

    # if save_df_path:
    #     scan_df.to_csv(save_df_path, index=False)

    return target, predicted


def create_output_file_name(scan_type, age_lt_2, filteration_criteria, target_col, pred_col):
    filter_column, filter_operator, filter_threshold = filteration_criteria
    tgt_wt, tgt_ht = target_col
    pred_wt, pred_ht = pred_col

    tgt_wt = ''.join([word[0]for word in tgt_wt.split('_')])
    tgt_ht = ''.join([word[0]for word in tgt_ht.split('_')])
    pred_wt = ''.join([word[0]for word in pred_wt.split('_')])
    pred_ht = ''.join([word[0]for word in pred_ht.split('_')])

    df_name = '_'.join([
        'scan_df',
        scan_type,
        'age_lt_2' if age_lt_2 else 'age_gt_2',
        (filter_column or 'None'),
        (filter_operator or 'None'),
        str(filter_threshold or 'None'),
        'tgt',
        tgt_wt,
        tgt_ht,
        'pred',
        pred_wt,
        pred_ht
    ])
    print("-------------------------------------------------------------------------------------")
    print("file_name ", df_name)
    print("-------------------------------------------------------------------------------------")
    return df_name


def get_sl_score(standing_model_pred, scan_type_id):
    sl_score = float(str(standing_model_pred)[1:-1])
    if scan_type_id in [200, 201, 202]:
        # print("Laying child")
        # print("standing confidence ", sl_score)
        sl_score = 1.0 - sl_score
        # print("Laying confidence ", sl_score)
    return sl_score


def perform_hypothesis(df, save_df_path, kpi_file_path, age_lt_2, filteration_criteria, diagnosis_info):

    filter_column, filter_operator, filter_threshold = filteration_criteria
    scan_df, filteration_kpis = prepare_scan_df_after_filteration(
        df, age_lt_2, filter_column, filter_operator, filter_threshold
    )

    print("Preparation of Scan Df completed")
    print("Now calculating zscore and diagnosis")
    target, predicted = prepare_zscore_and_diagnosis(scan_df, diagnosis_info, save_df_path)
    print("zscore and diagnosis calculation completed")

    TP, FP, TN, FN, medical_kpis = calculate_medical_kpi(
        target_diagnosis=target,
        predicted_diagnosis=predicted
    )

    kpis = {
        "mae_height_b4": scan_df['mae_height_b4'].mean(),
        "mae_weight_b4": scan_df['mae_weight_b4'].mean(),
        "mae_height_after": scan_df['mae_height_after'].mean(),
        "mae_weight_after": scan_df['mae_weight_after'].mean(),
        "front": (scan_df.scan_type.values == 'front').sum(),
        "back": (scan_df.scan_type.values == 'back').sum(),
        "360": (scan_df.scan_type.values == '360').sum(),
        "standing": (scan_df.sl.values == 'standing').sum(),
        "laying": (scan_df.sl.values == 'laying').sum(),
        "filteration_kpis": filteration_kpis,
        "TP": int(TP),
        "FP": int(FP),
        "TN": int(TN),
        "FN": int(FN),
        "medical_kpis": medical_kpis
    }

    with open(kpi_file_path, 'w') as fp:
        json.dump(kpis, fp, indent=4)
