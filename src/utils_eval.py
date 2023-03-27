import numpy as np
import pandas as pd
from sklearn import metrics

import matplotlib.pyplot as plt

def get_best_f1(label, score):
    precision, recall, _ = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    return best_f1, best_p, best_r

def get_metrics(label, score):
    auroc = metrics.roc_auc_score(label, score)
    ap = metrics.average_precision_score(y_true=label, y_score=score, average=None)
    best_f1, best_p, best_r = get_best_f1(label, score)

    return auroc, ap, best_f1, best_p, best_r

def adjust_scores(label, score):
    """
    adjust the score for segment detection. i.e., for each ground-truth anomaly segment,
    use the maximum score as the score of all points in that segment. This corresponds to point-adjust f1-score.
    ** This function is copied/modified from the source code in [Zhihan Li et al. KDD21]
    :param score - anomaly score, higher score indicates higher likelihoods to be anomaly
    :param label - ground-truth label
    """
    score = score.copy()
    assert len(score) == len(label)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])
    return score

def get_event_metrics(df, label, score):
    """
    use the corresponding threshold of the best f1 of adjusted scores
    """
    precision, recall, threshold = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_threshold = threshold[np.argmax(f1)]
    label_predict = [s >= best_threshold for s in score]
    label_predict = np.array(label_predict, dtype=int)

    # time is previously used as index when reading data frame, rest index to ordered index here
    df = df.reset_index()
    if 'time' in df.columns:
        df_new = df[['time']].copy()
        df_new['time'] = pd.to_datetime(df_new['time']).dt.ceil('S')
        df_new['label'] = label
        df_new['label_predict'] = label_predict

        label_group = count_group('label', df=df_new, delta='12 hour')
        predict_group = count_group('label_predict', df=df_new, delta='12 hour')
        true_group = count_group('label', 'label_predict', df=df_new, delta='12 hour')

        event_precision = true_group / predict_group
        event_recall = true_group / label_group

    else:
        # @TODO event metrics for data frames that are without time column.
        event_precision = -1
        event_recall = -1

    return event_precision, event_recall

def count_group(*args, df, delta):
    if len(args) == 1:
        df_y = df[df[args[0]] == 1]
    if len(args) == 2:
        df_y = df[(df[args[0]] == 1) & (df[args[1]] == 1)]
    df_y_cur1 = df_y.iloc[:-1, :]
    df_y_cur2 = df_y.iloc[1:, :]
    df_y_cur = [df_y_cur2['time'].iloc[i] - df_y_cur1['time'].iloc[i] for i in range(df_y.shape[0] - 1)]
    num_group = 1
    for i in range(len(df_y_cur)):
        if df_y_cur[i] > pd.Timedelta(delta):
            num_group += 1
    return num_group
