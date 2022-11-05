import json
import os
import re

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def get_tri_point(y):
    unique, counts = np.unique(y, return_counts=True)
    count_dict = dict(zip(unique, counts))
    denom = sum(counts)

    coefs = []

    for i in (0, 1, 2):
        val = count_dict.get(i, 0) / denom
        coefs.append(val + 0.1)

    return coefs


def main(group_to_eval):
    trainings_dir = os.path.join('..', 'trainings', group_to_eval)

    params = '_'.join([group_to_eval.split('_')[-2], group_to_eval.split('_')[-1]])


    preds_all = []
    preds_all_h0 = []

    for train_id in os.listdir(trainings_dir):
        if group_to_eval not in train_id:
            continue


        # y_preds = np.load(os.path.join(trainings_dir, train_id, 'pred.npy'), allow_pickle=True)
        try:
            y_preds = np.load(os.path.join(trainings_dir, train_id, 'preds_h0_p0.npy'), allow_pickle=True)
        except FileNotFoundError as e:
            y_preds = np.load(os.path.join(trainings_dir, train_id, 'pred_h0.npy'), allow_pickle=True)

        # y_preds_1d = y_preds[:, 1]
        y_preds_h0_1d = y_preds[:, 1]

        # preds_all.append(y_preds_1d.tolist())
        preds_all_h0.append(y_preds_h0_1d.tolist())

    with open(os.path.join(trainings_dir, os.listdir(trainings_dir)[0], 'model_summary.txt')) as f:
        content = f.read()
        match = re.search(r'Total params:(.*\n)', content)
        params = match[1].replace(',', '').strip()

    group_results = {
        'preds': preds_all_h0,
        # 'preds_h0': preds_all_h0,
        'params': params
    }

    target_name = f'{group_to_eval}.json'.replace('_test','')

    with open(os.path.join('..', 'group_evals', 'data', target_name), 'w+') as f:
        json.dump(group_results, f)


if __name__ == '__main__':
    h = [0,1, 3]
    p = [0,1, 3, 5, 8, 13]
    models = ['cnn', 'mlp', 'lstm', 'encoder_block', 'encoder_stack']
    for model in models:
        for hh in h:
            for pp in p:
                model_name = f'{model}_h{hh}_p{pp}_test'
                main(model_name)
