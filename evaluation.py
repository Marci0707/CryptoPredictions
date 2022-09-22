import json
import os

import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow.python.keras.models
from keras.callbacks import History
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from _common import TrainingConfig


def viz_history(hist : History, training_dir):
    ax = pd.DataFrame(hist.history).filter(regex='loss').plot(figsize=(8, 5))
    pd.DataFrame(hist.history).filter(regex='accuracy').plot(secondary_y=True, ax=ax)
    plt.savefig(os.path.join(training_dir,'history.png'))


def save_model(model : tensorflow.keras.models.Model, config:TrainingConfig,training_dir):

    with open(os.path.join(training_dir,'config.json'),'w') as f:
        config_dict = config.to_dict()
        f.write(json.dumps(config_dict))

    with open(os.path.join(training_dir,'model_summary.txt'),'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    model.save(os.path.join(training_dir,'model'))


def eval_results(y_preds, y_true,training_dir,class_borders=None):

    y_preds = np.argmax(y_preds, axis=1)
    y_true = np.argmax(y_true, axis=1)

    labels = []
    for idx in range(len(class_borders)+1):
        if idx == 0:
            label = f'< {class_borders[idx]}'
        elif idx == len(class_borders):
            label = f'{class_borders[idx-1]} <'
        else:
            label = f'{class_borders[idx-1]} < x < {class_borders[idx]}'
        labels.append(label)

    matrix = confusion_matrix(y_true,y_preds)
    df_cm = pd.DataFrame(matrix, index=labels, columns=labels)
    plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.savefig(os.path.join(training_dir,'confusion_matrix.png'))


    diffs = np.abs(y_preds - y_true)
    res = dict()
    with open(os.path.join(training_dir,'results.json'),'w') as f:
        res['test_size'] = int(len(y_preds))
        res['total_class_distance'] = int(sum(diffs))
        res['total_misclassifications'] = int(np.count_nonzero(diffs))
        res['average_class_distance'] = sum(diffs) / int(len(y_preds))
        res['accuracy'] = 1 - int(np.count_nonzero(diffs)) / int(len(y_preds))
        res['n_classes'] = len(class_borders)+1

        f.write(json.dumps(res))

