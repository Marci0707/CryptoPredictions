import json
import os

import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow.python.keras.models
from keras.callbacks import History
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.dates as mdates
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


def _get_prediction_colors(ypred,ytrue):

    n_classes = len(np.unique(ytrue))
    if n_classes == 3:
        colors = ['r','y','g']
    elif n_classes == 4:
        colors = ['maroon','darksalmon','palegreen','green']
    elif n_classes == 5:
        colors = ['maroon', 'darksalmon','yellow', 'palegreen', 'green']
    else:
        raise ValueError(f'too many classes to color nicely at the moment. {np.unique(ytrue)}')

    good_prediction_alpha = 0.5
    bad_prediction_alpha = 1
    alphas = []
    for i in range(len(ypred)):
        alpha = good_prediction_alpha if ypred[i] == ytrue[i] else bad_prediction_alpha
        alphas.append(alpha)
    pred_colors = [colors[i] for i in ypred]
    true_colors = [colors[i] for i in ytrue]

    return colors,alphas,pred_colors,true_colors


def eval_results(y_preds, y_true,test_data, training_dir,class_borders=None):

    y_preds = np.argmax(y_preds, axis=1)
    y_true = np.argmax(y_true, axis=1)

    #plot confusion matrix
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
    plt.show()


    #plot predictions
    colors,alphas,pred_colors,true_colors = _get_prediction_colors(y_preds,y_true)
    fig,axes = plt.subplots(nrows=1,ncols=2)
    plt.suptitle('prediction classes vs real classes')
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 12, 1)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    axes[0].plot(test_data['time'], test_data["close"],alpha=alphas, c=pred_colors)
    axes[0].set_title('predictions')
    axes[1].plot(test_data['time'], test_data["close"],alpha=alphas, c=true_colors)
    axes[0].set_title('true classes')
    plt.legend()
    plt.savefig('comparison.png')


    #save results.json
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





