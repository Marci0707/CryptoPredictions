import json
import os
from typing import Sequence

import matplotlib.colors
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow.python.keras.models
from keras.callbacks import History
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.dates as mdates
from _common import TrainingConfig
import matplotlib.patches as mpatches


def pad_values(array, labels):
    padded = []
    for i in range(labels):
        if i in array:
            padded.append(i)


def viz_history(hist: History, training_dir):
    ax = pd.DataFrame(hist.history).filter(regex='loss').plot(figsize=(8, 5))
    pd.DataFrame(hist.history).filter(regex='accuracy').plot(secondary_y=True, ax=ax)
    plt.savefig(os.path.join(training_dir, 'history.png'))


def save_model(model: tensorflow.keras.models.Model, config: TrainingConfig, training_dir):
    with open(os.path.join(training_dir, 'config.json'), 'w') as f:
        config_dict = config.to_dict()
        f.write(json.dumps(config_dict))

    with open(os.path.join(training_dir, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    model.save(os.path.join(training_dir, 'model'),save_format="h5")


def _get_prediction_colors(ypred, ytrue, banned_indices, test_data):
    is_predicted_index = [i not in banned_indices for i in range(len(test_data))]

    test_indices = 0
    ypreds_verbose = []  # zipping predictions with not predicted indices
    ytrue_verbose = []  # zipping predictions with not predicted indices
    for idx in range(len(is_predicted_index)):
        if is_predicted_index[idx]:
            ypreds_verbose.append(ypred[test_indices])
            ytrue_verbose.append(ytrue[test_indices])
            test_indices += 1
        else:
            ytrue_verbose.append(None)
            ypreds_verbose.append(None)

    n_classes = len(np.unique(ytrue))
    if n_classes == 3:
        colors = ['r', 'y', 'g']
    elif n_classes == 4:
        colors = ['maroon', 'darksalmon', 'palegreen', 'green']
    elif n_classes == 5:
        colors = ['maroon', 'darksalmon', 'yellow', 'palegreen', 'green']
    else:
        raise ValueError(f'too many classes to color nicely at the moment. {np.unique(ytrue)}')

    pred_colors = [colors[value] if value is not None else 'k' for value in ypreds_verbose]
    true_colors = [colors[value] if value is not None else 'k' for value in ytrue_verbose]

    rgba_pred = []
    rgba_true = []
    good_prediction_alpha = 0.5
    bad_prediction_alpha = 1

    for i in range(len(ytrue_verbose)):
        alpha = good_prediction_alpha if ytrue_verbose is None or ytrue_verbose[i] == ypreds_verbose[
            i] else bad_prediction_alpha
        color_pred = matplotlib.colors.to_rgba(pred_colors[i], alpha=alpha)
        color_true = matplotlib.colors.to_rgba(true_colors[i], alpha=alpha)

        rgba_pred.append(color_pred)
        rgba_true.append(color_true)

    return colors, rgba_pred, rgba_true


def eval_results(y_preds, y_true, test_data, training_dir, banned_indices: Sequence[int], regression_ahead: int,
                 class_borders=None, ):
    y_preds = np.argmax(y_preds, axis=1)
    y_true = np.argmax(y_true, axis=1)

    # save results.json
    diffs = np.abs(y_preds - y_true)
    res = dict()
    with open(os.path.join(training_dir, 'results.json'), 'w') as f:
        res['test_size'] = int(len(y_preds))
        res['total_class_distance'] = int(sum(diffs))
        res['total_misclassifications'] = int(np.count_nonzero(diffs))
        res['average_class_distance'] = sum(diffs) / int(len(y_preds))
        res['accuracy'] = 1 - int(np.count_nonzero(diffs)) / int(len(y_preds))
        res['n_classes'] = len(class_borders) + 1

        f.write(json.dumps(res))

    # plot confusion matrix
    labels = []
    for idx in range(len(class_borders) + 1):
        if idx == 0:
            label = f'slope < {class_borders[idx]}'
        elif idx == len(class_borders):
            label = f'{class_borders[idx - 1]} < slope'
        else:
            label = f'{class_borders[idx - 1]} < slope < {class_borders[idx]}'
        labels.append(label)

    matrix = confusion_matrix(y_true, y_preds)
    df_cm = pd.DataFrame(matrix, index=labels, columns=labels)
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.savefig(os.path.join(training_dir, 'confusion_matrix.png'))


    # plot prediction comparisons
    colors, pred_colors, true_colors = _get_prediction_colors(y_preds, y_true, banned_indices, test_data)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
    acc = 1 - int(np.count_nonzero(diffs)) / int(len(y_preds))
    plt.suptitle(f'Predicted Classes vs Labels\naccuracy: {round(acc, 2)}', fontweight="bold")
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 12, 1)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
        ax.tick_params(axis='x', labelrotation=90)
    axes[0].scatter(test_data['time'], test_data["close"], c=pred_colors, s=50)
    axes[0].set_title('Predictions')
    axes[0].set_ylabel('BTC/USD rate')

    axes[1].scatter(test_data['time'], test_data["close"], c=true_colors, s=50)
    axes[1].set_title('Labels')
    axes[1].set_ylabel('BTC/USD rate')

    patches = []
    for idx, color in enumerate(colors):
        patch = mpatches.Patch(color=color, label=labels[idx])
        patches.append(patch)
    patches.append(mpatches.Patch(color='black', label='not predicted'))
    plt.legend(handles=patches, fancybox=True, title=f"slope of regression for {regression_ahead} days",
               loc='lower left')

    plt.tight_layout()
    plt.savefig(os.path.join(training_dir, 'comparison.png'))


    # plot distributions
    pred_distr = {x: list(y_preds).count(x) for x in y_preds}
    true_distr = {x: list(y_true).count(x) for x in y_true}
    for i in range(len(labels)):
        if i not in pred_distr.keys():
            pred_distr[i] = 0
        if i not in true_distr.keys():
            true_distr[i] = 0

    pred_distr = {k: v for k, v in sorted(pred_distr.items(), key=lambda item: item[0])}
    true_distr = {k: v for k, v in sorted(true_distr.items(), key=lambda item: item[0])}

    df = pd.concat([pd.Series(pred_distr.values()), pd.Series(true_distr.values()),
                    pd.Series([label.replace(' ', '').replace('slope', 's') for label in labels])], axis=1)
    df.columns = ['Predictions', 'Labels', 'classes']
    df.set_index('classes', inplace=True, drop=True)
    df.plot.bar(ylabel='Count', xlabel='Classified Slopes', rot=0)
    plt.title("Class Distributions", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(training_dir, 'histplots.png'))
