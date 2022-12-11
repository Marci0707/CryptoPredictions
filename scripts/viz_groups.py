import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from plotly.express import scatter_ternary

from scripts.eval_group import get_tri_point


def viz_tricoords(tricoords_all_df):
    y_true = np.argmax(np.load('../splits/test/y_preprocessed.npy', allow_pickle=True), axis=1)

    target = get_tri_point(y_true)
    tri_coord_target = pd.DataFrame.from_records([target], columns=['Decrease', 'Stationary', 'Increase'])

    tri_coord_target['Model Type'] = 'True Labels'

    tricoords_all_df = pd.concat([tricoords_all_df, tri_coord_target], axis=0, ignore_index=True)

    sizes = [3] * len(tricoords_all_df)
    sizes[-1] = 10  # last point should be bigger showing the target

    fig = scatter_ternary(tricoords_all_df, a='Decrease', b='Stationary', c='Increase', color='Model Type', size=sizes)
    fig.layout.ternary.aaxis.layer = 'below traces'
    fig.layout.ternary.baxis.layer = 'below traces'
    fig.layout.ternary.caxis.layer = 'below traces'
    fig.update_traces(cliponaxis=False, selector=dict(type='scatter'))
    fig.write_image(r'../group_evals/tricoords.png')


def viz_bias_variance(accuracies, grpname):
    data = []

    for model_type in accuracies.columns:
        median = accuracies[model_type].median()
        std = accuracies[model_type].var()
        data.append([median, std, model_type])

    data = pd.DataFrame(data, columns=['Bias', 'Variance', 'Model Type'])

    sns.scatterplot(data=data, hue='Model Type', x='Bias', y='Variance', s=400, alpha=0.5)

    plt.savefig(fr'../group_evals/bias_var_{grpname}.png')
    plt.show()


def viz_preds_boxplots(accuracies):
    accs = []

    models = []

    for model_type in accuracies.columns:
        accs += accuracies[model_type].values.tolist()
        models += len(accuracies) * [model_type]

    data = pd.DataFrame(zip(accs, models), columns=['Accuracies', 'Model Type'])

    ax = sns.boxplot(data=data, x="Accuracies", y="Model Type", hue='Model Type')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(r'../group_evals/preds_boxplot.png')


def main(model_name,p,h):
    data_dir = os.path.join('..','group_evals','data')


    model_type = model_name.split('_')[0]
    if model_type == 'encoder':
        model_type = '_'.join(model_name.split('_')[2])

    nn_types = []
    accuracies = []

    for training_data in os.listdir(data_dir):

        with open(os.path.join(data_dir, model_name+'.json')) as f:
            data = json.load(f)


        accuracies.append(pd.Series(data['accs'], name=f'{model_type}'))
        nn_types += [model_type] * len(data['preds'])



    accuracies = pd.DataFrame(accuracies).T


    # viz_bias_variance(accuracies, grpid)
    plt.show()
    viz_preds_boxplots(accuracies)


if __name__ == '__main__':
    h = [1]
    p = [0,1,2,3]
    models = ['cnn','mlp','lstm','encoder_block','encoder_stack']

    for pp in p:
        for hh in h:
            param_suffix = f'h{hh}_p{pp}'
            for model in models:
                model_name = f'{model}_{param_suffix}'
                main(model_name,pp,hh)
