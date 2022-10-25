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

    sizes = [15] * len(tricoords_all_df)
    sizes[-1] = 20  # last point should be bigger showing the target

    fig = scatter_ternary(tricoords_all_df, a='Decrease', b='Stationary', c='Increase', color='Model Type', size=sizes)
    fig.write_image(r'../group_evals/tricoords.png')


def viz_bias_variance(accuracies):


    data = []

    for model_type in accuracies.columns:
        median = accuracies[model_type].median()
        std = accuracies[model_type].std()
        data.append([median,std,model_type])

    data = pd.DataFrame(data,columns = ['Bias','Variance','Model Type'])

    sns.scatterplot(data=data, hue='Model Type', x='Bias', y='Variance',s=400)
    plt.savefig(r'../group_evals/bias_var.png')


def viz_preds_boxplots(accuracies):


    accs = []

    models = []

    for model_type in accuracies.columns:
        accs += accuracies[model_type].values.tolist()
        models += len(accuracies)*[model_type]

    data = pd.DataFrame(zip(accs,models),columns = ['Accuracies','Model Type'])

    ax = sns.boxplot(data=data, x="Accuracies", y="Model Type",hue='Model Type')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(r'../group_evals/preds_boxplot.png')

def main(examined_groups):

    base_dir = r'../group_evals'



    nn_types = []
    tricords = []

    f1scores = []
    accuracies = []

    for group_data_f in os.listdir(base_dir):
        if 'json' not in group_data_f:
            continue

        group_name = group_data_f.split('.')[0]

        if group_name not in examined_groups:
            continue


        with open(os.path.join(base_dir,group_data_f)) as f:
            data = json.load(f)
        nn_type = group_data_f.split('_')[0].upper()

        if nn_type == 'ENCODER':
            nn_type = (group_data_f.split('_')[0] + ' ' + group_data_f.split('_')[1]).capitalize()

        tricords += data['tricoords']

        f1scores.append(pd.Series(data['f1scores'],name=f'{nn_type}'))
        accuracies.append(pd.Series(data['accs'],name=f'{nn_type}'))
        nn_types += [nn_type]*len(data['preds'])

    tricoords_all_df = pd.DataFrame.from_records(tricords,columns=['Decrease', 'Stationary', 'Increase'])

    tricoords_all_df['Model Type'] = nn_types


    accuracies = pd.DataFrame(accuracies).T
    f1scores = pd.DataFrame(f1scores).T



    viz_tricoords(tricoords_all_df)
    plt.show()
    viz_bias_variance(accuracies)
    plt.show()
    viz_preds_boxplots(accuracies)
















if __name__ == '__main__':

    groups_to_eval = [
        'mlp_2022_10_25',
        'cnn_2022_10_25',
        'encoder_block_2022_10_25'
    ]

    main(groups_to_eval)