import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def get_tri_point(y):
    unique, counts = np.unique(y, return_counts=True)
    count_dict = dict(zip(unique,counts))
    denom = sum(counts)

    coefs = []

    for i in (0,1,2):
        val = count_dict.get(i,0)/denom
        coefs.append(val)


    return coefs

def main(group_to_eval):

    trainings_dir = '../trainings'

    y_true = np.argmax(np.load('../splits/test/y_preprocessed.npy',allow_pickle=True),axis=1)

    accuracies = []
    f1scores = []


    tricoords = []

    for train_id in os.listdir(trainings_dir):
        if group_to_eval not in train_id:
            continue

        y_preds = np.load(os.path.join(trainings_dir,train_id,'preds.npy'))

        accuracies.append(accuracy_score(y_true,y_preds))
        f1scores.append(f1_score(y_true,y_preds,average ='macro'))

        coord =  get_tri_point(y_preds)
        tricoords.append(coord)


    group_results = {
        'f1scores' : f1scores,
        'accs' : accuracies,
        'tricoords' : tricoords,
        'var' : np.std(y_preds)
    }


    with open(os.path.join('..','group_evals',f'{group_to_eval}.json'),'w+') as f:
        json.dump(group_results,f)









if __name__ == '__main__':
    group_to_eval = 'mlp_2022_10_25'
    main(group_to_eval)