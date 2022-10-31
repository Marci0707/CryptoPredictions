import os

import numpy as np
import tensorflow as tf
from tensorflow_addons.metrics import F1Score


def main():
    for grp in os.listdir('../archive'):
        print(grp)

        h_type = grp.split('_')[-2][-1]
        p_type = grp.replace('_test','').split('_')[-1][-1]
        for training in os.listdir(f'../archive/{grp}'):
            model = tf.keras.models.load_model(f'../archive/{grp}/{training}/model',custom_objects={'f1':F1Score(num_classes=2, average='weighted')})
            wo_smoothing = np.load(f'../splits/test/x_preprocessed_h0_p{p_type}.npy')
            preds = model.predict(wo_smoothing)
            np.save(f'../archive/{grp}/{training}/preds_h0.npy',preds)

if __name__ == '__main__':
    main()