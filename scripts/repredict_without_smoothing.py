import os
import time

import numpy as np
import tensorflow as tf
from tensorflow_addons.metrics import F1Score


def main():
    wo_smoothing = np.load(f'../splits/test/x_preprocessed_h0_p0.npy')
    for grp in os.listdir('../trainings'):
        print(grp,time.time())
        if 'cnn' in grp or 'lstm' in grp or 'encoder' in grp or 'mlp' in grp:

            for training in os.listdir(f'../trainings/{grp}'):
                model = tf.keras.models.load_model(f'../trainings/{grp}/{training}/model',custom_objects={'f1':F1Score(num_classes=2, average='weighted')})
                preds = model.predict(wo_smoothing)
                np.save(f'../trainings/{grp}/{training}/preds_h0_p0.npy',preds)

if __name__ == '__main__':
    main()