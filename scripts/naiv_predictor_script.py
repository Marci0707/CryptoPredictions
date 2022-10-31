from _preprocessing import CryptoCompareReader, LinearCoefficientTargetGenerator, drop_columns_deemed_as_useless
import numpy as np

def main():

    y_test = np.load('../splits/test/y_preprocessed.npy',allow_pickle=True)
    x_test = np.load('../splits/test/x_preprocessed.npy',allow_pickle=True)


    close_idx = 3
    window_size = 10
    regression_ahead = 5

    print(x_test[:,:,3].shape)
    print(x_test.shape)


if __name__ == '__main__':
    main()
