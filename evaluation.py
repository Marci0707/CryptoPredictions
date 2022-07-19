import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import mean_squared_error


def evalute_model(model, X, Y, y_scaler, model_name, only_last_values=None):
    X = X.copy()
    Y = Y.copy()

    if only_last_values is not None:
        X = X[- min(len(X), only_last_values):]
        Y = Y[- min(len(Y), only_last_values):]



    y_preds = model.predict(X)
    y_preds = y_preds.reshape((-1,1))
    y_preds = y_scaler.inverse_transform(y_preds)
    y_true = y_scaler.inverse_transform(Y)

    mse = mean_squared_error(y_true, y_preds)

    time_axis = np.arange(0, len(y_true), 1)

    figure(figsize=(15, 10))
    plt.title(f"{model_name} predictions. mse={round(mse, 2)}")
    plt.plot(time_axis, y_preds)
    plt.plot(time_axis, y_true)

    plt.savefig(model_name + ".png")

    plt.show()
