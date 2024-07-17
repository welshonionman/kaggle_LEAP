import numpy as np


def standardize(x_train, y_train, min_std):
    mean_x = x_train.mean(axis=0)
    mean_y = y_train.mean(axis=0)

    std_x = np.maximum(x_train.std(axis=0), min_std)
    std_y = np.maximum(y_train.std(axis=0), min_std)

    x_train = (x_train - mean_x.reshape(1, -1)) / std_x.reshape(1, -1)
    y_train = (y_train - mean_y.reshape(1, -1)) / std_y.reshape(1, -1)
    return x_train, y_train, mean_x, std_x, mean_y, std_y


def normalize(x_train, y_train):
    # 各列の1パーセンタイルと99パーセンタイルを計算
    p01_x = np.percentile(x_train, 1, axis=0)
    p99_x = np.percentile(x_train, 99, axis=0)
    p01_y = np.percentile(y_train, 1, axis=0)
    p99_y = np.percentile(y_train, 99, axis=0)

    x_train = np.clip(x_train, p01_x, p99_x)
    y_train = np.clip(y_train, p01_y, p99_y)

    # 正規化関数
    def normalize_column(col, p01, p99):
        return (col - p01) / (p99 - p01)

    # 各列を1パーセンタイルと99パーセンタイルを使って正規化
    x_train = np.apply_along_axis(lambda col: normalize_column(col, p01_x, p99_x), axis=1, arr=x_train)
    y_train = np.apply_along_axis(lambda col: normalize_column(col, p01_y, p99_y), axis=1, arr=y_train)

    return x_train, y_train
