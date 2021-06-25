import numpy as np
import argparse
import os.path as osp
from dataset.loader import TrafficDataset, DEFAULT_CLASS_LIST

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def main(config: dict):
    root_path = osp.expanduser('~/datasets/Traffic')
    dataset = TrafficDataset(root_path)
    df = dataset.data
    df = df.iloc[:, 1:]
    x, y = df.iloc[:, :-1], df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    train_data = xgb.DMatrix(data=x_train, label=y_train)
    test_data = xgb.DMatrix(data=x_test, label=y_test)

    param = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'max_depth': 6,
        'nthread': 16,
        'num_class': 8
    }

    watchlist = [(train_data, 'train'), (test_data, 'test')]
    num_round = 200

    model = xgb.train(param, train_data, num_round, watchlist)

    y_true = np.array(y_test)
    y_pred = model.predict(test_data)
    y_pred = np.array(y_pred, dtype=np.int)

    acc = np.sum(y_pred == y_true) / y_pred.shape[0]

    print(classification_report(y_true, y_pred, target_names=DEFAULT_CLASS_LIST))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='~/datasets/Traffic')
    parser.add_argument('--slice_length', type=float, default=15.0)
    args = parser.parse_args()
    main(args.__dict__)
