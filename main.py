import os

from dataset.loader import TrafficDataset
import utils


if __name__ == '__main__':
    root_path = os.path.expanduser('~/datasets/Traffic')
    dataset = TrafficDataset(root_path)
    data = dataset.load_dataset()
