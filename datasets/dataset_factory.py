from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import shtech_dataset

datasets_map = {
    'shtech_part_A': shtech_dataset,
    'shtech_part_B': shtech_dataset
}

def get_dataset(name, dataset_dir, FLAGS):
    if name not in datasets_map:
        raise ValueError('currently NOT support dataset name: %s' % name)
    return datasets_map[name].get_dataset(dataset_dir, FLAGS)

