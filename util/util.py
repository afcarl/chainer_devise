#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import os
import re
import collections


LABEL_PATTERNS = {
    "val": re.compile(r"(\w+)_val"),
    "train": re.compile(r"(\w+)_train"),
    "trainval": re.compile(r"(\w+)_trainval"),
}

# DATA_DIR = "/Users/kumada/Data/VOC2012/ImageSets/Main"
# LABEL_DICT_PATH = "./label_dictionary.txt"
# DATASET_PATH = "./dataset.txt"

# IMAGE_DIR_PATH = "/home/ubuntu/data/voc2012/JPEGImages"
# DATASET_PATH = "/home/ubuntu/data/voc2012/dataset.txt"
# LABEL_DICT_PATH = "/home/ubuntu/data/voc2012/label_dictionary.txt"
FULL_PATH_DATASET_PATH = "/home/ubuntu/data/voc2012/full_path_dataset.txt"
TRAINING_DATASET_PATH = "/home/ubuntu/data/voc2012/training.txt"
TESTING_DATASET_PATH = "/home/ubuntu/data/voc2012/testing.txt"
RATE = 0.7


def file_path_generator(path):
    for fname in os.listdir(path):
        full_path = os.path.join(path, fname)
        yield full_path


def make_labels(path, output_path):
    result = set()
    for fp in file_path_generator(path):
        file_name = os.path.basename(fp)
        name, _ = os.path.splitext(file_name)
        for pattern in LABEL_PATTERNS.values():
            m = pattern.match(name)
            if m:
                result.add(m.group(1))

    ls = list(result)
    with open(output_path, "w") as fout:
        for (i, l) in enumerate(ls):
            fout.write("{i} {L}\n".format(i=i, L=l))


LabelInfo = collections.namedtuple("LabelInfo", ["val", "train", "trainval"])


def extract_file_path(path, label, fout):
    for line in open(path):
        tokens = line.strip().split()
        if tokens[1] == "1":
            fout.write("{p} {L}\n".format(p=tokens[0], L=label))


def make_dataset(path, kind, output_path):
    fout = open(output_path, "w")
    for fp in file_path_generator(path):
        file_name = os.path.basename(fp)
        name, _ = os.path.splitext(file_name)
        m = LABEL_PATTERNS[kind].match(name)
        if m:
            extract_file_path(fp, m.group(1), fout)


def make_full_path_dataset(dataset_path, image_dir_path,
                           full_path_dataset_path):
    with open(dataset_path) as fin:
        with open(full_path_dataset_path, "w") as fout:
            for line in fin:
                tokens = line.strip().split()
                filename = tokens[0]
                label = tokens[1]
                full_path = os.path.join(image_dir_path, filename + ".jpg")
                assert os.path.exists(full_path), ""
                fout.write("{p} {L}\n".format(p=full_path, L=label))


def split_dataset(full_path_dataset_path, rate):
    paths = collections.defaultdict(list)
    with open(full_path_dataset_path) as fin:
        for line in fin:
            tokens = line.strip().split()
            label = tokens[1]
            path = tokens[0]
            paths[label].append(path)

    for (key, value) in paths.items():
        print(key, len(value))


if __name__ == "__main__":

    make_labels(DATA_DIR, LABEL_DICT_PATH)
    # make_dataset(DATA_DIR, "trainval", DATASET_PATH)

    # make_full_path_dataset(DATASET_PATH, IMAGE_DIR_PATH,
    #                        FULL_PATH_DATASET_PATH)

    # split_dataset(FULL_PATH_DATASET_PATH, RATE)
