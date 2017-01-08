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

DATA_DIR = "/Users/kumada/Data/VOC2012/ImageSets/Main"
LABEL_DICT_PATH = "./label_dictionary.txt"
DATASET_PATH = "./dataset.txt"


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
            fout.write("{i} {l}\n".format(i=i, l=l))


LabelInfo = collections.namedtuple("LabelInfo", ["val", "train", "trainval"])


def extract_file_path(path, label, fout):
    for line in open(path):
        tokens = line.strip().split()
        if tokens[1] == "1":
            fout.write("{p} {l}\n".format(p=tokens[0], l=label))


def make_dataset(path, kind, output_path):
    fout = open(output_path, "w")
    for fp in file_path_generator(path):
        file_name = os.path.basename(fp)
        name, _ = os.path.splitext(file_name)
        m = LABEL_PATTERNS[kind].match(name)
        if m:
            extract_file_path(fp, m.group(1), fout)


if __name__ == "__main__":

    make_labels(DATA_DIR, LABEL_DICT_PATH)
    make_dataset(DATA_DIR, "trainval", DATASET_PATH)
