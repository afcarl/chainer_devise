#!/usr/bin/env python
#coding:utf-8

import os.path
import random

DirPath = "/home/ubuntu/data/lsp15_256"

LabelMap = {
    "MITcoast":         0,
    "MIThighway":       1,
    "MITmountain":      2,
    "MITstreet":        3,
    "MITforest":        4,
    "MITinsidecity":    5,
    "MITopencountry":   6,
    "MITtallbuilding":  7,    
    "bedroom":          8,
    "CALsuburb":        9,
    "industrial":       10,
    "kitchen":          11,
    "livingroom":       12,
    "PARoffice":        13,
    "store":            14   
}

def extract_file_paths(path):
    paths = []
    for (root, dirs, files) in os.walk(path):
        for file in files:
            if ".jpg" in file:
                
                paths.append(os.path.join(root, file))
    return paths

def create_list(dir_name, label):
    src_path = os.path.join(DirPath, dir_name)
    dst_path = os.path.join(DirPath, dir_name, "list.txt")
    paths = extract_file_paths(src_path)
    random.shuffle(paths)
    print(dir_name, len(paths))
    with open(dst_path, "w") as f:
       for path in paths:
           f.write("{p} {l}\n".format(p = path, l = label))

if __name__ == "__main__":
    for (key, value) in LabelMap.items():
        create_list(key, value)

