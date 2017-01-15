#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from PIL import Image
import sys

DIR_PATH = "/Users/kumada/Data/image_net/images"

def check_dir(dir_path):
    c = 0
    for f in os.listdir(dir_path):
        fp = os.path.join(dir_path, f)
        try: 
            image = Image.open(fp)
            c += 1
        except IOError, e:
            print(" {} is not image".format(fp))
    return c


if __name__ == "__main__":
    for root, dirs, files in os.walk(DIR_PATH):
        for d in dirs:
            dir_path = os.path.join(root, d)
            count = check_dir(dir_path)
            print("{a}:{c}".format(a=dir_path, c=count))



