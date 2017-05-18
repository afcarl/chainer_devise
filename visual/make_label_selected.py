#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import argparse

if __name__ == '__main__':

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_path", help='input: a path to a file in which selected classes are described')
        parser.add_argument("--output_path", help='output: a path to a file in which selected classes are described')
        args = parser.parse_args()

        with open(args.output_path, 'w') as fout:
            for i, line in enumerate(open(args.input_path)):
                line = line.strip()
                fout.write('{} {}\n'.format(line, i))
    except IOError, e:
        print(e)
