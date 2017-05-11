#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest
from create_total_list_by_accuracy import *  # noqa
import shutil

TEST_FILE_DIR_PATH = "../unittest_files"
TEST_FILE_PATH = os.path.join(TEST_FILE_DIR_PATH, "selected_classes.txt")


class Test_create_total_list_by_accuracy(unittest.TestCase):

    def setUp(self):
        selected_classes = Test_create_total_list_by_accuracy.make_selected_classes()
        if not os.path.isdir(TEST_FILE_DIR_PATH):
            os.mkdir(TEST_FILE_DIR_PATH)

        for c in selected_classes:
            path = os.path.join(TEST_FILE_DIR_PATH, c)
            if not os.path.isdir(path):
                os.mkdir(path)

        with open(TEST_FILE_PATH, 'w') as fout:
            fout.write('\n'.join(selected_classes))
            fout.write('\n')

    def tearDown(self):
        if os.path.isdir(TEST_FILE_DIR_PATH):
            shutil.rmtree(TEST_FILE_DIR_PATH)

    @staticmethod
    def make_selected_classes():
        return ['agua', 'aircraft', 'airliner']

    def test_make_map(self):
        label_map = make_map(TEST_FILE_PATH)
        answers = {'aircraft': 1, 'airliner': 2, 'agua': 0}
        self.assertTrue(label_map == answers)

    def test_child_dir_path_generator(self):
        childs = child_dir_path_generator(TEST_FILE_DIR_PATH, TEST_FILE_PATH)
        self.assertTrue(list(childs) == Test_create_total_list_by_accuracy.make_selected_classes())


if __name__ == '__main__':
    unittest.main()
