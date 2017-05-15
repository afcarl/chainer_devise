#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest
from create_total_list_by_accuracy import *  # noqa
import shutil

TEST_FILE_DIR_PATH = "../unittest_files"
TEST_FILE_PATH = os.path.join(TEST_FILE_DIR_PATH, "selected_classes.txt")

CONTENTS = {
    'agua': [
        '132_DSC00163.jpg 0 train',
        '26_fj0766.JPG 0 train',
        '151_bufoMarinus_kl.jpg 0 train',
    ],
    'aircraft': [
        '682_1204109417_1da972f127.jpg 1 train',
        '104_aircraft-carrier-47.jpg 1 train',
        '519_87393main_EC02-0106-01.jpg 1 train',
    ],
    'airliner': [
        '615_993813802_2d27695daa.jpg 2 train',
        '40_104958073_9c13cb4a16.jpg 2 train',
        '49_252815126_25932f8fe1.jpg 2 train',
    ],
}


class Test_create_total_list_by_accuracy(unittest.TestCase):

    def make_file(self, file_name, contents, dir_path):
        with open(os.path.join(dir_path, file_name), 'w') as fout:
            fout.write('\n'.join(contents))
            fout.write('\n')

    def setUp(self):
        selected_classes = Test_create_total_list_by_accuracy.make_selected_classes()
        if not os.path.isdir(TEST_FILE_DIR_PATH):
            os.mkdir(TEST_FILE_DIR_PATH)

        for c in selected_classes:
            path = os.path.join(TEST_FILE_DIR_PATH, c)
            if not os.path.isdir(path):
                os.mkdir(path)
            self.make_file('dataset_list.txt', CONTENTS[c], path)

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

    def test_create_total_list(self):
        label_map = make_map(TEST_FILE_PATH)
        total_list = []
        for child in child_dir_path_generator(TEST_FILE_DIR_PATH, TEST_FILE_PATH):
            create_total_list(TEST_FILE_DIR_PATH, child, total_list, label_map[child])
        print(total_list)

    def test_replace_labels(self):
        tmp_list = ['agua/132_DSC00163.jpg 0 train', 'agua/26_fj0766.JPG 0 train', 'agua/151_bufoMarinus_kl.jpg 0 train']
        new_label = 3
        new_tmp_list = replace_labels(tmp_list, new_label)
        answer_list = ['agua/132_DSC00163.jpg 3 train', 'agua/26_fj0766.JPG 3 train', 'agua/151_bufoMarinus_kl.jpg 3 train']
        self.assertTrue(new_tmp_list == answer_list)


if __name__ == '__main__':
    unittest.main()
