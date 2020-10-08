#encoding: utf-8
from __future__ import print_function
import os
import platform
from utils import mkdir
from xml.dom import minidom
import json


class LPBAConfig(object):
    """LPBA40数据集"""

    def __init__(self):
        if 'Win' in platform.system():
            print("using data from windows")
            #self.data_dir = 'F:/datesets/Dataset/LPBA40/delineation_space/LPBA40/'
            self.data_dir = 'F:/datesets/Dataset/LPBA40/delineation_space/LPBA40/'
            self.n_workers = 1
        else:
            print("using data from linux")
            self.data_dir = '/home/eric/dataset/LPBA40'
            #self.data_dir = '/home/zzy/origin_data/lpba_96/'
            self.n_workers = 15
        self.label_xml_file = os.path.join('./lpba40.label.xml')
        self.n_split_folds = 4
        self.select = 0
        self.seed = 42
        self.n_labels = 40
        self.lr = 1e-4
        self.batch_size = 1
        self.log_dir = './logs/LPBA'
        self.epoch = 300
        mkdir(self.log_dir)
        self.parse_label()

    def parse_label(self):
        label_xml = minidom.parse(self.label_xml_file)
        label_list = label_xml.getElementsByTagName('label')
        print('number of labels is ', len(label_list))
        self.label = {}
        for label in label_list:
            self.label[label.attributes['id'].value] = label.attributes['fullname'].value

    def __str__(self):
        print("net work config")
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]
        str_list.insert(0, "*" * 80)
        str_list.append("*" * 80)

        return '\n'.join(str_list)
