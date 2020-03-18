# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: help.py
@Time: 2020/3/11 10:50 AM
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import argparse


if os.name == "nt":
    bert_dir = ""
    root_dir = ""
else:
    bert_dir = "/home/liushaoweihua/pretrained_lm/bert_chinese/"
    root_dir = "/home/projects/kbqa/tools/Keras-Bert-Kbqa/"


def train_args_parser():

    parser = argparse.ArgumentParser()

    config_group = parser.add_argument_group(
        "Config File Paths", "Config all train information")
    config_group.add_argument("-config",
                             type=str,
                             required=True,
                             help="(REQUIRED) train_config.json")

    save_group = parser.add_argument_group(
        "Model Output Paths", "Config the output paths for model")
    save_group.add_argument("-save_path",
                            type=str,
                            default=os.path.join(root_dir, "models"),
                            help="Model output paths")

    action_group = parser.add_argument_group(
        "Action Configs", "Config the actions during running")
    action_group.add_argument("-device_map",
                              type=str,
                              default="cpu",
                              help="Use CPU/GPU to train. If use CPU, then 'cpu'. "
                                   "If use GPU, then assign the devices, such as '0'. Default is 'cpu'")

    return parser.parse_args()


if __name__ == "__main__":
    parser_type = sys.argv[1].lower()
    if parser_type == "train":
        parser = train_args_parser()
    else:
        raise ValueError("Parser type should be 'train'")
    parser.parse_args()
