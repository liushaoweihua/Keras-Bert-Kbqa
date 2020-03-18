# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: models.py
@Time: 2020/3/13 03:58 PM
"""


import sys
sys.path.append("../..")

from keras_bert_kbqa.train import train
from keras_bert_kbqa.helper import train_args_parser


def run_train():
    args = train_args_parser()
    if True:
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    train(args=args)


if __name__ == "__main__":
    run_train()