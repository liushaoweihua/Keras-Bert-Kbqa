# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: predict.py
@Time: 2020/3/16 3:39 PM
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.models import load_model
from keras.utils import CustomObjectScope
from .utils import custom_objects


def predict(model_path):
    """模型预测流程
    """
    # 环境设置
    with CustomObjectScope(custom_objects):
        model = load_model(model_path)

    return model
