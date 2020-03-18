# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: callbacks.py
@Time: 2020/3/9 10:16 AM
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import codecs
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping
from .decoder import Viterbi


class TaskSwitch(Callback):
    """模型开关，达到阈值时，同时训练所有层参数
    """
    def __init__(self, monitor, threshold):
        super(TaskSwitch, self).__init__()
        self.monitor = monitor
        self.threshold = threshold
        if "acc" in self.monitor:
            self.monitor_op = np.greater
        elif "loss" in self.monitor:
            self.monitor_op = np.less
        else:
            raise ValueError("monitor is not either 'acc' or 'loss'")
            
    def on_epoch_end(self, epoch, logs=None):
        if self.monitor_op(logs.get(self.monitor), self.threshold):
            self.model.stop_training = True
            
            
def KbqaCallbacks(best_fit_params):
    """Kbqa模型训练的指标回调函数
    """
    callbacks = []
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=best_fit_params.get("early_stop_patience"),
        verbose=1)
    reduce_lr_on_plateau = ReduceLROnPlateau(
        monitor="val_loss",
        factor=best_fit_params.get("reduce_lr_factor"),
        patience=best_fit_params.get("reduce_lr_patience"),
        verbose=1)
    callbacks.extend([early_stopping, reduce_lr_on_plateau])
    return callbacks
