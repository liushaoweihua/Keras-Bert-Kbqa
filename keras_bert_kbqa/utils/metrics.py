# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: metrics.py
@Time: 2020/3/9 1:48 PM
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import keras.backend as K


class CrfAcc:
    """训练过程中显示的CRF精度
    """
    def __init__(self, tag_to_id, mask_tag=None):
        self.tag_to_id = tag_to_id
        self.mask_tag_id = tag_to_id.get(mask_tag)
        self.numb_tags = len(tag_to_id)

    def crf_accuracy(self, y_true, y_pred):
        """计算viterbi-crf精度
        """
        crf, idx = y_pred._keras_history[:2]
        X = crf._inbound_nodes[idx].input_tensors[0]
        y_pred = crf.viterbi_decoding(X, None)
        return self._get_accuracy(y_true, y_pred, crf.sparse_target)

    def _get_accuracy(self, y_true, y_pred, sparse_target=False):
        y_pred = K.argmax(y_pred, -1)
        mask = K.cast(1. - K.one_hot(
            K.squeeze(K.cast(y_true, "int32"), axis=-1),
            num_classes=self.numb_tags)[:, :, self.mask_tag_id], K.floatx())
        if sparse_target:
            y_true = K.cast(y_true[:, :, 0], K.dtype(y_pred))
        else:
            y_true = K.argmax(y_true, -1)
        judge = K.cast(K.equal(y_true, y_pred), K.floatx())
        if self.mask_tag_id is None:
            return K.mean(judge)
        else:
            return K.sum(judge * mask) / K.sum(mask)

class CrfLoss:
    """训练过程中显示的CRF损失
    """
    def __init__(self, tag_to_id, mask_tag=None):
        self.tag_to_id = tag_to_id
        self.mask_tag_id = tag_to_id.get(mask_tag)
        self.numb_tags = len(tag_to_id)

    def crf_loss(self, y_true, y_pred):
        """计算viterbi-crf损失
        """
        crf, idx = y_pred._keras_history[:2]
        if crf.sparse_target:
            y_true = K.one_hot(K.cast(y_true[:, :, 0], "int32"), crf.units)
        X = crf._inbound_nodes[idx].input_tensors[0]
        mask = K.cast(1. - y_true[:, :, self.mask_tag_id], K.floatx()) if self.mask_tag_id else None
        nloglik = crf.get_negative_log_likelihood(y_true, X, mask)
        return nloglik