# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: __init__.py
@Time: 2020/3/9 10:13 AM
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.utils import get_custom_objects
from .bert import MultiHeadAttention, LayerNormalization, PositionEmbedding, FeedForward, EmbeddingDense
from .models import ExpandDims, MultiLossLayer
from .metrics import CrfAcc, CrfLoss
from .models import CRF, gelu_erf, gelu_tanh
    

custom_objects = {
    "MultiHeadAttention": MultiHeadAttention,
    "LayerNormalization": LayerNormalization,
    "PositionEmbedding": PositionEmbedding,
    "FeedForward": FeedForward,
    "EmbeddingDense": EmbeddingDense,
    "ExpandDims": ExpandDims,
    "MultiLossLayer": MultiLossLayer,
    "CrfAcc": CrfAcc,
    "CrfLoss": CrfLoss,
    "CRF": CRF,
    "gelu_erf": gelu_erf,
    "gelu_tanh": gelu_tanh,
    "gelu": gelu_tanh,
}


get_custom_objects().update(custom_objects)