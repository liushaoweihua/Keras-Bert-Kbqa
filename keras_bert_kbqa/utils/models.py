# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: models.py
@Time: 2020/3/9 2:43 PM
"""

# Some codes come from <bert4keras>:
#    Author: Jianlin Su
#    Github: https://github.com/bojone/bert4keras
#    Site: kexue.fm
#    Version: 0.2.5


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import keras
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras import Model
from keras.initializers import Constant
from keras_contrib.layers import CRF
from .bert import *
from .metrics import CrfAcc, CrfLoss


def set_gelu(version):
    """设置gelu版本
    """
    version = version.lower()
    assert version in ["erf", "tanh"], "gelu version must be erf or tanh"
    if version == "erf":
        keras.utils.get_custom_objects()["gelu"] = gelu_erf
    else:
        keras.utils.get_custom_objects()["gelu"] = gelu_tanh


def gelu_erf(x):
    """基于Erf直接计算的gelu函数
    """
    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))


def gelu_tanh(x):
    """基于Tanh近似计算的gelu函数
    """
    cdf = 0.5 * (1.0 + K.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3)))))
    return x * cdf


set_gelu("tanh")


class KbqaModel:
    """Bert Kbqa模型基础类
    """
    def __init__(self,
                 bert_config,
                 bert_checkpoint,
                 albert,
                 clf_configs,
                 ner_configs,
                 max_len,
                 numb_labels,
                 numb_tags,
                 tag_to_id,
                 tag_padding):
        super(KbqaModel, self).__init__()
        self._build_bert_model(bert_config, bert_checkpoint, albert)
        self.clf_configs = clf_configs
        self.ner_configs = ner_configs
        self.max_len = max_len
        self.numb_labels = numb_labels
        self.numb_tags = numb_tags
        self.tag_to_id = tag_to_id
        self.tag_padding = tag_padding
        
    def _build_bert_model(self, bert_config, bert_checkpoint, albert):
        self.bert_model = build_bert_model(
            bert_config,
            bert_checkpoint,
            albert=albert)
        for layer in self.bert_model.layers:
            layer.trainable = True
            
    def train_all_tasks(self):
        """是否开启所有冻结的层进行训练
        """
        for layer in self.full_train_model.layers:
            layer.trainable = True        

    def build(self):
        """Kbqa模型
        """
        # 1. Embeddings层建立
        x_in = Input(shape=(self.max_len,), name="Origin-Input-Token")
        s_in = Input(shape=(self.max_len,), name="Origin-Input-Segment")
        x = self.bert_model([x_in, s_in])
        clf_x = Lambda(lambda X: X[:, 0], name="Clf-Embedding")(x)
        ner_x = Lambda(lambda X: X[:, 1:], name="Ner-Embedding")(x)

        # 2. 下游网络层建立
        # clf任务
        clf_o = self.clf(clf_x)
        # ner任务，添加了clf_o的先验信息
        ner_o = self.ner(ner_x, clf_o)

        # 3. 返回训练模型与预测模型        
        clf_true = Input(shape=(self.numb_labels,), name="Clf-True")
        ner_true = Input(shape=(self.max_len - 1, 1), name="Ner-True")
        clf_loss = K.categorical_crossentropy(clf_true, clf_o)
        clf_acc = keras.metrics.categorical_accuracy(clf_true, clf_o)
        ner_loss = CrfLoss(self.tag_to_id, self.tag_padding).crf_loss(ner_true, ner_o)
        ner_acc = CrfAcc(self.tag_to_id, self.tag_padding).crf_accuracy(ner_true, ner_o)
        train_o = MultiLossLayer(self.tag_to_id, self.tag_padding)([clf_true, ner_true, clf_o, ner_o])
        # 较难任务训练模型
        self.hard_train_model = Model([x_in, s_in], ner_o)
        # 全训练模型
        self.full_train_model = Model([x_in, s_in, clf_true, ner_true], train_o)
        self.full_train_model.add_metric(clf_loss, name="clf_loss")
        self.full_train_model.add_metric(clf_acc, name="clf_acc")
        self.full_train_model.add_metric(ner_loss, name="ner_loss")
        self.full_train_model.add_metric(ner_acc, name="ner_acc")
        self.full_train_model.train_all_tasks = self.train_all_tasks
        # 预测模型
        self.pred_model = Model([x_in, s_in], [clf_o, ner_o])
        
    def clf(self, clf_x):
        """分类任务模型，由于上游模型已经采用bert，所以下游模型尽可能简单
        简单给了TextCNN和Dense两类样例
        """
        # 配置解析
        clf_type = self.clf_configs.get("clf_type").lower()
        assert clf_type in ["textcnn", "dense"], "clf_type should be 'textcnn' or 'dense'"
        dropout_rate = self.clf_configs.get("dropout_rate")
        dense_units = self.clf_configs.get("dense_units")
        
        # clf模型定义
        def textcnn(clf_x):
            clf_x = Lambda(lambda X: K.expand_dims(X, axis=-1))(clf_x)
            clf_pool_output = []
            for kernel_size in self.clf_configs.get("kernels"):
                clf_conv = Conv1D(filters=self.clf_configs.get("filters"), kernel_size=kernel_size, strides=1, padding="same",
                                  activation="relu", name="Clf-Conv-%s" % kernel_size, trainable=False)(clf_x)
                clf_pool = MaxPooling1D(name="Clf-MaxPooling-%s" % kernel_size)(clf_conv)
                clf_pool_output.append(clf_pool)
            clf_o = concatenate(clf_pool_output)
            clf_o = Dropout(self.clf_configs.get("dropout_rate"), name="Clf-Dropout", trainable=False)(clf_o)
            clf_o = Flatten(name="Clf-Flatten")(clf_o)
            clf_o = Dense(self.clf_configs.get("dense_units"), activation="relu", name="Clf-Dense-In", trainable=False)(clf_o)
            clf_o = Dense(self.numb_labels, activation="softmax", name="Clf-Dense-Out", trainable=False)(clf_o)
            return clf_o
        
        def dense(clf_x):
            clf_o = Dense(self.clf_configs.get("dense_units"), activation="relu", name="Clf-Dense-In", trainable=False)(clf_x)
            clf_o = Dense(self.numb_labels, activation="softmax", name="Clf-Dense-Out", trainable=False)(clf_o)
            return clf_o
        
        # 模型构建
        if clf_type == "textcnn":
            clf_o = textcnn(clf_x)
        else:
            clf_o = dense(clf_x)
        return clf_o
    
    def ner(self, ner_x, clf_o):
        """序列标注任务模型，需要添加clf_o的先验信息
        给了Idcnn和Bilstm两类样例
        """
        # 配置解析
        ner_type = self.ner_configs.get("ner_type").lower()
        assert ner_type in ["idcnn", "bilstm"], "ner_type should be 'idcnn' or 'bilstm'"
        
        # ner模型定义
        def idcnn(ner_x):
            def dilation_conv1d(dilation_rate, name):
                return Conv1D(self.ner_configs.get("filters"), self.ner_configs.get("kernel_size"), padding="same", dilation_rate=dilation_rate, name=name)

            def idcnn_block(name):
                return [dilation_conv1d(1, name + "1"), dilation_conv1d(1, name + "2"), dilation_conv1d(2, name + "3")]

            ner_o = []
            for layer_idx in range(self.ner_configs.get("blocks")):
                name = "Idcnn-Block-%s-Layer-" % layer_idx
                idcnns = idcnn_block(name)
                cnn = idcnns[0](ner_x)
                cnn = idcnns[1](cnn)
                cnn = idcnns[2](cnn)
                ner_o.append(cnn)
            ner_o = concatenate(ner_o, axis=-1)
            return ner_o
        
        def bilstm(ner_x):
            for layer_idx in range(self.ner_configs.get("num_hidden_layers")):
                name = "Bilstm-Layer-%s" % layer_idx
                ner_x = Bidirectional(LSTM(units=self.ner_configs.get("units"), return_sequences=True, recurrent_dropout=self.ner_configs.get("dropout_rate")),
                                      name=name)(ner_x)
            return ner_x
        
        # 模型构建
        clf_o = ExpandDims(self.max_len-1, name="Clf-Prior")(clf_o)
        ner_x = Concatenate(name="Ner-Clf-Joint")([ner_x, clf_o])
        if ner_type == "idcnn":
            ner_o = idcnn(ner_x)
        else:
            ner_o = bilstm(ner_x)
        ner_o = CRF(self.numb_tags, sparse_target=True, name="Ner-CRF")(ner_o)
        return ner_o
    

class ExpandDims(Layer):
    """需要写成自定义层的形式而不能用Lambda函数写，在保存时会出错
    """
    def __init__(self, max_len, **kwargs):
        self.max_len = max_len
        super(ExpandDims, self).__init__(**kwargs)
        
    def call(self, inputs):
        return K.tile(K.expand_dims(inputs, 1), [1, self.max_len, 1])
    
    def get_config(self):
        config = {
            "max_len": self.max_len}
        base_config = super(ExpandDims, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.max_len, input_shape[1])
    
    
class MultiLossLayer(Layer):
    """以下论文提出方法的魔改
    Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
    paper: http://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html
    Keras code: https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example.ipynb
    """
    def __init__(self, tag_to_id, tag_padding):
        self.tag_to_id = tag_to_id
        self.tag_padding = tag_padding
        self.loss_funcs = [K.categorical_crossentropy, CrfLoss(tag_to_id, tag_padding).crf_loss]
        super(MultiLossLayer, self).__init__()

    def build(self, input_shape=None):
        self.log_vars = []
        for i in range(2):
            self.log_vars += [
                self.add_weight(name="log_var" + str(i), shape=(1,), initializer=Constant(0.), trainable=False)]
        super(MultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        loss = 0
        for y_true, y_pred, loss_func, log_var in zip(ys_true, ys_pred, self.loss_funcs, self.log_vars):
            precision = K.exp(-log_var[0])
            loss += K.sum(precision * loss_func(y_true, y_pred) + log_var[0], -1)
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:2]
        ys_pred = inputs[2:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        return inputs

    def get_config(self):
        config = {
            "tag_to_id": self.tag_to_id,
            "tag_padding": self.tag_padding}
        base_config = super(MultiLossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))