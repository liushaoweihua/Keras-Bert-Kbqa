# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: bert.py
@Time: 2020/3/3 10:37 AM
"""

# Codes come from <bert4keras>:
#    Author: Jianlin Su
#    Github: https://github.com/bojone/bert4keras
#    Site: kexue.fm
#    Version: 0.2.5


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import keras
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras import initializers, activations
from keras.layers import *
from keras.models import Model
from .tokenizer import is_string


def sequence_masking(x, mask, mode=0, axis=None, heads=1):
    """为序列条件mask的函数
    mask: 形如(batch_size, sequence)的0-1矩阵；
    mode: 如果是0，则直接乘以mask；
          如果是1，则在padding部分减去一个大正数。
    axis: 序列所在轴，默认为1；
    heads: 相当于batch这一维要被重复的次数。
    """
    if mask is None or mode not in [0, 1]:
        return x
    else:
        if heads is not 1:
            mask = K.expand_dims(mask, 1)
            mask = K.tile(mask, (1, heads, 1))
            mask = K.reshape(mask, (-1, K.shape(mask)[2]))
        if axis is None:
            axis = 1
        if axis == -1:
            axis = K.ndim(x) - 1
        assert axis > 0, "axis must be greater than 0"
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        for _ in range(K.ndim(x) - K.ndim(mask) - axis + 1):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 0:
            return x * mask
        else:
            return x - (1 - mask) * 1e12


class MultiHeadAttention(Layer):
    """多头注意力机制
    """

    def __init__(self,
                 heads,
                 head_size,
                 key_size=None,
                 kernel_initializer="glorot_uniform",
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = heads * head_size
        self.key_size = key_size if key_size else head_size
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(units=self.key_size * self.heads,
                             kernel_initializer=self.kernel_initializer)
        self.k_dense = Dense(units=self.key_size * self.heads,
                             kernel_initializer=self.kernel_initializer)
        self.v_dense = Dense(units=self.out_dim,
                             kernel_initializer=self.kernel_initializer)
        self.o_dense = Dense(units=self.out_dim,
                             kernel_initializer=self.kernel_initializer)

    def call(self, inputs, q_mask=False, v_mask=False, a_mask=False):
        """实现多头注意力
        q_mask: 对输入的query序列的mask。
                主要是将输出结果的padding部分置0。
        v_mask: 对输入的value序列的mask。
                主要是防止attention读取到padding信息。
        a_mask: 对attention矩阵的mask。
                不同的attention mask对应不同的应用。
        """
        q, k, v = inputs[:3]
        # 处理mask
        idx = 3
        if q_mask:
            q_mask = inputs[idx]
            idx += 1
        else:
            q_mask = None
        if v_mask:
            v_mask = inputs[idx]
            idx += 1
        else:
            v_mask = None
        if a_mask:
            if len(inputs) > idx:
                a_mask = inputs[idx]
            else:
                a_mask = "history_only"
        else:
            a_mask = None
        # 线性变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # 转为三阶张量
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.key_size))
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.head_size))
        # Attention
        a = K.batch_dot(qw, kw, [2, 2]) / self.key_size ** 0.5
        a = sequence_masking(a, v_mask, 1, -1, self.heads)
        if a_mask is not None:
            if is_string(a_mask) and a_mask == "history_only":
                ones = K.ones_like(a[:1])
                a_mask = (ones - tf.linalg.band_part(ones, -1, 0)) * 1e12
                a = a - a_mask
            else:
                a = a - (1 - a_mask) * 1e12
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [2, 1])
        o = K.reshape(o, (-1, self.heads, K.shape(q)[1], self.head_size))
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.o_dense(o)
        o = sequence_masking(o, q_mask, 0)
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)

    def get_config(self):
        config = {
            "heads": self.heads,
            "head_size": self.head_size,
            "key_size": self.key_size,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LayerNormalization(Layer):
    """实现基本的Layer Norm，只保留核心运算部分
    """

    def __init__(self, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = K.epsilon() * K.epsilon()

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)
        shape = (input_shape[-1],)
        self.gamma = self.add_weight(shape=shape,
                                     initializer="ones",
                                     name="gamma")
        self.beta = self.add_weight(shape=shape,
                                    initializer="zeros",
                                    name="beta")

    def call(self, inputs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        outputs *= self.gamma
        outputs += self.beta
        return outputs


class PositionEmbedding(Layer):
    """定义位置Embedding，这里的Embedding是可训练的。
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 merge_mode="add",
                 embeddings_initializer="zeros",
                 **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.embeddings_initializer = initializers.get(embeddings_initializer)

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
        )

    def call(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        pos_embeddings = self.embeddings[:seq_len]
        pos_embeddings = K.expand_dims(pos_embeddings, 0)
        pos_embeddings = K.tile(pos_embeddings, [batch_size, 1, 1])
        if self.merge_mode == "add":
            return inputs + pos_embeddings
        else:
            return K.concatenate([inputs, pos_embeddings])

    def compute_output_shape(self, input_shape):
        if self.merge_mode == "add":
            return input_shape
        else:
            return input_shape[:2] + (input_shape[2] + self.v_dim,)

    def get_config(self):
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "merge_mode": self.merge_mode,
            "embeddings_initializer": initializers.serialize(self.embeddings_initializer),
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FeedForward(Layer):
    """FeedForward层，其实就是两个Dense层的叠加
    """

    def __init__(self,
                 units,
                 activation="relu",
                 kernel_initializer="glorot_uniform",
                 **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]
        self.dense_1 = Dense(units=self.units,
                             activation=self.activation,
                             kernel_initializer=self.kernel_initializer)
        self.dense_2 = Dense(units=output_dim,
                             kernel_initializer=self.kernel_initializer)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x

    def get_config(self):
        config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EmbeddingDense(Layer):
    """运算跟Dense一致，但kernel用Embedding层的embeddings矩阵。
    根据Embedding层的名字来搜索定位Embedding层。
    """

    def __init__(self, embedding_name, activation="softmax", **kwargs):
        super(EmbeddingDense, self).__init__(**kwargs)
        self.embedding_name = embedding_name
        self.activation = activations.get(activation)

    def call(self, inputs):
        if not hasattr(self, "kernel"):
            embedding_layer = inputs._keras_history[0]

            if embedding_layer.name != self.embedding_name:

                def recursive_search(layer):
                    """递归向上搜索，根据名字找Embedding层
                    """
                    last_layer = layer._inbound_nodes[0].inbound_layers
                    if isinstance(last_layer, list):
                        if len(last_layer) == 0:
                            return None
                        else:
                            last_layer = last_layer[0]
                    if last_layer.name == self.embedding_name:
                        return last_layer
                    else:
                        return recursive_search(last_layer)

                embedding_layer = recursive_search(embedding_layer)
                if embedding_layer is None:
                    raise Exception("Embedding layer not found")

                self.kernel = K.transpose(embedding_layer.embeddings)
                self.units = K.int_shape(self.kernel)[1]
                self.bias = self.add_weight(name="bias",
                                            shape=(self.units,),
                                            initializer="zeros")

        outputs = K.dot(inputs, self.kernel)
        outputs = K.bias_add(outputs, self.bias)
        outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

    def get_config(self):
        config = {
            "embedding_name": self.embedding_name,
            "activation": activations.serialize(self.activation),
        }
        base_config = super(EmbeddingDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BertModel:
    """构建跟Bert一样结构的Transformer-based模型
    这是一个比较多接口的基础类，然后通过这个基础类衍生出更复杂的模型
    """

    def __init__(self,
                 vocab_size,  # 词表大小
                 max_position_embeddings,  # 序列最大长度
                 hidden_size,  # 编码维度
                 num_hidden_layers,  # Transformer总层数
                 num_attention_heads,  # Attention的头数
                 intermediate_size,  # FeedForward的隐层维度
                 hidden_act,  # FeedForward隐层的激活函数
                 dropout_rate,  # Dropout比例
                 initializer_range=None,  # 权重初始化方差
                 embedding_size=None,  # 是否指定embedding_size
                 with_pool=False,  # 是否包含Pool部分
                 with_nsp=False,  # 是否包含NSP部分
                 with_mlm=False,  # 是否包含MLM部分
                 keep_words=None,  # 要保留的词ID列表
                 block_sharing=False,  # 是否共享同一个transformer block
                 ):
        if keep_words is None:
            self.vocab_size = vocab_size
        else:
            self.vocab_size = len(keep_words)
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate
        if initializer_range:
            self.initializer_range = initializer_range
        else:
            self.initializer_range = 0.02
        if embedding_size:
            self.embedding_size = embedding_size
        else:
            self.embedding_size = hidden_size
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.hidden_act = hidden_act
        self.keep_words = keep_words
        self.block_sharing = block_sharing
        self.additional_outputs = []

    def build(self):
        """Bert模型构建函数
        """
        x_in = Input(shape=(None,), name="Input-Token")
        s_in = Input(shape=(None,), name="Input-Segment")
        x, s = x_in, s_in

        # 自行构建Mask
        sequence_mask = Lambda(lambda x: K.cast(K.greater(x, 0), "float32"),
                               name="Sequence-Mask")(x)

        # Embedding部分
        x = Embedding(input_dim=self.vocab_size,
                      output_dim=self.embedding_size,
                      embeddings_initializer=self.initializer,
                      name="Embedding-Token")(x)
        s = Embedding(input_dim=2,
                      output_dim=self.embedding_size,
                      embeddings_initializer=self.initializer,
                      name="Embedding-Segment")(s)
        x = Add(name="Embedding-Token-Segment")([x, s])
        x = PositionEmbedding(input_dim=self.max_position_embeddings,
                              output_dim=self.embedding_size,
                              merge_mode="add",
                              embeddings_initializer=self.initializer,
                              name="Embedding-Position")(x)
        x = LayerNormalization(name="Embedding-Norm")(x)
        if self.dropout_rate > 0:
            x = Dropout(rate=self.dropout_rate, name="Embedding-Dropout")(x)
        if self.embedding_size != self.hidden_size:
            x = Dense(units=self.hidden_size,
                      kernel_initializer=self.initializer,
                      name="Embedding-Mapping")(x)

        # 主要Transformer部分
        layers = None
        for i in range(self.num_hidden_layers):
            attention_name = "Encoder-%d-MultiHeadSelfAttention" % (i + 1)
            feed_forward_name = "Encoder-%d-FeedForward" % (i + 1)
            x, layers = self.transformer_block(
                inputs=x,
                sequence_mask=sequence_mask,
                attention_mask=self.compute_attention_mask(i, s_in),
                attention_name=attention_name,
                feed_forward_name=feed_forward_name,
                input_layers=layers)
            x = self.post_processing(i, x)
            if not self.block_sharing:
                layers = None

        outputs = [x]

        if self.with_pool:
            # Pooler部分（提取CLS向量）
            x = outputs[0]
            x = Lambda(lambda x: x[:, 0], name="Pooler")(x)
            x = Dense(units=self.hidden_size,
                      activation="tanh",
                      kernel_initializer=self.initializer,
                      name="Pooler-Dense")(x)
            if self.with_nsp:
                # Next Sentence Prediction 部分
                x = Dense(units=2,
                          activation="softmax",
                          kernel_initializer=self.initializer,
                          name="NSP-Proba")(x)
                outputs.append(x)

        if self.with_mlm:
            # Masked Language Model 部分
            x = outputs[0]
            x = Dense(units=self.embedding_size,
                      activation=self.hidden_act,
                      kernel_initializer=self.initializer,
                      name="MLM-Dense")(x)
            x = LayerNormalization(name="MLM-Norm")(x)
            x = EmbeddingDense(embedding_name="Embedding-Token",
                               name="MLM-Proba")(x)
            outputs.append(x)

        outputs += self.additional_outputs
        if len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) == 2:
            outputs = outputs[1]
        else:
            outputs = outputs[1:]

        self.model = Model([x_in, s_in], outputs)

    def transformer_block(self,
                          inputs,
                          sequence_mask,
                          attention_mask=None,
                          attention_name="attention",
                          feed_forward_name="feed-forword",
                          input_layers=None):
        """构建单个Transformer Block
        如果没有传入input_layers则新建层；如果传入则重用旧层"""
        x = inputs
        if input_layers is None:
            layers = [
                MultiHeadAttention(heads=self.num_attention_heads,
                                   head_size=self.attention_head_size,
                                   kernel_initializer=self.initializer,
                                   name=attention_name),
                Dropout(rate=self.dropout_rate,
                        name="%s-Dropout" % attention_name),
                Add(name="%s-Add" % attention_name),
                LayerNormalization(name="%s-Norm" % attention_name),
                FeedForward(units=self.intermediate_size,
                            activation=self.hidden_act,
                            kernel_initializer=self.initializer,
                            name=feed_forward_name),
                Dropout(rate=self.dropout_rate,
                        name="%s-Dropout" % feed_forward_name),
                Add(name="%s-Add" % feed_forward_name),
                LayerNormalization(name="%s-Norm" % feed_forward_name)
            ]
        else:
            layers = input_layers
        # Self Attention
        xi = x
        if attention_mask is None:
            x = layers[0]([x, x, x, sequence_mask], v_mask=True)
        elif attention_mask is "history_only":
            x = layers[0]([x, x, x, sequence_mask], v_mask=True)
        else:
            x = layers[0]([x, x, x, sequence_mask, attention_mask],
                          v_mask=True,
                          a_mask=True)
        if self.dropout_rate > 0:
            x = layers[1](x)
        x = layers[2]([xi, x])
        x = layers[3](x)
        # Feed Forward
        xi = xx = layers[4](x)
        if self.dropout_rate > 0:
            x = layers[5](x)
        x = layers[6]([xi, x])
        x = layers[7](x)
        return x, layers

    def compute_attention_mask(self, layer_id, segment_ids):
        """定义每一层的Attention Mask，来实现不同的功能
        """
        return None

    def post_processing(self, layer_id, inputs):
        """自定义每一个block的后处理操作
        """
        return inputs

    @property
    def initializer(self):
        """默认使用截断正态分布初始化
        """
        return keras.initializers.TruncatedNormal(
            stddev=self.initializer_range)

    def load_weights_from_checkpoint(self, checkpoint_file):
        """从预训练好的Bert的checkpoint中加载权重
        为了简化写法，对变量名的匹配引入了一定的模糊匹配能力。
        """
        model = self.model
        load_variable = lambda name: tf.train.load_variable(checkpoint_file, name)
        variable_names = [n[0] for n in tf.train.list_variables(checkpoint_file)]
        variable_names = [n for n in variable_names if "adam" not in n]

        def similarity(a, b, n=4):
            # 基于n-grams的jaccard相似度
            a = set([a[i: i + n] for i in range(len(a) - n)])
            b = set([b[i: i + n] for i in range(len(b) - n)])
            a_and_b = a & b
            if not a_and_b:
                return 0.
            a_or_b = a | b
            return 1. * len(a_and_b) / len(a_or_b)

        def loader(name):
            sims = [similarity(name, n) for n in variable_names]
            found_name = variable_names.pop(np.argmax(sims))
            print("==> searching: %s, found name: %s" % (name, found_name))
            return load_variable(found_name)

        if self.keep_words is None:
            keep_words = slice(0, None)
        else:
            keep_words = self.keep_words

        model.get_layer(name="Embedding-Token").set_weights(
            [loader("bert/embeddings/word_embeddings")[keep_words]])
        model.get_layer(name="Embedding-Position").set_weights(
            [loader("bert/embeddings/position_embeddings")])
        model.get_layer(name="Embedding-Segment").set_weights(
            [loader("bert/embeddings/token_type_embeddings")])
        model.get_layer(name="Embedding-Norm").set_weights(
            [loader("bert/embeddings/LayerNorm/gamma"),
             loader("bert/embeddings/LayerNorm/beta")])
        if self.embedding_size != self.hidden_size:
            model.get_layer(name="Embedding-Mapping").set_weights(
                [loader("bert/encoder/embedding_hidden_mapping_in/kernel"),
                 loader("bert/encoder/embedding_hidden_mapping_in/bias")])

        for i in range(self.num_hidden_layers):
            try:
                model.get_layer(name="Encoder-%d-MultiHeadSelfAttention" % (i + 1))
            except ValueError:
                continue
            if ("bert/encoder/layer_%d/attention/self/query/kernel" % i) in variable_names:
                layer_name = "layer_%d" % i
            else:
                layer_name = "transformer/group_0/inner_group_0"
            model.get_layer(name="Encoder-%d-MultiHeadSelfAttention" % (i + 1)).set_weights(
                [loader("bert/encoder/%s/attention/self/query/kernel" % layer_name),
                 loader("bert/encoder/%s/attention/self/query/bias" % layer_name),
                 loader("bert/encoder/%s/attention/self/key/kernel" % layer_name),
                 loader("bert/encoder/%s/attention/self/key/bias" % layer_name),
                 loader("bert/encoder/%s/attention/self/value/kernel" % layer_name),
                 loader("bert/encoder/%s/attention/self/value/bias" % layer_name),
                 loader("bert/encoder/%s/attention/output/dense/kernel" % layer_name),
                 loader("bert/encoder/%s/attention/output/dense/bias" % layer_name)])
            model.get_layer(name="Encoder-%d-MultiHeadSelfAttention-Norm" % (i + 1)).set_weights(
                [loader("bert/encoder/%s/attention/output/LayerNorm/gamma" % layer_name),
                 loader("bert/encoder/%s/attention/output/LayerNorm/beta" % layer_name)])
            model.get_layer(name="Encoder-%d-FeedForward" % (i + 1)).set_weights(
                [loader("bert/encoder/%s/intermediate/dense/kernel" % layer_name),
                 loader("bert/encoder/%s/intermediate/dense/bias" % layer_name),
                 loader("bert/encoder/%s/output/dense/kernel" % layer_name),
                 loader("bert/encoder/%s/output/dense/bias" % layer_name)])
            model.get_layer(name="Encoder-%d-FeedForward-Norm" % (i + 1)).set_weights(
                [loader("bert/encoder/%s/output/LayerNorm/gamma" % layer_name),
                 loader("bert/encoder/%s/output/LayerNorm/beta" % layer_name)])

        if self.with_pool:
            model.get_layer(name="Pooler-Dense").set_weights(
                [loader("bert/pooler/dense/kernel"),
                 loader("bert/pooler/dense/bias")])
            if self.with_nsp:
                model.get_layer(name="NSP-Proba").set_weights(
                    [loader("cls/seq_relationship/output_weights").T,
                     loader("cls/seq_relationship/output_bias")])

        if self.with_mlm:
            model.get_layer(name="MLM-Dense").set_weights(
                [loader("cls/predictions/transform/dense/kernel"),
                 loader("cls/predictions/transform/dense/bias")])
            model.get_layer(name="MLM-Norm").set_weights(
                [loader("cls/predictions/transform/LayerNorm/gamma"),
                 loader("cls/predictions/transform/LayerNorm/beta")])
            model.get_layer(name="MLM-Proba").set_weights(
                [loader("cls/predictions/output_bias")[keep_words]])


class Bert4Seq2seq(BertModel):
    """用来做seq2seq任务的Bert
    """

    def __init__(self, *args, **kwargs):
        super(Bert4Seq2seq, self).__init__(*args, **kwargs)
        self.with_pool = False
        self.with_nsp = False
        self.with_mlm = True
        self.attention_mask = None

    def compute_attention_mask(self, layer_id, segment_ids):
        """为seq2seq采用特定的attention mask
        """
        if self.attention_mask is None:
            def seq2seq_attention_mask(s, repeats=1):
                seq_len = K.shape(s)[1]
                ones = K.ones((1, repeats, seq_len, seq_len))
                a_mask = tf.linalg.band_part(ones, -1, 0)
                s_ex12 = K.expand_dims(K.expand_dims(s, 1), 2)
                s_ex13 = K.expand_dims(K.expand_dims(s, 1), 3)
                a_mask = (1 - s_ex13) * (1 - s_ex12) + s_ex13 * a_mask
                a_mask = K.reshape(a_mask, (-1, seq_len, seq_len))
                return a_mask

            self.attention_mask = Lambda(
                seq2seq_attention_mask,
                arguments={"repeats": self.num_attention_heads},
                name="Attention-Mask")(segment_ids)

        return self.attention_mask


class Bert4LM(BertModel):
    """用来做语言模型任务的Bert
    """

    def __init__(self, *args, **kwargs):
        super(Bert4LM, self).__init__(*args, **kwargs)
        self.with_pool = False
        self.with_nsp = False
        self.with_mlm = True
        self.attention_mask = "history_only"

    def compute_attention_mask(self, layer_id, segment_ids):
        return self.attention_mask


def build_bert_model(config_path,
                     checkpoint_path=None,
                     with_pool=False,
                     with_nsp=False,
                     with_mlm=False,
                     application="encoder",
                     keep_words=None,
                     albert=False,
                     return_keras_model=True):
    """根据配置文件构建bert模型，可选加载checkpoint权重
    """
    config = json.load(open(config_path))
    mapping = {
        "encoder": BertModel,
        "seq2seq": Bert4Seq2seq,
        "lm": Bert4LM
    }

    assert application in mapping, "application must be one of %s" % list(mapping.keys())
    Bert = mapping[application]

    bert = Bert(vocab_size=config["vocab_size"],
                max_position_embeddings=config["max_position_embeddings"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_hidden_layers"],
                num_attention_heads=config["num_attention_heads"],
                intermediate_size=config["intermediate_size"],
                hidden_act=config["hidden_act"],
                dropout_rate=config["hidden_dropout_prob"],
                initializer_range=config.get("initializer_range"),
                embedding_size=config.get("embedding_size"),
                with_pool=with_pool,
                with_nsp=with_nsp,
                with_mlm=with_mlm,
                keep_words=keep_words,
                block_sharing=albert)

    bert.build()

    if checkpoint_path is not None:
        bert.load_weights_from_checkpoint(checkpoint_path)

    if return_keras_model:
        return bert.model
    else:
        return bert