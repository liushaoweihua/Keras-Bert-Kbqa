# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: processor.py
@Time: 2020/3/3 10:37 AM
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import codecs
import numpy as np
from keras.utils import to_categorical
from .tokenizer import Tokenizer


class Processor:

    def __init__(self, data_path, dict_path, tag_padding=None):
        self.data = self._load_data(data_path)
        if tag_padding is not None:
            self.tag_padding = tag_padding
        else:
            self.tag_padding = "X"
        self._load_tags()
        self._load_labels()
        self.tokenizer = Tokenizer(dict_path)

    def _load_data(self, path):
        """加载数据集
        """
        with codecs.open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def _load_tags(self):
        """tag转换为对应id
        """
        tags = set()
        for item in self.data:
            for tag in item[2].split(" "):
                if len(tag) == 1:
                    tags.add(tag)
                else:
                    entity_type = tag.split("-")[-1]
                    tags.add("B-%s" % entity_type)
                    tags.add("I-%s" % entity_type)
                    tags.add("S-%s" % entity_type)
        tags = list(tags)
        self.tag_to_id = {tags[i]: i for i in range(len(tags))}
        if self.tag_padding not in self.tag_to_id:
            self.tag_to_id[self.tag_padding] = len(self.tag_to_id)
        else:
            raise ValueError("tag_padding %s already exists" % self.tag_padding)
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}
        self.numb_tags = len(self.tag_to_id)

    def _load_labels(self):
        """label转换为对应id
        """
        labels = set()
        for item in self.data:
            labels.add(item[1])
        labels = list(labels)
        self.label_to_id = {labels[i]: i for i in range(len(labels))}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.numb_labels = len(self.label_to_id)

    def process(self, path, max_len):
        """适配于Bert/Albert的训练数据生成
        """
        data = self._load_data(path)
        np.random.shuffle(data)
        origin_texts, origin_labels, origin_tags = np.stack(data, axis=-1)
        tokens, segs = [], []
        for text in origin_texts:
            token, seg = self.tokenizer.encode(text, first_length=max_len)
            tokens.append(token)
            segs.append(seg)
        labels = self._norm_labels(origin_labels)
        tags = self._pad_and_truncate(origin_tags, max_len - 2)

        return tokens, segs, labels, tags

    def _norm_labels(self, origin_labels):
        """将label转化为指定数据形式
        """
        return to_categorical([self.label_to_id[i] for i in origin_labels], self.numb_labels)

    def _pad_and_truncate(self, origin_tags, max_len):
        """填充或截断至指定长度
        """
        tags = []
        for tag in origin_tags:
            tag_len = len(tag.split(" "))
            if tag_len >= max_len:
                tags.append([self.tag_padding] +
                            tag.split(" ")[:max_len] +
                            [self.tag_padding])
            else:
                tags.append([self.tag_padding] +
                            tag.split(" ") +
                            [self.tag_padding] * (max_len - tag_len + 1))
        tags = np.expand_dims(
            [[self.tag_to_id[item] for item in term[1:]] for term in tags],
            axis=-1)

        return tags