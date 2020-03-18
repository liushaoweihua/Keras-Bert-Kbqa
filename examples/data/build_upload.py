#-*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: build_upload.py
@Time: 2020/3/13 5:50 PM
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
sys.path.append("../..")

import os
import json
import codecs
import numpy as np
# from keras_bert_kbqa.utils.graph_builder import NodeCreator, RelationCreator


def generate(sample_size, templates, fills, data_type):
    """填充式数据生成
    """
    results = []
    data = templates.get(data_type)
    for label in data:
        for text_template in data[label]:
            suffix = text_template.split("{")[1].split("}")[0]
            tag_template = "O "*(len(text_template.split(suffix)[0])-1) + "{" + suffix + "}" + " O"*(len(text_template.split(suffix)[1])-1)
            choose_fills = np.random.choice(fills[suffix], sample_size)
            tags = ["B-"+suffix+(" I-"+suffix)*(len(item)-1) if len(item) > 1 else "S-"+suffix for item in choose_fills]
            for choose_fill, tag in zip(choose_fills, tags):
                results.append([text_template.replace("{"+suffix+"}", choose_fill), label, tag_template.replace("{"+suffix+"}", tag)])
    return results


def build(raw_data_path, use_attrs, template_path, save_path, train_sample_size_per_class, dev_sample_size_per_class):
    """建立训练集与测试集
    """
    with codecs.open(raw_data_path, "r", encoding="utf-8") as f:
        origin_data = json.load(f)
    data = []
    for datum in origin_data:
        new_datum = {}
        for attr in use_attrs:
            new_datum[attr] = datum.get(attr) or []
        data.append(new_datum)
    fills = {attr: [] for attr in use_attrs}
    for datum in data:
        for attr in datum:
            if isinstance(datum[attr], list):
                fills[attr].extend(datum[attr])
            else:
                fills[attr].append(str(datum[attr]))
    with codecs.open(template_path, "r", encoding="utf-8") as f:
        templates = json.load(f)
    train_data = generate(train_sample_size_per_class, templates, fills, "train")
    dev_data = generate(dev_sample_size_per_class, templates, fills, "dev")
    save_path = os.path.abspath(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with codecs.open(os.path.join(save_path, "train_data.txt"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with codecs.open(os.path.join(save_path, "dev_data.txt"), "w", encoding="utf-8") as f:
        json.dump(dev_data, f, ensure_ascii=False, indent=4)
    with codecs.open(os.path.join(save_path, "database.txt"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    with codecs.open(os.path.join(save_path, "prior_check.txt"), "w", encoding="utf-8") as f:
        json.dump(fills, f, ensure_ascii=False, indent=4)


# class Uploader:
#     """图数据库内容上传器（有需要的可以用neo4j跑一下试试，预测部分没写）
#     """
#     def __init__(self, graph_config_path, raw_data_path):
#         self.graph = self._read(graph_config_path)
#         self.raw_data = self._read(raw_data_path)
#         self.nodes = {}
#         self.relations = {}
#
#     def _read(self, path):
#         with codecs.open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         return data
#
#     def define_node(self, node_name, key_property, properties):
#         assert isinstance(properties, list), "param `properties` should be type list"
#         nodes = [{item: term[item] for item in properties} for term in self.raw_data]
#         self.nodes[node_name] = {"key_property": key_property, "properties": properties}
#         self._build_node(node_name, nodes, [key_property])
#         return nodes
#
#     def define_relation(self, left_node_name, relation_name, right_node_name, build=True):
#         assert left_node_name in self.nodes, "`left_node_name` not found"
#         assert right_node_name in self.nodes, "`right_node_name` not found"
#         left_key_property = self.nodes[left_node_name]["key_property"]
#         right_key_property = self.nodes[right_node_name]["key_property"]
#         relations = [
#             {
#                 "left_node": {
#                     left_key_property: item[left_key_property]
#                 },
#                 "right_node": {
#                     right_key_property: item[right_key_property]
#                 }
#             } for item in self.raw_data]
#         self.relations[relation_name] = {
#             "left_node_name": left_node_name,
#             "left_key_property": left_key_property,
#             "right_node_name": right_node_name,
#             "right_key_property": right_key_property
#         }
#         if build:
#             self._build_relation(left_node_name, relation_name, right_node_name, relations)
#         return relations
#
#     def define_twin_relations(self, left_node_name, relation_names, right_node_name):
#         assert isinstance(relation_names, list), "`relation_names` should be type list"
#         assert len(relation_names) == 2, "length of `relation_names` should be 2"
#         relations = self.define_relation(left_node_name, relation_names[0], right_node_name, build=False)
#         _ = self.define_relation(right_node_name, relation_names[1], left_node_name, build=False)
#         self._build_relation(left_node_name, relation_names, right_node_name, relations)
#
#     def _build_node(self, node_name, nodes, key_property):
#         NodeCreator(self.graph, node_name)(nodes, indexes=key_property)
#
#     def _build_relation(self, left_node_name, relation_names, right_node_name, relations):
#         RelationCreator(self.graph, left_node_name, relation_names, right_node_name)(relations)


if __name__ == "__main__":
    # build train/dev data
    raw_data_path = "./origin_data/douban_movies.txt"
    use_attrs = ["title", "rate", "actor", "director", "category", "language", "showtime"]
    template_path = "./templates/text_templates.txt"
    save_path = "data"
    train_sample_size_per_class = 50
    dev_sample_size_per_class = 5
    build(raw_data_path, use_attrs, template_path, save_path, train_sample_size_per_class, dev_sample_size_per_class)
    # # upload data to neo4j
    # graph_config_path = "../neo4j_config.txt"
    # uploader = Uploader(graph_config_path, raw_data_path)
    # uploader.define_node(
    #     node_name="豆瓣_影片",
    #     key_property="title",
    #     properties=use_attrs)
    # uploader.define_node(
    #     node_name="豆瓣_导演",
    #     key_property="director",
    #     properties=["director"])
    # uploader.define_node(
    #     node_name="豆瓣_演员",
    #     key_property="actor",
    #     properties=["actor"])
    # uploader.define_twin_relations(
    #     left_node_name="豆瓣_影片",
    #     relation_names=["导演", "拍摄"],
    #     right_node_name="豆瓣_导演")
    # uploader.define_twin_relations(
    #     left_node_name="豆瓣_影片",
    #     relation_names=["演员", "参演"],
    #     right_node_name="豆瓣_演员")
