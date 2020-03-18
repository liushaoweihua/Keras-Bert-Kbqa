# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: graph_builder.py
@Time: 2020/3/9 1:58 PM
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import codecs
from copy import deepcopy
from py2neo import Graph, Node, Relationship, Subgraph, NodeMatcher, Schema


def from_json(path, use_properties=None):
    with codecs.open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if use_properties:
        data = [{prop:datum[prop] for prop in use_properties} for datum in data]
    return data


class Creator:

    def __init__(self, graph):
        self.graph = Graph(graph.get("host"), auth=graph.get("auth"))

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("__call__ function not defined.")

    def _create(self):
        raise NotImplementedError("_create function not defined.")


class NodeCreator(Creator):

    def __init__(self, labels):
        super(NodeCreator, self).__init__()
        self.labels = labels
        self.node_template = Node(labels)
        self.schema = Schema(self.graph)

    def __call__(self, data, indexes=None, *args, **kwargs):
        assert isinstance(data, list), "except data to be list, but got %s" % type(data)
        nodes = []
        for datum in data:
            new_node = deepcopy(self.node_template)
            for attr in datum:
                new_node[attr] = datum[attr]
            nodes.append(new_node)
        nodes = Subgraph(nodes)
        self._create(nodes, indexes)

    def _create(self, nodes, indexes):
        self.graph.create(nodes)
        if indexes:
            self.schema.create_index(self.labels, *indexes)


class RelationCreator(Creator):
    """Hint：Py2neo的matcher太慢了，原生的cypher语句也很慢，最好用neo4j的import工具
    Reference：https://www.zhihu.com/question/45401120?sort=created
    """
    def __init__(self, left_node_label, relations, right_node_label):
        super(RelationCreator, self).__init__()
        self.matcher = NodeMatcher(self.graph)
        if isinstance(relations, str):
            relations = [relations]
        assert 1 <= len(relations) <= 2, "except len(relations) to be either 1 or 2, but got %s" % len(relations)
        self.relations = relations
        self.relation_type = "unidirectional" if len(relations) == 1 else "bidirectional"
        self.left_node_label = left_node_label
        self.right_node_label = right_node_label

    def _node_searcher(self, *label, **properties):
        return self.matcher.match(*label, **properties).__iter__()

    def __call__(self, data, *args, **kwargs):
        assert isinstance(data, list), "except data to be list, but got %s" % type(data)
        relations = []
        for datum in data:
            left_node = self._node_searcher(self.left_node_label, **datum["left_node"])
            right_node = self._node_searcher(self.right_node_label, **datum["right_node"])
            relations.extend([Relationship(l, self.relations[0], r) for l in left_node for r in right_node])
            if self.relation_type == "bidirectional":
                relations.extend([Relationship(r, self.relations[1], l) for l in left_node for r in right_node])
        relations = Subgraph(relations)
        self._create(relations)

    def _create(self, relations):
        self.graph.create(relations)