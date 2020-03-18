# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: decoder.py
@Time: 2020/3/9 1:46 PM
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


class Viterbi:

    def __init__(self, model, numb_tags):
        self.model = model
        self.numb_tags = numb_tags
        self._get_crf_trans()

    def _get_crf_trans(self):
        """CRF转移矩阵
        """
        self.crf_trans = {}
        crf_weights = self.model.layers[-1].get_weights()[0]
        for i in range(self.numb_tags):
            for j in range(self.numb_tags):
                self.crf_trans[str(i) + "-" + str(j)] = crf_weights[i, j]

    def _viterbi(self, nodes):
        """生成路径表
        """
        paths = nodes[0]
        for l in range(1, len(nodes)):
            paths_old, paths = paths, {}
            for n, ns in nodes[l].items():
                max_path, max_score = "", -1e10
                for p, ps in paths_old.items():
                    score = ns + ps + self.crf_trans[p.split("-")[-1] + "-" + str(n)]
                    if score > max_score:
                        max_path, max_score = p + "-" + n, score
                paths[max_path] = max_score

        return self._max_in_dict(paths)

    def _max_in_dict(self, paths):
        """获取路径表中的最大值
        """
        paths_inv = {v: k for k, v in paths.items()}

        return paths_inv[max(paths_inv)]

    def decode(self, data):
        """解码过程
        """
        preds = np.array(self.model.predict(data))
        decodes = []
        for pred in preds:
            nodes = [dict([[str(idx), item] for idx, item in enumerate(term)]) for term in pred]
            decodes.append([int(item) for item in self._viterbi(nodes).split("-")])

        return np.array(decodes)