# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: run_deploy.py
@Time: 2020/3/16 3:33 PM
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
sys.path.append("../..")

import os
import json
import keras
import codecs
import pickle
import numpy as np
import tensorflow as tf
from loguru import logger
from termcolor import colored
from flask import Flask, Response, request
from keras_bert_kbqa.helper import deploy_args_parser
from keras_bert_kbqa.predict import predict
from keras_bert_kbqa.utils.tokenizer import Tokenizer


app = Flask(__name__)
app.model_configs = {}


def log_init(log_path):
    log_file_path = os.path.join(log_path, "info.log")
    err_file_path = os.path.join(log_path, "error.log")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger.add(sys.stderr, format="{time} {level} {message}",
               filter="my_module", level="INFO")
    logger.add(log_file_path, rotation="12:00", retention="14 days",
               encoding="utf-8")
    logger.add(err_file_path, rotation="100 MB", retention="14 days",
               encoding="utf-8", level="ERROR")
    logger.debug("logger initialized")
    return logger


def run_deploy():
    # 基础配置
    args = deploy_args_parser()
    if True:
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_map if args.device_map != "cpu" else ""
    sess = tf.Session()
    graph = tf.get_default_graph()
    keras.backend.set_session(sess)
    # 属性设置
    with codecs.open(args.model_configs, "r", encoding="utf-8") as f:
        model_configs = json.load(f)
    tokenizer = Tokenizer(model_configs.get("bert_vocab"))
    max_len = model_configs.get("max_len")
    tag_padding = model_configs.get("tag_padding")
    with codecs.open(model_configs.get("id_to_label"), "rb") as f:
        id_to_label = pickle.load(f)
    with codecs.open(model_configs.get("id_to_tag"), "rb") as f:
        id_to_tag = pickle.load(f)
    model = predict(model_configs.get("model_path"))
    # 新增属性
    app.model_configs["logger"] = log_init(args.log_path)
    app.model_configs["tokenizer"] = tokenizer
    app.model_configs["max_len"] = max_len
    app.model_configs["tag_padding"] = tag_padding
    app.model_configs["id_to_label"] = id_to_label
    app.model_configs["id_to_tag"] = id_to_tag
    app.model_configs["model"] = model
    with codecs.open(args.prior_checks, "r", encoding="utf-8") as f:
        app.prior_checks = json.load(f)
    with codecs.open(args.utter_search, "r", encoding="utf-8") as f:
        app.utter_search = json.load(f)
    with codecs.open(args.database, "r", encoding="utf-8") as f:
        app.database = json.load(f)
    return graph, sess


graph, sess = run_deploy()


def parse(text):
    # 编码
    text = text[:app.model_configs["max_len"]]
    token, seg = app.model_configs["tokenizer"].encode(text, first_length=app.model_configs["max_len"])
    # 解码
    # 显存不足时注释掉以下代码
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
    global graph, sess
    with graph.as_default():
        keras.backend.set_session(sess)
        clf_pred, ner_pred = app.model_configs["model"].predict([[token], [seg]])
        clf_res = app.model_configs["id_to_label"][np.argmax(clf_pred)]
        ner_pred = [app.model_configs["id_to_tag"][item] for item in np.argmax(ner_pred, axis=-1)[0]]
        ner_res = get_entity_with_check(text, ner_pred)
        subject, response = utter(clf_res, ner_res)
        return json.dumps({
            "text": text[:app.model_configs["max_len"]],
            "predicate": clf_res,
            "subject": subject,
            "response": response},
            ensure_ascii=False, indent=4)


def get_entity_with_check(text, tokens):
    """通过模糊匹配原始数据进行二次确认
    避免小数据集下的CRF转移矩阵训练不准确带来的误识别
    """
    entities = get_entity(text, tokens)
    checked_entities = []
    for entity in entities:
        check_list = app.prior_checks[entity[0]]
        if entity[1] not in check_list:
            # less_list：表示算法获取的实体值字符少于原始值，进行最大匹配，如【周星】->【周星驰】
            # more_list：表示算法获取的实体值字符多于原始值，进行最小匹配，如【周星驰的】->【周星驰】
            less_list, more_list = [], []
            for item in check_list:
                if entity[1] in item:
                    less_list.append(item)
                if item in entity[1]:
                    more_list.append(item)
            # 粗暴做法：直接获取两个list中的最长串，如果有多个就返回首个
            max_match_len = max([len(i) for i in list(set(less_list + more_list))]) if (less_list + more_list) != [] else None
            entity[1] = [i for i in list(set(less_list + more_list)) if len(i) == max_match_len][0] \
                if max_match_len is not None else entity[1]
        checked_entities.append(entity)
    return checked_entities


def get_entity(text, tokens):
    """获取ner结果
    """
    # 如果text长度小于规定的max_len长度，则只保留text长度的tokens
    text_len = len(text)
    tokens = tokens[:text_len]

    entities = []
    entity = ""
    for idx, char, token in zip(range(text_len), text, tokens):
        if token.startswith("O") or token.startswith(app.model_configs["tag_padding"]):
            token_prefix = token
            token_suffix = None
        else:
            token_prefix, token_suffix = token.split("-")
        if token_prefix == "S":
            entities.append([token_suffix, char])
            entity = ""
        elif token_prefix == "B":
            if entity != "":
                entities.append([tokens[idx-1].split("-")[-1], entity])
                entity = ""
            else:
                entity += char
        elif token_prefix == "I":
            if entity != "":
                entity += char
            else:
                entity = ""
        else:
            if entity != "":
                entities.append([tokens[idx-1].split("-")[-1], entity])
                entity = ""
            else:
                continue

    return entities


def utter(clf_res, ner_res):
    subject, response = [], []
    for ner_item in ner_res:
        if ner_item[0] in app.utter_search[clf_res]:
            print(ner_item)
            print(app.database[:3])
            print(app.utter_search[clf_res])
            print(app.utter_search[clf_res].format(ner_item[1]))
            subject.append({ner_item[0]: ner_item[1]})
            response = eval(app.utter_search[clf_res].format(ner_item[1]))
        else:
            response = None
    return subject, response


def first_predict():
    """第一次使用模型时需要加载，否则会降低预测速度
    """
    parse("")


first_predict()


@app.route("/query", methods=["POST"])
def decode():
    app.model_configs["logger"].info(colored("[RECEIVE]: ", "red") + colored(request.json["text"], "cyan"))
    res = parse(request.json["text"])
    app.model_configs["logger"].info(colored("[SEND]: ", "green") + colored(res, "cyan"))
    return res


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2020, debug=True, use_reloader=False)
