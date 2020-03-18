# -*- coding: utf-8 -*-

"""
@Author: Shaoweihua.Liu
@Contact: liushaoweihua@126.com
@Site: github.com/liushaoweihua
@File: train.py
@Time: 2020/3/9 5:58 PM
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import json
import keras
import codecs
import pickle
import numpy as np
from .utils.processor import Processor
from .utils.models import KbqaModel
from .utils.callbacks import KbqaCallbacks, TaskSwitch
from .utils.metrics import CrfAcc, CrfLoss


def train(args):
    """模型训练流程
    """
    # 环境设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_map if args.device_map != "cpu" else ""
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with codecs.open(args.config, "r", encoding="utf-8") as f:
        args.config = json.load(f)
    args.data_params = args.config.get("data")
    args.bert_params = args.config.get("bert")
    args.model_params = args.config.get("model")
    # 数据准备
    processor = Processor(args.data_params.get("train_data"), args.bert_params.get("bert_vocab"), args.data_params.get("tag_padding"))
    train_tokens, train_segs, train_labels, train_tags = processor.process(args.data_params.get("train_data"), args.data_params.get("max_len"))
    train_x = [np.array(train_tokens), np.array(train_segs), np.array(train_labels), np.array(train_tags)]
    train_y = None
    if args.data_params.get("dev_data") is not None:
        dev_tokens, dev_segs, dev_labels, dev_tags = processor.process(args.data_params.get("dev_data"), args.data_params.get("max_len"))
        devs = [[np.array(dev_tokens), np.array(dev_segs), np.array(dev_labels), np.array(dev_tags)], None]
    else:
        devs = None
    # 模型准备
    model = KbqaModel(
        bert_config=args.bert_params.get("bert_config"),
        bert_checkpoint=args.bert_params.get("bert_checkpoint"),
        albert=args.bert_params.get("albert"),
        clf_configs=args.model_params.get("clf_configs"),
        ner_configs=args.model_params.get("ner_configs"),
        max_len=args.data_params.get("max_len"),
        numb_labels=processor.numb_labels,
        numb_tags=processor.numb_tags,
        tag_to_id=processor.tag_to_id,
        tag_padding=args.data_params.get("tag_padding"))
    model.build()
    crf_accuracy = CrfAcc(processor.tag_to_id, args.data_params.get("tag_padding")).crf_accuracy
    crf_loss = CrfLoss(processor.tag_to_id, args.data_params.get("tag_padding")).crf_loss
    # 模型基础信息
    bert_type = "ALBERT" if args.bert_params.get("albert") is "True" else "BERT"
    clf_type = args.model_params.get("clf_configs").get("clf_type").upper()
    ner_type = args.model_params.get("ner_configs").get("ner_type").upper() + "-CRF"
    model_save_path = os.path.abspath(
        os.path.join(args.save_path, "%s-%s-%s.h5" % (bert_type, clf_type, ner_type)))
    # 训练较难任务
    model.hard_train_model.compile(
        optimizer=keras.optimizers.Adam(lr=args.model_params.get("lr"), beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        loss=crf_loss,
        metrics=[crf_accuracy])
    hard_train_model_callbacks = KbqaCallbacks({
        "early_stop_patience": args.model_params.get("early_stop_patience"),
        "reduce_lr_patience": args.model_params.get("reduce_lr_patience"),
        "reduce_lr_factor": args.model_params.get("reduce_lr_factor")})
    hard_train_model_callbacks.append(TaskSwitch("val_crf_accuracy", args.model_params.get("all_train_threshold")))
    model.hard_train_model.fit(
        x=train_x[:2],
        y=train_x[3],
        batch_size=args.model_params.get("batch_size"),
        epochs=args.model_params.get("max_epochs"),
        validation_data=[[devs[0][0], devs[0][1]], devs[0][3]],
        callbacks=hard_train_model_callbacks)
    # 训练所有任务
    model.train_all_tasks()
    model.full_train_model.compile(
        optimizer=model.hard_train_model.optimizer)
    model.full_train_model.fit(
        x=train_x,
        y=train_y,
        batch_size=args.model_params.get("batch_size"),
        epochs=args.model_params.get("max_epochs"),
        validation_data=devs,
        callbacks=KbqaCallbacks({
            "early_stop_patience": args.model_params.get("early_stop_patience"),
            "reduce_lr_patience": args.model_params.get("reduce_lr_patience"),
            "reduce_lr_factor": args.model_params.get("reduce_lr_factor")}))
    # 保存信息
    with codecs.open(os.path.join(args.save_path, "label_to_id.pkl"), "wb") as f:
        pickle.dump(processor.label_to_id, f)
    with codecs.open(os.path.join(args.save_path, "id_to_label.pkl"), "wb") as f:
        pickle.dump(processor.id_to_label, f)
    with codecs.open(os.path.join(args.save_path, "tag_to_id.pkl"), "wb") as f:
        pickle.dump(processor.tag_to_id, f)
    with codecs.open(os.path.join(args.save_path, "id_to_tag.pkl"), "wb") as f:
        pickle.dump(processor.id_to_tag, f)
    model_configs = {
        "tag_padding": args.data_params.get("tag_padding"),
        "max_len": args.data_params.get("max_len"),
        "bert_vocab": os.path.abspath(args.bert_params.get("bert_vocab")),
        "model_path": model_save_path,
        "id_to_label": os.path.abspath(os.path.join(args.save_path, "id_to_label.pkl")),
        "id_to_tag": os.path.abspath(os.path.join(args.save_path, "id_to_tag.pkl"))}
    with codecs.open(os.path.join(args.save_path, "model_configs.json"), "w") as f:
        json.dump(model_configs, f, ensure_ascii=False, indent=4)
    model.pred_model.save(model_save_path)
