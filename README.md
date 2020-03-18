[English Version](https://github.com/liushaoweihua/keras-bert-kbqa/blob/master/README.md)  |  [中文版说明](https://github.com/liushaoweihua/keras-bert-kbqa/blob/master/README_ZH.md)

# Keras-Bert-Kbqa

Implementation of *Neural Network (NN)* model in the field of *Knowledge Based Question Answering (KBQA)* with Pre-trained Language Model: supports BERT/RoBERTa/ALBERT。

## Core Tasks of KBQA

* **Build Knowledge System (KB)**
  * Organize knowledge system based on business characteristics.
  * Extract triples from unstructured input text `(Subject, Predicate, Object)` and store them in a specific way (usually a graph database). 
    * For example: "Chow SingChi's movie *Kung Fu Hustle * was released in 2004", including two pairs of triples `(Chow SingChi, filmed, Kung Fu Hustle)`, `(Kung Fu Hustle, release time, 2004)`.
  
* **Standard Question Answering (QA)**
  * Relational entity extraction
    * Extracting `(Subject, Predicate)` from query statements. 
    * For example: "In which year is *Kung Fu Hustle* released" includes `(Kung Fu Hustle, release time)`.
  * Entity disambiguation
    * Solving the problem of ambiguity caused by entities with the same name. 
    * For example: Chow SingChi and Xing Ye should correspond to the same entity.
  * Relationship linking
    * Linking the extracted entities and relationships to ensure that the linked entity relationship is valid in the knowledge system.
    * For example, in the *Douban Movie Review* scene, asking "What is the name of Chou Xingchi's mother", the obtained `(Chow SingChi, mother)` is illegal because the relationship is not established in the knowledge system.
  * Response generating
    * Retrieve legal relational entity pairs in the knowledge system and generate output results.

## What's Involved

This project mainly focuses on the **relational entity extraction** part of the **standard question and answer (QA)** task. Regular KBQA Query contains the following categories:
* One-hop derivation
  * For example: "In which year is *Kung Fu Hustle* released," including `(Kung Fu Hustle, release time)`.
* Comparison of derivation results
  * For example: "Is *Kung Fu Hustle* release earlier than *All for the Winner*", which includes `((Kung Fu Hustle, release time) ~ (All for the Winner, release time))`, you need to retrieve all results for comparison.
* Nested derivation
  * For example: "What is the age of Chow SingChi's mother", includes `((Chow SingChi, mother), age)`, which requires nested query.
* Comparison of nested derivation results
  * For example: "Is Chow SingChi's mother older than Ng Mang Tat", including `(((Chow SingChi, mother), age) ~ (Ng Mang Tat, age))`.

The project only deals with the first case, which is most commonly used. **Relationships are obtained through classification with global semantics, at the same time, the entities are obtained through sequence labeling**.

For the latter three cases, **first obtain the entities through sequence labeling, and then use global semantics and local semantics of entities (prior information) to obtain multiple relationships**. At present, the difficulties lie in **how to accurately link multi-relationships with multi-entities** and **how to handle the amplification of loss of multi-tasking**. It's hard to deal with using *neural network* models alone.

## Method

### Model Structure

Methods of One-hop derivation
* Pipeline method: The Relationship classification and entity extraction tasks are divided into two tasks to calculate the loss separately without affecting each other.
  * Easy to train as both of them are regular NLP tasks: classification and sequence labeling.
  * Slow in inference, the input needs to be fed into both models at inference phase.
  * Illegal results are prone to occur due to the model prediction error, which need to perform relationship linking task.
    * For example: `(Chow SingChi, release time)`
* Joint method: The relationship classification and entity extraction tasks interact with each other, using the same embedding layer to obtain semantic encoding, and loss is calculated in multi-tasking way.
  * Hard to train, both loss and gradient descent speed of the two tasks are not on the same magnitude;
  * Fast in inference, the input is fed into a single model at inference phase.
  * Illegal results rarely occur due to the same semantic encoding for both tasks, which avoid the relationship linking task.

**Joint method is used in this project**.

### Training Method

* Train difficult sequence labeling task first and freeze the downstream weights of classification task until the validation set accuracy reaches a default threshold.
* MultiLoss mentioned in [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115.pdf) is modified to calculate the multi-loss of classification task and sequence labeling task. The optimizer is inherit from the previous stage to train both tasks at the same time.

## Project Framework

```bash
keras_bert_kbqa
├── helper.py                       # help file for training paramters
├── __init__.py
├── predict.py                      # load the trained model
├── train.py                        # train and save model
└── utils
    ├── bert.py                     # keras implementation of bert model
    ├── callbacks.py                # EarlyStopping，ReduceLROnPlateau，TaskSwitch
    ├── decoder.py                  # Viterbi decoder of sequence labeling task
    ├── graph_builder.py            # neo4j graph database processing function, not used
    ├── __init__.py
    ├── metrics.py                  # Crf_accuracy and Crf_loss of sequence labeling task, support mask
    ├── models.py                   # support textcnn and dense for classification task, idcnn-crf and bilstm-crf for sequence labeling task
    ├── processor.py                # Standardized training/validation dataset
    └── tokenizer.py                # tokenizer of bert model
```

## Dependencies

* flask == 1.1.1
* keras == 2.3.1
* numpy == 1.18.1
* loguru == 0.4.1
* requests == 2.22.0
* termcolor == 1.1.0
* tensorflow == 1.15.2
* keras_contrib == 2.0.8

## Case: Douban Movie Review

* This project mainly focuses on the **relational entity extraction** part of the **standard question and answer (QA)** task. For the rest, it is implemented in a relatively crude way (not using a graph database).
* This case mimics the online case in engineering practice, which exist problems of **low amount of query data** and **difference between the data generated by `template + filling` method and actural data**.
* Test results show that in the case of low data volume:
   * **Generalization error is large**, the use of neural network models alone does not work well. It should be used in combination with regular expressions and rules.
   * **Models are difficult to train**.

### Case Framework

```bash
examples
├── data
│   ├── build_upload.py              # generate training/validation data from raw data
│   ├── data
│   │   ├── database.txt             # database generated from raw data for query result retrieval (not using graph database)
│   │   ├── dev_data.txt             # validation data
│   │   ├── prior_check.txt          # double check, correcting the errors of entities obtained by nn model
│   │   └── train_data.txt           # training data
│   ├── origin_data
│   │   └── douban_movies.txt        # raw data
│   └── templates
│       ├── neo4j_config.txt         # configs of graph database, not used
│       ├── text_templates.txt       # templates for generating training/validation data
│       └── utter_search.txt         # query result retrieval instructions(crude impletementation, not using graph database)
├── deploy                           # deploy a trained model for use
│   ├── run_deploy.py
│   └── run_deploy.sh
├── models                           # model save path
│   ├── ALBERT-IDCNN-CRF.h5
│   ├── id_to_label.pkl
│   ├── id_to_tag.pkl
│   ├── label_to_id.pkl
│   ├── model_configs.json
│   └── tag_to_id.pkl
└── train                            # train a new model
    ├── run_train.py
    ├── run_train.sh
    └── train_config.json            # train configs

```

### Data Format

The form of training/validation data is `[text information, category information, sequence labeling information]`, shown as follows:

```json
[
    [
        "骗中骗的评分高吗",
        "豆瓣评分",
        "B-title I-title I-title O O O O O"
    ],
    [
        "安东尼娅家族啥时候上映的呀",
        "电影上映时间是什么",
        "B-title I-title I-title I-title I-title I-title O O O O O O O"
    ],
...
]
```

### Some Tricks for Setting Training Parameters

This part is located in `examples/train/train_config.json`:

* The sentence length parameter `max_len` should be adapted to the length of the training/validation text. Excessively long sentence length will occupy a large amount of video memory and have a large impact on the convergence of the sequence labeling task.
* ALBERT model is easier to train than BERT model in low data volume scene, and the performance has no significant difference compared with BERT model.
* `all_train_threshold` indicates that when the validation accuracy of the sequence labeling task reaches this value, both the classification task and the sequence labeling task are trained:
  * If it is too small, the sequence labeling task cannot converge, and the classification task is prone to over-fitting.
  * If it is too large, the classification task is prone to under-fitting.
  * The recommended value is between 0.9 and 0.98.
* `clf_type` can be `textcnn` and `dense`:
  * When it is `textcnn`, the rest parameters are `dense_units`, `dropout_rate`, `filters` and `kernel_size`.
  * When it is `dense`, the rest parameter is `dense_units`.
* `ner_type`can be `idcnn` and `bilstm`:
  * When it is `idcnn`, the rest parameters are `filters`, `kernel_size` and `blocks`.
  * When it is `bilstm`, the rest parameters are `units`, `num_hidden_layers` and `dropout_rate`.

### Implementation Process

```bash
python examples/data/build_upload.py # generate all files in examples/data/data
bash examples/train/run_train.sh     # train a new model
bash examples/deploy/run_deploy.sh   # deploy a trained model for use
```

### Usage

Send a request to API:

```python
import requests

r = requests.post(
    "http://your_ip:your_port/query",
    json={
        "text": "大话西游之大圣娶亲是最近刚上的电影吗"})

print(r.text)
```

Returns:

```json
{
    "text": "大话西游之大圣娶亲是最近刚上的",
    "predicate": "电影上映时间是什么",
    "subject": [
        {
            "title": "大话西游之大圣娶亲"
        }
    ],
    "response": "2014"
}
```

## Future Work

* Optimize model structure to make it easier to train.
* Try to handle more complex KBQA scenarios.
* Improve some details.
* Migrate to tensorflow 2.0.
* Add other BERTs models, like Distill_Bert, Tiny_Bert.

## Some Chinese Pretrained Language Model

> **BERT**
* [Google_bert](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)
* [HIT_bert_wwm_ext](https://storage.googleapis.com/chineseglue/pretrain_models/chinese_wwm_ext_L-12_H-768_A-12.zip)

> **ALBERT**
* [Google_albert_base](https://storage.googleapis.com/albert_models/albert_base_zh.tar.gz)
* [Google_albert_large](https://storage.googleapis.com/albert_models/albert_large_zh.tar.gz)
* [Google_albert_xlarge](https://storage.googleapis.com/albert_models/albert_xlarge_zh.tar.gz)
* [Google_albert_xxlarge](https://storage.googleapis.com/albert_models/albert_xxlarge_zh.tar.gz)
* [Xuliang_albert_xlarge](https://storage.googleapis.com/albert_zh/albert_xlarge_zh_177k.zip)
* [Xuliang_albert_large](https://storage.googleapis.com/albert_zh/albert_large_zh.zip)
* [Xuliang_albert_base](https://storage.googleapis.com/albert_zh/albert_base_zh.zip)
* [Xuliang_albert_base_ext](https://storage.googleapis.com/albert_zh/albert_base_zh_additional_36k_steps.zip)
* [Xuliang_albert_small](https://storage.googleapis.com/albert_zh/albert_small_zh_google.zip)
* [Xuliang_albert_tiny](https://storage.googleapis.com/albert_zh/albert_tiny_zh_google.zip)

> **Roberta**
* [roberta](https://storage.googleapis.com/chineseglue/pretrain_models/roeberta_zh_L-24_H-1024_A-16.zip)
* [roberta_wwm_ext](https://storage.googleapis.com/chineseglue/pretrain_models/chinese_roberta_wwm_ext_L-12_H-768_A-12.zip)
* [roberta_wwm_ext_large](https://storage.googleapis.com/chineseglue/pretrain_models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip)

## Reference

* [Keras-Bert-Ner](https://github.com/liushaoweihua/keras-bert-ner)
* [bert4keras](https://github.com/bojone/bert4keras)
* [albert_zh](https://github.com/brightmart/albert_zh)
* [BERT](https://github.com/google-research/bert), [ALBERT](https://github.com/google-research/albert), [RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta)。

Thanks for all these wonderful works! 