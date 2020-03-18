[English Version](https://github.com/liushaoweihua/keras-bert-kbqa/blob/master/README.md)  |  [中文版说明](https://github.com/liushaoweihua/keras-bert-kbqa/blob/master/README_ZH.md)

# Keras-Bert-Kbqa

**预训练语言模型BERT系列**在**知识图谱问答领域**的纯神经网络模型实现尝试，支持**BERT/RoBERTa/ALBERT**。

## KBQA的核心任务

* **知识体系构建（KB）**
  * 基于业务特点，梳理知识体系；
  * 非结构化输入文本抽取三元组`（主实体Subject，关系Predicate，客实体Object）`，并以特定方式进行存储（通常为图数据库）。
    * 如："周星驰的电影功夫上映于2004年"，包含两对三元组`（周星驰，拍摄的电影，功夫）`，`（功夫，上映时间，2004年）`；
  
* **标准问答查询（QA）**
  * 关系实体抽取
    * 查询语句抽取二元组`（主实体Subject，关系Predicate）`；
    * 如："功夫上映于哪一年"，包含一对二元组`（功夫，上映时间）`；
  * 实体消歧
    * 解决同名实体产生歧义的问题；
    * 如：周星驰和星爷应对应同一实体；
  * 关系链接
    * 将抽取得到的实体与关系进行链接，保证链接后的实体关系在知识体系中是有效的；
    * 如：豆瓣影评任务下询问"周星驰的母亲叫什么名字"，所得到的二元组`（周星驰，母亲）`是非法的，因为知识体系中未建立该关系；
  * 结果查询
    * 在知识体系中检索合法的关系实体对，获取结果输出。

## 涉及内容

本项目主要关注**标准问答查询（QA）**任务中的**关系实体抽取**部分。常规KBQA的Query包含以下类别：
* 单跳推导
  * 如："功夫上映于哪一年"，二元组为`（功夫，上映时间）`；
* 推导比较
  * 如："功夫的上映时间和赌圣比哪个早"，二元组为`（（功夫，上映时间）~（赌圣，上映时间）)`，需要分别查询结果进行比较；
* 嵌套推导
  * 如："周星驰的母亲的年龄"，二元组为`（（周星驰，母亲），年龄）`，需要进行嵌套查询；
* 嵌套推导比较
  * 如："周星驰的母亲的年龄和吴孟达的年龄谁大"，二元组为`（（（周星驰，母亲），年龄）~（吴孟达，年龄)）`。

项目仅处理最常用的第一种情况，**同时采用全局语义进行分类获取关系，并逐字进行序列标注获取实体**。

对于后三种情况，需要**先逐字进行序列标注获取实体，并采用全局语义与实体的局部语义（先验信息）获取多个关系**，目前难点在于所获取的**多关系与多实体的准确链接**与**多任务处理的损失放大**上，纯模型较难处理。

## 处理方式

### 模型结构

单跳推导的两类方法：
* 流水线式（Pipeline）方法：关系分类和实体抽取分成两个任务进行，分别计算loss，不互相影响
  * 模型训练简单，为常规NLP下游任务：分类和序列标注；
  * 预测速度慢，需要同时输入两个模型；
  * 因为模型预测误差，容易出现非法的二元组，如：`（周星驰，上映时间）`，需要执行关系链接操作；
* 联合式（Joint）方法：关系分类和实体抽取采用同一个公共Embedding层进行编码，并采用multi-task的处理方式计算loss，互相影响
  * 模型训练困难，两类下游任务的loss及梯度下降速度均不在一个量级上；
  * 预测速度快，仅通过单模型即可获得两个下游任务的输出；
  * 公共Embedding层编码共用信息的情况下，极少出现非法二元组，避免后续再执行关系链接的操作。

**项目采用联合式方法进行构建。**

### 训练方式
* 先训练较难的序列标注任务，冻结分类任务的下游权重，直至验证集精度达到设定阈值；
* 采用[Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115.pdf)提到的MultiLoss，魔改并计算分类任务和序列标注任务的multi-loss，继承前一阶段的optimizer，同时训练两类下游任务。

## 项目框架

```bash
keras_bert_kbqa
├── helper.py                       # 训练参数帮助文件
├── __init__.py
├── predict.py                      # 加载已训练模型
├── train.py                        # 新模型训练与保存
└── utils
    ├── bert.py                     # bert模型的keras实现
    ├── callbacks.py                # EarlyStopping，ReduceLROnPlateau，TaskSwitch
    ├── decoder.py                  # 序列标注任务的Viterbi解码器
    ├── graph_builder.py            # neo4j图数据库处理函数，在本项目中未进行使用
    ├── __init__.py
    ├── metrics.py                  # 序列标注任务的Crf_accuracy和Crf_loss，支持mask
    ├── models.py                   # 分类任务支持textcnn和dense，序列标注任务支持idcnn-crf和bilstm-crf
    ├── processor.py                # 标准化训练/验证数据集
    └── tokenizer.py                # bert模型的分词器
```

## 依赖项

* flask == 1.1.1
* keras == 2.3.1
* numpy == 1.18.1
* loguru == 0.4.1
* requests == 2.22.0
* termcolor == 1.1.0
* tensorflow == 1.15.2
* keras_contrib == 2.0.8

## 案例：豆瓣影评

* 由于项目主要关注**标准问答查询（QA）**任务中的**关系实体抽取**部分，故而对于其余部分用较为简陋的方式进行实现（未采用图数据库）；
* 该案例模仿了工程实践中的线上案例，**存在问法数据量较低的问题，通过模板+填充的方式生成数据与实际数据分布存在较大差异，但仍是目前的主流做法**；
* 测试结果表明在数据量较低的情况下：
  * **泛化误差较大**，仅采用模型效果不佳，应与正则、规则结合使用；
  * **模型难以训练**。



### 案例框架

```bash
examples
├── data
│   ├── build_upload.py              # 从原始数据中生成训练数据、验证数据等
│   ├── data
│   │   ├── database.txt             # 从原始数据中生成的数据库，用于查询结果检索（未使用图数据库）
│   │   ├── dev_data.txt             # 验证数据
│   │   ├── prior_check.txt          # 双重验证兜底，对于算法识别实体错误的结果进行纠正
│   │   └── train_data.txt           # 训练数据
│   ├── origin_data
│   │   └── douban_movies.txt        # 原始数据
│   └── templates
│       ├── neo4j_config.txt         # 图数据库配置文件，在本项目中未进行使用
│       ├── text_templates.txt       # 训练/验证数据生成模板
│       └── utter_search.txt         # 问题查询数据库命令（未使用图数据库，因此写的比较丑陋）
├── deploy                           # 发布使用
│   ├── run_deploy.py
│   └── run_deploy.sh
├── models                           # 模型信息保存位置
│   ├── ALBERT-IDCNN-CRF.h5
│   ├── id_to_label.pkl
│   ├── id_to_tag.pkl
│   ├── label_to_id.pkl
│   ├── model_configs.json
│   └── tag_to_id.pkl
└── train                            # 模型训练
    ├── run_train.py
    ├── run_train.sh
    └── train_config.json            # 训练配置文件

```

### 数据形式

训练、验证数据形式为`[文本信息，类别信息，序列标注信息]`，如下：

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

### 训练参数配置的一些技巧

该部分内容位于`examples/train/train_config.json`中：

* 句长参数`max_len`应适配于训练、测试文本的长度，过长的句长将占用较大的显存，且对于序列标注任务的收敛影响较大；
* 在数据量较低的情况下，ALBERT模型比BERT模型更易训练，且效果与BERT模型相差不大；
* `all_train_threshold`表示序列标注任务的验证精度达到该值时，同时训练分类任务和序列标注任务：
  * 该值过小将导致序列标注任务无法收敛，而分类任务易过拟合；
  * 该值过大将导致分类任务欠拟合；
  * 建议取值在0.9~0.98之间；
* `clf_type`可取`textcnn`和`dense`：
  * 为`textcnn`时，其余参数为`dense_units`，`dropout_rate`，`filters`和`kernel_size`；
  * 为`dense`时，其余参数为`dense_units`；
* `ner_type`可取`idcnn`和`bilstm`：
  * 为`idcnn`时，其余参数为`filters`，`kernel_size`和`blocks`；
  * 为`bilstm`时，其余参数为`units`，`num_hidden_layers`和`dropout_rate`。

### 执行流程

```bash
python examples/data/build_upload.py # 生成examples/data/data中的所有文件
bash examples/train/run_train.sh     # 训练模型
bash examples/deploy/run_deploy.sh   # 使用模型
```

### 模型使用

调用接口：

```python
import requests

r = requests.post(
    "http://your_ip:your_port/query",
    json={
        "text": "大话西游之大圣娶亲是最近刚上的电影吗"})

print(r.text)
```

接口返回结果：

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

## 未来工作

* 优化训练难度，使模型更容易训练；
* 尝试处理更为复杂的KBQA场景；
* 项目细节完善；
* 项目迁移至tensorflow 2.0；
* 新增BERT系列改进模型，如Distill Bert，Tiny Bert。

## 一些常用的中文预训练模型

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

## 参考
* 同源项目：[Keras-Bert-Ner](https://github.com/liushaoweihua/keras-bert-ner)
* 项目的BERT代码参考：[bert4keras](https://github.com/bojone/bert4keras)
* ALBERT中文预训练模型系列，更快的推理时间和较高的预测精度：[albert_zh](https://github.com/brightmart/albert_zh)
* [BERT](https://github.com/google-research/bert), [ALBERT](https://github.com/google-research/albert), [RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta)。

感谢以上作者和项目的贡献！