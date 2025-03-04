#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/27 18:28
# @Author  : Gongle
# @File    : Query_clf_model.py
# @Version : 1.0
# @Desc    : None

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import Dataset,DataLoader

from transformers import BertConfig,BertForSequenceClassification, BertPreTrainedModel, BertModel
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoModelForTokenClassification,AutoModel,AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import BertTokenizerFast
from transformers import AdamW

def softmax(x):
    orig_shape = x.shape
    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax,1,x)
        denominator = np.apply_along_axis(denom,1,x)
        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0],1))
        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator =  1.0 / np.sum(numerator)
        x = numerator.dot(denominator)
    assert x.shape == orig_shape
    return x


class Query_clf_model(BertPreTrainedModel):
    def __init__(self, config):
        super(Query_clf_model, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        hidden_size = self.bert.config.hidden_size
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, config.num_labels)

    def forward(self,input_ids=None,token_type_ids=None,attention_mask=None,labels=None):
        output = self.bert(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
        pooled_output = output[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss if loss is not None else logits



