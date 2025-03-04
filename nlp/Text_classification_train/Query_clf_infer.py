#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/27 18:29
# @Author  : Gongle
# @File    : Query_clf_infer.py
# @Version : 1.0
# @Desc    : None

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertForTokenClassification, BertPreTrainedModel, BertModel
from transformers import AutoModelForTokenClassification,AutoModelForPreTraining,AutoModel,AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import BertTokenizerFast
from typing import Optional,Union,List
from datetime import datetime
import time
import logging
import copy
import json
import re
import os
from pprint import pprint

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

from Query_clf_model import Query_clf_model,softmax


class Query_clf_Infer():
    def __init__(self,model_path,label2id_file='data/type_label2id_0206.json',batch_size=300):
        self.model_path=model_path
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer=BertTokenizerFast.from_pretrained(self.model_path)
        self.model= Query_clf_model.from_pretrained(self.model_path)
        self.model.eval()
        self.model.to(self.device)
        self.batch_size=batch_size
        self.label2id_file=label2id_file
        if self.label2id_file:
            with open(self.label2id_file,'r') as f :
                self.label2id=json.load(f)
            self.id2label=dict(zip(self.label2id.values(),self.label2id.keys()))
            self.num_classes=len(self.id2label)
        else:
            self.label2id=None
            self.id2label=None
            self.num_classes =2
        logger.info('load model and tokenizer : {}, num_classes: {}'.format(self.model_path,self.num_classes))

    def batch_infer(self,query:Union[str,List[str]]):
        if isinstance(query,str):
            query=[query]
        max_seq_len=max([len(query[i]) for i in range(len(query))])
        max_seq_len=min(max_seq_len,510)
        tokenized_examples=self.tokenizer(query,padding="max_length",max_length=max_seq_len+2,truncation=True)
        all_logits=np.empty(shape=[0,self.num_classes])
        with torch.no_grad():
            for idx in range(0,len(query),self.batch_size):
                bth_input_ids=torch.as_tensor(tokenized_examples['input_ids'][idx:idx+self.batch_size])
                bth_token_type_ids=torch.as_tensor(tokenized_examples['token_type_ids'][idx:idx+self.batch_size])
                bth_attent_mask=torch.as_tensor(tokenized_examples['attention_mask'][idx:idx+self.batch_size])
                logits_tensor= self.model(
                    input_ids=bth_input_ids.to(self.device),
                    token_type_ids=bth_token_type_ids.to(self.device),
                    attention_mask=bth_attent_mask.to(self.device))
                logits_np=logits_tensor.cpu().numpy()
                all_logits= np.append(all_logits,logits_np, axis = 0)
        batch_pred_proba = softmax(all_logits)
        batch_pred_class = batch_pred_proba.argmax(axis=1)
        if self.id2label:
            batch_pred_class=[self.id2label.get(item) for item in batch_pred_class]
        batch_pred_probability=batch_pred_proba.max(axis=1)
        batch_pred_probability=[round(item,4) for item in batch_pred_probability]
        return batch_pred_probability,batch_pred_class


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    clf_model = 'models/query_bool_model_0210_merge_L12'
    label2id_file = 'data/query_bool_label2id_map.json'
    QCI = Query_clf_Infer(clf_model,label2id_file)
    s_time = time.time()
    pred_question_bth = ['宝贝几岁可以戒奶',
                         '我的卡怎么没法用呀']
    all_predictions = QCI.batch_infer(pred_question_bth)
    print('all_predictions: ', all_predictions)
    for i in range(len(all_predictions)):
        print('***' * 15)
        print('Group index : ', i)
        print('Question    : ', pred_question_bth[i])
        print('Proba       : {}, Label:  {}  '.format(all_predictions[0][i], all_predictions[1][i]))
    print('cost_time: ', time.time() - s_time)





