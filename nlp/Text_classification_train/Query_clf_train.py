#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/27 18:29
# @Author  : Gongle
# @File    : Query_clf_train.py
# @Version : 1.0
# @Desc    : None

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from rouge import Rouge

from transformers import BertTokenizerFast
from transformers import AdamW
from collections import OrderedDict
from datetime import datetime
from typing import Optional,Union,List
import ahocorasick
import logging
import json
import copy
import time
import argparse
from pprint import pprint
import re



from Query_clf_proc_dataset import load_data,logger,Query_clf_dataset,collate_fn
from Query_clf_model import Query_clf_model


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

def clf_evaluator(dev_df,id2label,dev_dataloader,model,device):
    true_labels=[]
    num_classes=len(id2label)
    batch_pred_result = np.empty(shape=[0,num_classes])
    for batch in dev_dataloader:
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        b_labels = batch['all_labels']
        b_labels_list=b_labels.to('cpu').numpy().tolist()

        true_labels.extend(b_labels_list)
        with torch.no_grad():
            logits = model(input_ids=batch['all_input_ids'],
                       attention_mask=batch['all_attention_mask'],
                       token_type_ids=batch['all_token_type_ids'])
        logits = logits.to('cpu').numpy()
        batch_pred_result= np.append(batch_pred_result,logits, axis = 0)
    batch_pred_proba=softmax(batch_pred_result)
    batch_pred_class = batch_pred_proba.argmax(axis=1)
    batch_pred_class=batch_pred_class.astype(np.int64)
    batch_pred_label_text=[id2label[item] for item in batch_pred_class]
    f1=f1_score(true_labels,batch_pred_class,average='weighted')
    acc=accuracy_score(true_labels,batch_pred_class)
    precision = precision_score(true_labels, batch_pred_class, average='weighted')
    recall = recall_score(true_labels, batch_pred_class, average='weighted')
    batch_pred_probability = batch_pred_proba.max(axis=1)
    dev_df.loc[:, 'pred_proba'] = batch_pred_probability
    dev_df.loc[:,'pred_label']=batch_pred_label_text
    dev_df.loc[:, 'equal_bool'] = dev_df.apply(lambda x: 1 if x['pred_label'] == x['label_answer'] else 0, axis=1)
    logger.info('dev acc: {:.4f}'.format(acc))
    #save_file = 'output_data/query_clf_eval_df_' + datetime.now().strftime("%Y-%m-%d_%H") + '.xlsx'
    #dev_df.to_excel(save_file)
    return f1,precision,recall,acc


def argument_parse():
    parser=argparse.ArgumentParser('setting parameters for sent bert training similarity')
    parser.add_argument('-a', '--max_answer_length', default=30, type=int, help="max answer length")
    parser.add_argument('-d','--data-file',default='data/query_clf_labeled_data_20221230.xlsx',type=str,help='train data  excel file ')
    parser.add_argument('-l', '--label-file', default='data/instruct_label2id.json', type=str,help='label2id json file ')
    parser.add_argument('-b','--batch-size', default=64,type=int, help=" batch size")
    parser.add_argument('-e', '--epoch', default=80, type=int, help=" num epochs")
    parser.add_argument('-ml', '--max-len', default=220, type=int, help="question_len+context_len+3_sep_token")
    parser.add_argument('-me','--max_early', default=12, type=int, help="max early stop")
    parser.add_argument('-p', '--pretrain', default='/home/stops/Work_space/NLP_models/chinese-roberta-wwm-ext', type=str, help="pretrain model name")
    parser.add_argument('-o', '--output', default='clf_model/', type=str, help="save train recall model path")
    opt=parser.parse_args()
    return(opt)

def train_main():
    args=argument_parse()
    mrc_data_path = args.data_file
    batch_size=args.batch_size
    epochs=args.epoch
    save_model_path=args.output
    logger.info('Loading  data : {} '.format(mrc_data_path))
    #split_mode = 'privilege'
    split_mode = 'random'
    #label2id_file = 'data/type_label2id_0207.json'
    label2id_file =args.label_file
    with open(label2id_file,'r') as f :
        label2id=json.load(f)
    logger.info('load label2id file: {},nums: {}'.format(label2id_file,len(label2id)))
    train_df, dev_df = load_data(mrc_data_path,label2id_file=label2id_file, dev_size=0.06, split_mode=split_mode)
    logger.info('Loading Tokenizer')
    pretrained_model=args.pretrain
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
    ####  train dataset-->dataloader
    logger.info('train parameters  epochs : {} ,batch_size : {} '.format(epochs,batch_size))
    clf_train_dataset = Query_clf_dataset(train_df,tokenizer)
    logger.info('dataset nums :{}'.format(clf_train_dataset.__len__()))
    train_clf_dataloader = DataLoader(clf_train_dataset, shuffle=True, batch_size=batch_size,
                                      collate_fn=collate_fn)
    logger.info('Processing train dataset dataloader ')
    ####  dev dataset-->dataloader
    clf_dev_dataset = Query_clf_dataset(dev_df,tokenizer)
    logger.info('dataset nums :{}'.format(clf_dev_dataset.__len__()))
    dev_clf_dataloader = DataLoader(clf_dev_dataset, shuffle=False, batch_size=batch_size,
                                    collate_fn=collate_fn)
    logger.info('Processing dev dataset dataloader ')
    id2label = dict(zip(label2id.values(), label2id.keys()))
    num_labels = len(label2id)
    logger.info('model classes num: {} '.format(num_labels))
    model = Query_clf_model.from_pretrained(pretrained_model,num_labels=num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    model.to(device)
    logger.info('Loading  CLF model')
    # optimizer
    adam_epsilon = 1e-8
    weight_decay = 0.01
    learning_rate = 1e-5
    linear_learning_rate = 1e-3
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': learning_rate
         },
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': learning_rate
         },
        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': linear_learning_rate
         },
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': linear_learning_rate
         },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    ####  main training
    ####  main training
    best_f1 = 0
    global_step = 0
    logging_steps = 10
    valid_steps = 30
    for epoch, _ in enumerate(range(epochs)):
        batch_loss = 0
        for step, batch in enumerate(train_clf_dataloader):
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            loss = model(input_ids=batch['all_input_ids'],
                         attention_mask=batch['all_attention_mask'],
                         token_type_ids=batch['all_token_type_ids'],
                         labels=batch['all_labels'])
            loss.backward()
            batch_loss += loss.item()
            optimizer.step()
            model.zero_grad()
            global_step += 1
            if global_step % logging_steps == 0:
                logging.info('epoch: {} step :{} loss: {}'.format(epoch, step, batch_loss / (step + 1)))
            if global_step % valid_steps == 0:
                save_dir = os.path.join(save_model_path, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                logger.info('save path: {}'.format(save_dir))
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                f1_val,precision_val,recall_val,acc_val = clf_evaluator(dev_df,id2label,dev_clf_dataloader,model,device)
                logger.info("Evaluation f1: %.5f, precision: %.5f, recall: %.5f; acc: %.5f" % (f1_val,precision_val,recall_val,acc_val))
                if f1_val > best_f1:
                    logger.info(f"best F1 performance has been updated: {best_f1:.5f} --> {f1_val:.5f}")
                    best_f1 = f1_val
                    save_dir = os.path.join('./clf_checkpoint', "model_best")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    logger.info('save path: {}'.format(save_dir))
                    model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)


if __name__=='__main__':
    train_main()


"""
cls-18


#### update 0908

nohup python Query_clf_train.py  -e 5 -b 30 -d data/instruct_clf_df_0908.xlsx -l data/instruct_label2id.json > log/instruct_clf_log_0908.txt 2>&1 & 
tail -F log/instruct_clf_log_0908.txt


nohup python Query_clf_train.py  -e 3 -b 60 -d data/Doctor_query_type_clf_train_data_norm_v0_240125.xlsx -l data/doc_type_label2id_240125.json  -o doc_query_clf_model > log/doc_query_clf_log_v2_240105.txt 2>&1 & 
tail -F log/doc_query_clf_log_v2_240105.txt


"""








