#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/01/12 18:28
# @Author  : Gongle
# @File    : Query_clf_proc_dataset.py
# @Version : 1.0
# @Desc    : None


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import ahocorasick
import logging
from datetime import datetime
from typing import Optional,Union,List
import json
import re
import os
import copy
"""
处理迭代模型数据
"""
logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
logger=logging.getLogger('Medical MRC APP')

def train_val_test(train_data_df, mode_index=False,balance_label='label', balanced=True, val_size=0.1, test_size=0.2, seed=666):
    train_data_df=train_data_df.reset_index(drop=True)
    if test_size>0. :
        train_idx_pre, test_idx = train_test_split(range(train_data_df.shape[0]), \
                                               test_size=test_size, random_state=seed)
    else:
        train_idx_pre,test_idx=list(range(train_data_df.shape[0])),[]
    train_data_pre = train_data_df.loc[train_idx_pre, :]
    test_data = train_data_df.loc[test_idx, :]
    val_size = round(val_size / (1 - test_size), 2)
    if balanced:
        if val_size>0. :
            x_tra_idx_pre, x_tes_idx, y_tra_label_pre, y_tes_label = train_test_split(train_idx_pre, \
                       train_data_pre.loc[train_idx_pre, balance_label], test_size=val_size, random_state=666)
        else:
            x_tra_idx_pre, x_tes_idx, y_tra_label_pre, y_tes_label=train_idx_pre,[],train_data_pre.loc[train_idx_pre, balance_label],[]
        x_tra_idx_pre_arr = np.array(x_tra_idx_pre).reshape(-1, 1)
        x_tra_idx_balanced, y_tra_label_balanced = RandomOverSampler(random_state=666).fit_resample(x_tra_idx_pre_arr,y_tra_label_pre)
        #x_tra_idx_balanced, y_tra_label_balanced = RandomUnderSampler(random_state=666).fit_resample(x_tra_idx_pre_arr,y_tra_label_pre)
        x_tra_idx = [x[0] for x in x_tra_idx_balanced]
        if val_size>0.:
            y_tra_label, y_tes_label = y_tra_label_balanced.tolist(), y_tes_label.tolist()
        else:
            y_tra_label, y_tes_label = y_tra_label_balanced.tolist(), []
    else:
        if val_size>0. :
            x_tra_idx, x_tes_idx, y_tra_label, y_tes_label = train_test_split(train_idx_pre, \
                        train_data_pre.loc[train_idx_pre, balance_label], test_size=val_size, random_state=666)
        else:
            x_tra_idx_pre, x_tes_idx, y_tra_label_pre, y_tes_label = train_idx_pre, [], train_data_pre.loc[
                train_idx_pre, balance_label], []
        y_tra_idx, y_tes_idx = y_tra_label.values.tolist(), y_tes_label.tolist()
    train_idx, val_idx = x_tra_idx, x_tes_idx
    logger.info('train size : {}  val size : {}  test size: {}'.format(len(train_idx_pre), len(val_idx), len(test_idx)))
    train_data = train_data_pre.loc[train_idx, :]
    val_data = train_data_pre.loc[val_idx, :]
    if mode_index:
        return train_idx, val_idx, test_idx
    else:
        return train_data, val_data, test_data

def load_data(data_file,label2id_file='data/type_label2id_0207.json',dev_size=0.05,split_mode='random'):
    prefix_text="""针对医生的疑问句问诊意图，进行分类，包含【主要症状(时间，性质等)，伴随症状，病因诱因，诊疗经过，既往史，其他】。\n问诊内容："""
    data_df = pd.read_excel(data_file)
    data_df = data_df.sample(frac=1.0, random_state=1234)
    with open(label2id_file,'r') as f :
        label2id=json.load(f)
    logger.info('load label2id file: {}'.format(label2id_file))
    logger.info('label2id: {}'.format(label2id))
    data_df['label']=data_df['label_answer'].map(label2id)
    data_df['text'] = data_df['text'].map(str)
    data_df['text']=prefix_text+data_df['text']
    #save_check_file='Query_clf_pre_train_df_'+datetime.now().strftime("%Y-%m-%d_%H")+'.xlsx'
    #data_df.to_excel(os.path.join('output_data', save_check_file))
    #logging.info(f'Written query clf pre-train corpus file {save_check_file}')
    logger.info('label range min: {}, max: {}'.format(data_df['label'].min(),data_df['label'].max()))
    if data_df.isnull().sum().sum()>0:
        logger.info('Found NULL,Please check data')
    data_df.loc[:, 'label_answer'] = data_df.loc[:,'label_answer'].map(lambda x: str(x).strip())
    data_df=data_df.reset_index(drop=True)
    random_seed=666
    if split_mode=='random':
        train_df_pre, dev_df = train_test_split(data_df, test_size=dev_size, random_state=666)
        dev_df = dev_df.reset_index(drop=True)
        if 'flag' in data_df.columns:
            flag3_dev_df = dev_df.loc[dev_df['flag'] == 3, :].copy()
            dev_df = dev_df.loc[dev_df['flag'] != 3, :]
            train_df_pre = pd.concat([train_df_pre,flag3_dev_df])
            focus_df = train_df_pre.loc[train_df_pre['flag'] == 3, :]
            repeat_nums = 6
            logger.info('repeat_nums: {}'.format(repeat_nums))
            focus_df_duplicated = pd.concat([focus_df] * repeat_nums)
            train_df_pre2, _ = RandomOverSampler(random_state=666).fit_resample(
                train_df_pre, train_df_pre['label'])
            logger.info('train_df_pre2 type: {}'.format(type(train_df_pre2)))
            super_train_df = pd.concat([focus_df_duplicated,train_df_pre2], sort=False)
            train_df = super_train_df.sample(frac=1.0, random_state=random_seed)
            train_df = train_df.reset_index(drop=True)
            #train_df.to_excel(os.path.join('output_data', save_check_file))
            dev_df = dev_df.reset_index(drop=True)
            logger.info('train size: {} ;test size: {}'.format(train_df.shape[0], dev_df.shape[0]))
            logger.info('label dist:\n{}'.format(train_df['label'].value_counts()))
            return train_df , dev_df
        else:
            return train_df_pre, dev_df
    elif split_mode=='privilege':
        ## 样本量少的数据，有些训练
        label_stand_text=data_df['label_answer'].unique().tolist()
        train_df=pd.DataFrame()
        dev_df=pd.DataFrame()
        for sub_text in label_stand_text:
            sub_df=data_df.loc[data_df['label_answer']==sub_text,:].copy()
            sub_nums=sub_df.shape[0]
            if sub_nums>30:
                sample_num=6
                sub_eval_df=sub_df.sample(n=sample_num,random_state=random_seed)
            elif sub_nums>5:
                sample_num=2
                sub_eval_df=sub_df.sample(n=sample_num,random_state=random_seed)
            else:
                sub_eval_df=pd.DataFrame()
            sub_train_df=sub_df.loc[~sub_df.index.isin(sub_eval_df.index),:]
            train_df=pd.concat([train_df,sub_train_df])
            dev_df=pd.concat([dev_df,sub_eval_df])
        train_df,_,_=train_val_test(train_df,val_size=0.0, test_size=0.0)
        focus_df = train_df.loc[train_df['flag'] == 3, :]
        repeat_nums = 5
        logger.info('repeat_nums: {}'.format(repeat_nums))
        focus_df_duplicated = pd.concat([focus_df] * repeat_nums)
        train_df = pd.concat([focus_df_duplicated, train_df], sort=False)
        train_df = train_df.reset_index(drop=True)
        train_df=train_df.sample(frac=1.0,random_state=random_seed)
        train_df = train_df.reset_index(drop=True)
        dev_df = dev_df.reset_index(drop=True)
        logger.info('train size: {} ;test size: {}'.format(train_df.shape[0], dev_df.shape[0]))
        logger.info('train label dist:\n{}'.format(train_df['label'].value_counts()))
        logger.info('dev   label dist:\n{}'.format(dev_df['label'].value_counts()))
        return train_df,dev_df
    else:
        return data_df

def collate_fn(batch):
    max_len = max([sum(x['attention_mask']) for x in batch])
    all_input_ids = torch.tensor([x['input_ids'][:max_len] for x in batch])
    all_token_type_ids = torch.tensor([x['token_type_ids'][:max_len] for x in batch])
    all_attention_mask = torch.tensor([x['attention_mask'][:max_len] for x in batch])
    all_labels = torch.tensor([x["label"] for x in batch],dtype=torch.long)
    return {
        "all_input_ids": all_input_ids,
        "all_token_type_ids": all_token_type_ids,
        "all_attention_mask": all_attention_mask,
        "all_labels": all_labels}

class Query_clf_dataset(Dataset):
    ## without prompt
    def __init__(self,data_df,tokenizer,max_seq_len=500):
        self.max_seq_len=max_seq_len
        self.data_df=data_df
        labels=data_df['label'].tolist()
        tokenized_example_list=tokenizer(data_df['text'].tolist(),padding="max_length",
                                    max_length=self.max_seq_len,truncation=True)
        tokenized_examples=[]
        for idx in range(len(labels)):
            sub_example={'input_ids':tokenized_example_list['input_ids'][idx],
                      'token_type_ids':tokenized_example_list['token_type_ids'][idx],
                      'attention_mask':tokenized_example_list['attention_mask'][idx],
                      'label':labels[idx]}
            tokenized_examples.append(sub_example)
        self.tokenized_examples = tokenized_examples

    def __len__(self):
        return len(self.tokenized_examples)

    def __getitem__(self,index):
        return self.tokenized_examples[index]


if __name__=='__main__':
    clf_data_path = 'data/query_bool_train_data_2023.xlsx'
    #split_mode = 'privilege'
    split_mode = 'random'
    train_df, dev_df = load_data(clf_data_path,'data/query_bool_label2id_map.json', dev_size=0.1, split_mode=split_mode)
    logger.info('Loading data; train size: {}; dev size: {}'.format(train_df.shape, dev_df.shape))

    pretrained_model = "hfl/chinese-roberta-wwm-ext"
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
    max_len = 222
    epochs = 35
    batch_size = 64
    logger.info('Loading Tokenizer')
    ####  train dataset-->dataloader
    clf_train_dataset = Query_clf_dataset(train_df,tokenizer)
    logger.info('dataset nums :{}'.format(clf_train_dataset.__len__()))
    clf_train_dataloader = DataLoader(clf_train_dataset, shuffle=True, batch_size=batch_size,
                                      collate_fn=collate_fn, num_workers=10)



