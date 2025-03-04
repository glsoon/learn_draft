#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 14:35
# @Author  : Gongle
# @File    : Text_Similariy_Sampler_Expl_1011.py
# @Version : 1.0
# @Desc    : None
"""
三种数据形态：
| 文本种类 | 标准文本 | 相似文本| 标签|
|:----:| :----:| :----:| :----:|
|A| A1_Stand_text | A1_Simi_text | label_1 |
|A| A2_Stand_text | A2_Simi_text | label_0 |
|B| B1_Stand_text | B1_Simi_text | label_0 |
|C| C1_Stand_text | C1_Simi_text | label_0 |

第一种[都互斥]：A1,A2,B1,C1都是绝对互斥，不同种类
第二种[外互斥]：A大类中A1,A2不那么互斥，如腹痛，下腹痛；重点与别的大类B,C作区分。
第三种[内互斥]：A大类中A1,A2要做绝对区分；由于与别的大类B,C也有类似的文本，如A中腹痛多久，B,C有胃炎多久，发热多久。
**同时嵌入数据增强模模块**
"""

import numpy   as np
import random
import pandas  as pd
import re
import json
import itertools
from datetime import datetime, date
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import os
import logging
from pprint import pprint

#os.chdir('Disease_10_Doctor_query_v2_1009')

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger=logging.getLogger(__name__)
logger.info('Starting')


###########################################################################
## Exclusive Strategy
###########################################################################

""" 
输入有几种数据：
1. 只有一个数据,all_data->   根据模式-'test'      : train_data,dev_data,test_data
                           或者模式-'train'     : train_data,_,test_data。
                           或者模式-'only-train': train_data,_,_。
2. 想快速验证模型上线，train_data,test_data
3. 已经有切分好的数据，train_data,dev_data,test_data, 验证模型以及调参的有效性。
4. 基于上面的数据，另有soft-负样本数据，用以增强模型学习。
5. 增加样本增强策略。
-------------------------------------------
数据集同一为4列：| 文本种类  |  标准文本   |  相似文本   | 标签  |
对应Column-En, |Category | Stand_text | Simi_text | label |
6.label最好有正负样本，但有时也只有正样本；要分别考虑处理。
7.正负样本抽样成对逻辑[初始化]:
    正样本逻辑
    7.1.1 正相似问之间构造正样本[超过样本量，可能无该正样本集]
    7.1.2 自抽样复制扩充正样本对数量[3次不同随机操作]。
    7.1.3 
    负样本逻辑：
    7.2.1 标准问之间构造负样本对,包含与其他所有重复笛卡尔和去重标准问笛卡尔成对，构成负样本对
    7.2.2 不同标准问的相似问构造负样本对[默认重复10次；2个样本集]。
    7.2.3 额外扩展数据集[假设为标准问]与相似问构造负样本[正负样本差额量]
    7.2.4 额外扩展数据集[假设为相似问]与其他标准问,构造负样本对[原始数据量]
    7.2.5 
8.样本增强策略

"""

"""
data file:  Doctor_query_all_df_1009.xlsx; 
            Doctor_query_train_df_1009.xlsx; 
            Doctor_query_eval_df_1009.xlsx
"""
#data_file='data/simi_data/Disease_general_norm_6k_220929.xlsx'
data_file='data/simi_data/Med_general_norm_category_1p7w_220929.xlsx'
data_df=pd.read_excel(data_file)

print(data_df.shape)
print(data_df.head())

random_seed=66
random.seed(random_seed)


##########################################################
##  Method-1: 初始All-Exclusive
##########################################################

def data_split_list(data_list):
    """
    :param data_list: List
    将列表进行分割，分成两个互斥的部分，
    return：抛出随机某个元素，且后半个非空List
    """
    if len(data_list) == 1:
        data_list_pre = data_list * 4
    elif len(data_list) == 0:
        return None, None
    else:
        data_list_pre = data_list
    n = random.randint(0, len(data_list_pre) - 1)
    return data_list_pre.pop(n), data_list_pre

xx=list(range(6))
x,y=data_split_list(xx)
logger.info(f'x: {x}; y: {y}')


def get_positive_pair(data_list):
    """
    同一标签数据,中间分割，进行成对
    :param data_list:List
    return:List[(tuple_1,tuple_2),..],元素对列表
    """

    former_num = int(len(data_list) / 2)
    former_latter_pair = zip(data_list[:former_num], data_list[former_num:2 * former_num])
    return list(former_latter_pair)

xx=list(range(5))
x=get_positive_pair(xx)
logger.info(f'x: {x}')

def get_pair_df(stand_simi_dict, force_simi_comb=True):
    """
    不同标准问中相似问之间,构成负样本对；正相似问之间，构成正样本对
    :param query_dict: {'Stand_text':'pos_Simi_text'},针对[标准问-正相似问列表]字典，
    return: 标准问之间负样本-df,相似问之间正样本-df,新的{'Stand_text':'rest-Simi_text'}字典
    """
    ## 存储不同标准问中的相似问，然后前后分割，构成负样本
    stand_simi_text_list = []
    ## 相似问之间，构成正样本
    pos_simi_text_list = []
    ## 标准问随机抽取相似问，构成字典
    stand_rest_simi_dict = {}
    for sub_stand_text, sub_simi_text_list in stand_simi_dict.items():
        ## sub_stand_text,标准问 ;sub_simi_text_list,相似问
        rand_simi_text, rest_simi_text_list = data_split_list(sub_simi_text_list)
        if rand_simi_text == None:
            continue
        stand_simi_text_list.append(rand_simi_text)
        if force_simi_comb:
            ## 强制相似样本组合成对
            rest_simi_text_pair_list = get_positive_pair(rest_simi_text_list)
            pos_simi_text_list.extend(rest_simi_text_pair_list)
        elif len(rest_simi_text_list) // 2:
            ##随机逻辑，当为奇数时，相似样本组合成对
            rest_simi_text_pair_list = get_positive_pair(rest_simi_text_list)
            pos_simi_text_list.extend(rest_simi_text_pair_list)
        ## 相似样本不组合成对
        stand_rest_simi_dict.update({sub_stand_text: rest_simi_text_list})
    ## 标准问顺序，构成成对，最终负样本，
    neg_stand_pair_list = get_positive_pair(stand_simi_text_list)
    return pd.DataFrame(neg_stand_pair_list), pd.DataFrame(pos_simi_text_list), stand_rest_simi_dict


xx={'a':list('bcdef'),'h':list('ijkmn')}
logger.info(f'stand_simi_dict: {xx}')
neg_df,pos_df,stand_rest_simi_dict=get_pair_df(xx)
logger.info(f'processed stand_simi_dict: {xx}')
logger.info(f'neg_df:\n{neg_df}')
logger.info(f'pos_df:\n{pos_df}')
logger.info(f'stand_rest_simi_dict: {stand_rest_simi_dict}')

## empty dict
xx={}
logger.info(f'stand_simi_dict: {xx}')
neg_df,pos_df,stand_rest_simi_dict=get_pair_df(xx)

from itertools import combinations,product

xx=['ab','cd','ef']
x_com=combinations(xx,2)
logger.info('combination: {}'.format(list(x_com)))

x=['a','b']
y=['c','d','e']
xy=list(product(x,y))
logger.info('product x_len: {}, y_len: {}; x_y_len: {}.'.format(len(x),len(y),len(xy)))
logger.info('product examples:: {}'.format(list(xy)))


def sample_focus_neg_df_pre(data_df):
    """
    标准问之间，负样本；正相似问之间，正样本
    形成最终dataframe
    :param: DataFrame,原始数据。
    return: train-df,基于不同标准问的相似问构造负样本-df,相似问正样本-df(可以为空),组合最终df
    """
    data = data_df.copy()
    ## 标准问对应相似问列表字典
    stand_simi_dict = {}
    for one in data.groupby('Stand_text'):
        standard_query = one[0]
        sim_query = one[1]['Simi_text'].tolist()
        if standard_query in sim_query:
            pass
        else:
            sim_query.append(standard_query)
        stand_simi_dict.update({standard_query: sim_query})
    stand_simi_new_dict = stand_simi_dict.copy()
    neg_stand0simi_pair_df = []
    ## 重复10次
    ## 超过一定次数，只有负样本数据；正样本数据为空。
    for i in range(10):
        sub_neg_stand0simi_df, sub_pos_simi_df, stand_simi_new_dict = get_pair_df(stand_simi_new_dict,force_simi_comb=False)
        neg_stand0simi_pair_df.append(sub_neg_stand0simi_df)
    sub_neg_stand0simi_df, pos_simi_df, stand_simi_new_dict = get_pair_df(stand_simi_new_dict,force_simi_comb=True)
    neg_stand0simi_pair_df.append(sub_neg_stand0simi_df)
    neg_stand_simi_pair_df = pd.concat(neg_stand0simi_pair_df, axis=0,sort=False)
    logger.info(f'pos_simi_df:\n{pos_simi_df.head()}')
    neg_stand_simi_pair_df['label'] = 0
    pos_simi_df['label'] = 1
    train_df = pd.concat([neg_stand_simi_pair_df, pos_simi_df], axis=0,sort=False).reset_index(drop=True)
    train_df.columns=['Stand_text', 'Simi_text', 'label']
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    return train_df


xx=pd.DataFrame({'Stand_text':['a']*5+['h']*5,'Simi_text':list('bcdef')+list('ijkmn')})
logger.info(f'xx:\n{xx}')

train_df=sample_focus_neg_df_pre(xx)
logger.info(f'train_df:\n{train_df}')



def sample_focus_simi_neg_df(data_df,extra_file):
    """
    重点基于相似语料构造负样本；如果样本比较多，可能顺便有正样本
    1. 基于基于不同标准问的相似问构造负样本-df的base-method,生成df
    2. 额外样本[假设为标准问]与相似样本[原始相似]，构造负样本集。
    :param data_df:DataFrame,原始数据。
    :param extra_file:额外扩展文件，csv格式
    return:train-df,基于1，2两种思路构造负样本训练集。
    """
    train_df1 = sample_focus_neg_df_pre(data_df)
    train_df2 = sample_focus_neg_df_pre(data_df)
    train_data_pre = pd.concat([train_df1, train_df2], axis=0)
    label_dist = train_data_pre.label.value_counts()
    if label_dist.shape[0]>1:
        diff = label_dist[1] - label_dist[0]
        #正负样本数量差额
        diff = abs(diff)
        #print('diff: ',diff)
        data = data_df.copy()
        simi_text_series = data.loc[:, 'Simi_text'].sample(diff, replace=True)
    else:
        diff=int(label_dist.shape[0]/2)
        simi_text_series = train_data_pre.loc[:, 'Simi_text'].sample(diff)
    if extra_file:
        corpus_extra = pd.read_csv(extra_file, header=None)
        logger.info('Extra  data shape: {}'.format(corpus_extra.shape))
        if diff < corpus_extra.shape[0]:
            extra_neg_text = corpus_extra.sample(diff).loc[:, 0]
        else:
            extra_neg_text = corpus_extra.sample(diff, replace=True).loc[:, 0]
        extra_neg_df = pd.DataFrame({'Simi_text': simi_text_series.tolist(), 'Stand_text': extra_neg_text.tolist(), 'label': 0})
        extra_neg_df = extra_neg_df.loc[:, ['Stand_text', 'Simi_text', 'label']]
        all_data = pd.concat([train_data_pre, extra_neg_df], axis=0)
    else:
        all_data=train_data_pre
    all_data = all_data.sample(frac=1)
    return all_data


xx=pd.DataFrame({'Stand_text':['a']*5+['h']*5,'Simi_text':list('bcdef')+list('ijkmn')})
logger.info(f'xx:\n{xx}')

extra_file=None
# extra_file='data/severe_simi_neg_extra_corpus.csv'
train_df=sample_focus_simi_neg_df(xx,extra_file)
logger.info(f'train_df:\n{train_df}')


def build_sample_df(data_df,extra_file):
    """
    1. 正样本自抽样，扩充正样本数量,构造正样本对
    2. 额外扩展样本与其他标准问,构造负样本对
    3. 当前标准问与其他标准问，包含与其他所有重复笛卡尔和去重标准问笛卡尔成对，构成负样本对
    :param data_df: 只有正样本的原始df
    :param extra_file: 额外扩展文件，csv格式
    return
    """
    ## 1. 正样本自抽样，扩充正样本数量
    data = data_df.loc[:,['Stand_text','Simi_text']].copy()
    # 自抽样扩充正样本数量
    data_pos_sample_df1 = data.sample(frac=0.5)
    data_pos_sample_df2 = data.sample(frac=1.5, replace=True)
    data_pos_sample_df3 = data.loc[:, ['Simi_text', 'Stand_text']].sample(frac=0.5)
    standard_query = data['Stand_text'].unique().tolist()
    # 2. 额外扩展样本与其他标准问，构造负样本。
    if extra_file:
        data_extra = pd.read_csv(extra_file, header=None)
        data_extra = data_extra[0].unique().tolist()
        ## 标准问与额外样本笛卡尔积获得负样本对
        data_neg_extra_df4 = pd.DataFrame(list(itertools.product(standard_query, data_extra)))
        data_neg_extra_df4 = data_neg_extra_df4.sample(len(data))
    else:
        data_neg_extra_df4=pd.DataFrame()
    #3.1 当前标准问与其他标准问，与其他所有重复标准问笛卡尔
    neg_stand_pair_list = []
    for sub_stand in standard_query:
        data2 = data[data['Stand_text'] != sub_stand]
        other_stand = data2['Stand_text'].tolist()
        neg_stand_pair_list.extend(list(itertools.product([sub_stand], other_stand)))
    data_neg_stand_df5 = pd.DataFrame(neg_stand_pair_list)
    data_neg_stand_df5 = data_neg_stand_df5.sample(len(data))
    #3.2 标准问之间去重标准问笛卡尔成对
    stand_unique_list = data['Stand_text'].unique().tolist()
    data_neg_stand_df6_pre = pd.DataFrame(list(itertools.combinations(stand_unique_list, 2)))
    if data_neg_stand_df6_pre.shape[0] > int(data.shape[0] / 2):
        data_neg_stand_df6 = data_neg_stand_df6_pre.sample(int(len(data) / 2))
    else:
        data_neg_stand_df6 = data_neg_stand_df6_pre.sample(int(len(data) / 2), replace=True)

    sample_df_list = [data_pos_sample_df1, data_pos_sample_df2, data_pos_sample_df3,
         data_neg_extra_df4, data_neg_stand_df5, data_neg_stand_df6]
    ## 前三个为正样本集，后三个为负样本集
    ##包含label字段
    train_norm_df_list = []
    for idx, sub_sample_df in enumerate(sample_df_list):
        logger.info('file : {}  shape: {}'.format(idx, sub_sample_df.shape))
        if sub_sample_df.shape[0]==0:
            continue

        sub_sample_df.columns = ['Simi_text', 'Stand_text']
        if idx < 3:
            sub_sample_df['label'] = 1
        else:
            sub_sample_df['label'] = 0
        train_norm_df_list.append(sub_sample_df)
    #重点基于相似问构造负样本和一定量的正样本
    train_focus_neg_df = sample_focus_simi_neg_df(data_df,extra_file)
    train_focus_neg_df = train_focus_neg_df.sample(len(data))
    train_norm_df_list.append(train_focus_neg_df)
    final_train_df = pd.concat(train_norm_df_list, axis=0, sort=False,ignore_index=True)
    final_train_df = final_train_df.sample(frac=1)
    final_train_df = final_train_df.dropna()
    final_train_df.loc[:, 'label'] = final_train_df.loc[:, 'label'].map(int)
    final_train_df = final_train_df.reset_index(drop=True)
    return final_train_df


xx=pd.DataFrame({'Stand_text':['a']*5+['h']*5,'Simi_text':list('bcdef')+list('ijkmn')})
logger.info(f'xx:\n{xx}')

#extra_file='data/severe_simi_neg_extra_corpus.csv'
extra_file=None
train_df=build_sample_df(xx,extra_file)
logger.info(f'train_df:\n{train_df}')

###
data_file='data/simi_data/Med_general_norm_NO_1p7w_220929.xlsx'
data_df=pd.read_excel(data_file)
logger.info(f'Raw label distribution:\n{data_df.label.value_counts()}')
train_df=build_sample_df(data_df,extra_file)
logger.info(f'label distribution:\n{train_df.label.value_counts()}')


train_df_unique=train_df.drop_duplicates()
logger.info(f'label unique distribution:\n{train_df_unique.label.value_counts()}')


def train_val_test(train_data_df,label_col="label", mode_index=False, balanced=True, dev_size=0.1, test_size=0.2, seed=666):
    """
    切分训练语料
    :param train_data_df: DataFrame,要求索引是顺序的(Index[0,1,2,...])
    :param mode_index:是否只返回DataFrame索引。
    :param balanced:是否采用向上均衡抽样。
    :param val_size:开发集大小，可以为0。
    :param test_size:测试集大小，可以为0。
    return: train_df,dev_df,test_df
    """
    train_data_df = train_data_df.reset_index(drop=True)
    ## 先解决test_size
    if test_size > 0.:
        train_id_pre, test_idx = train_test_split(range(train_data_df.shape[0]), \
                                                   test_size=test_size, random_state=seed)
    else:
        train_id_pre, test_idx = list(range(train_data_df.shape[0])), []
    logger.info('pre train size : {} test size: {}'.format(len(train_id_pre), len(test_idx)))
    train_data_pre = train_data_df.loc[train_id_pre, :]
    test_data = train_data_df.loc[test_idx, :]
    ## 再解决dev_size
    dev_size = round(dev_size / (1 - test_size), 2)
    if dev_size > 0.:
        x_tra_id_p,x_dev_idx,y_tra_lab_p,y_dev_lab=train_test_split(train_id_pre,train_data_pre.loc[train_id_pre,label_col],
                                                                   test_size=dev_size,random_state=seed)
    else:
        x_tra_id_p,x_dev_idx,y_tra_lab_p,y_dev_lab = train_id_pre,[],train_data_pre.loc[train_id_pre,label_col],[]
    x_tra_id_pre_arr = np.array(x_tra_id_p).reshape(-1, 1)
    if balanced:
        x_tra_id_balanced,y_tra_label_balanced=RandomOverSampler(random_state=seed).fit_resample(x_tra_id_pre_arr,
                                                                                             y_tra_lab_p)
        x_tra_idx = [x[0] for x in x_tra_id_balanced]
    else:
        x_tra_idx=x_tra_id_p
    train_idx, dev_idx = x_tra_idx, x_dev_idx
    logger.info('pro train size : {}  dev size : {}  test size: {}'.format(len(train_idx), len(dev_idx), len(test_idx)))
    train_data = train_data_pre.loc[train_idx, :]
    dev_data = train_data_pre.loc[dev_idx, :]
    if mode_index:
        return train_idx, dev_idx, test_idx
    else:
        return train_data, dev_data, test_data

logger.info('Raw data shape: {}'.format(data_df.shape))
logger.info('Raw label dist:\n{}'.format(data_df.label.value_counts()))
train_data, dev_data, test_data=train_val_test(data_df,dev_size=0.1, test_size=0.1)
logger.info('train_data shape: {}'.format(train_data.shape))
logger.info('train_data label dist:\n{}'.format(train_data.label.value_counts()))
logger.info(f'train_data:\n{train_data.head()}')
logger.info(f'dev_data:\n{dev_data.head()}')
logger.info(f'test_data:\n{test_data.head()}')



##########################################################
##  Method-2: Outer-Exclusive
##########################################################
""" 
Method-2.外互斥正负样本抽样成对逻辑:
    正样本逻辑
    M2-1.1 自抽样复制扩充正样本对数量[3次不同随机操作]。
    M2-1.2 正相似问之间构造正样本[原始样本量1.5倍]
    M2-1.3 
    负样本逻辑：
    M2-2.1 [不同种类]-标准问之间构造负样本对,包含与其他所有重复笛卡尔和去重标准问笛卡尔成对，构成负样本对
    M2-2.2 [不同种类]-标准问的相似问构造负样本对[默认重复10次；2个样本集]
    M2-2.3 额外扩展数据集(假设为标准问)与相似问构造负样本[正负样本差额量]
    M2-2.4 额外扩展数据集(假设为相似问)与其他标准问,构造负样本对[原始数据量]
    M2-2.5 
最后每种抽样标记📌抽样方法编号

Input :Category,Stand_text,Simi_text为正样本数据df
output:Stand_text,Simi_text,label,type共4个字段
"""
""" 
keep method: 1. data_split_list; 2. get_positive_pair; 3.get_pair_df
"""

def Outer_Exc_get_pair_df(cate_2_simi_dict, force_simi_comb=True):
    """
    不同种类中相似问之间,构成负样本对；正相似问之间，构成正样本对
    :param query_dict: {'Category':[List[Cate-Simi_text],List[Stand-Simi_text]]},针对[标准问-正相似问列表]字典
    :return:不同种类-相似问之间负样本-df;相似问之间正样本-df;{'Category':[List[rest-Cate-Simi_text],List[rest-Stand-Simi_text]]}
    """
    ## 存储不同种类中的相似问，构成负样本
    cate_simi_text_list = []
    ## 相似问之间，构成正样本
    pos_simi_text_list = []
    ## 种类-[相似问List,标准问-相似问List]，构成字典
    cate_rest_simi_dict = {}
    for sub_category, sub_cate_cont in cate_2_simi_dict.items():
        sub_cate_simi_text_list = sub_cate_cont[0]
        sub_stand_simi_block_list = sub_cate_cont[1]
        rand_simi_text, rest_simi_text_list = data_split_list(sub_cate_simi_text_list)
        if rand_simi_text == None:
            continue
        cate_simi_text_list.append(rand_simi_text)
        sub_stand_simi_block_list_new = []
        for sub_stand_simi_list in sub_stand_simi_block_list:
            if force_simi_comb:
                ## 强制相似样本组合成对
                rest_simi_text_pair_list = get_positive_pair(sub_stand_simi_list)
                pos_simi_text_list.extend(rest_simi_text_pair_list)
            elif len(sub_stand_simi_list) // 2:
                ##随机逻辑，当为奇数时，相似样本组合成对
                rest_simi_text_pair_list = get_positive_pair(sub_stand_simi_list)
                pos_simi_text_list.extend(rest_simi_text_pair_list)
            ## 相似样本不组合成对
            ## 同样随机删除正相似问一个元素
            _, sub_stand_simi_list2 = data_split_list(sub_stand_simi_list)
            sub_stand_simi_block_list_new.append(sub_stand_simi_list2)
        cate_rest_simi_dict.update({sub_category: [rest_simi_text_list, sub_stand_simi_block_list_new]})
    ## 不同种类的负样本，最终负样本，
    neg_cate_pair_list = get_positive_pair(cate_simi_text_list)
    return pd.DataFrame(neg_cate_pair_list), pd.DataFrame(pos_simi_text_list), cate_rest_simi_dict


def Outer_Exc_sample_df_pre(data_df):
    """
    不同种类-标准问的相似问，负样本；标准问-正相似问之间，正样本
    :param: DataFrame,原始数据。
    return: train-df,基于不同标准问的相似问构造负样本-df,相似问正样本-df,组合最终df
    """
    data = data_df.copy()
    ## {种类:[种类下面所有相似集List，[标准问下面的相似问List]]
    cate_2_simi_dict = {}
    ## 存储种类下相似样本数量
    cate_simi_sample_nums_list = []
    for cate_group_item in data.groupby(['Category']):
        cate_stand_tuple = cate_group_item[0]
        stand_text_list = cate_group_item[1]['Stand_text'].tolist()
        ## 针对种类-相似问，构造负样本对
        cate_simi_text_list = cate_group_item[1]['Simi_text'].tolist()
        for sub_stand_text in stand_text_list:
            if sub_stand_text in cate_simi_text_list:
                pass
            else:
                cate_simi_text_list.append(sub_stand_text)
        cate_simi_sample_nums_list.append(len(cate_simi_text_list))
        ## 不同标准问对应的相似问，用于正样本对
        stand_simi_block_list = []
        for stand_group_item in cate_group_item[1].groupby(['Stand_text']):
            simi_text_list = stand_group_item[1]['Simi_text'].tolist()
            stand_simi_block_list.append(simi_text_list)
        cate_2_simi_dict.update({cate_stand_tuple: [cate_simi_text_list, stand_simi_block_list]})
    cate_simi_new_dict = cate_2_simi_dict.copy()
    neg_stand0simi_pair_df = []
    pos_stand0simi_pair_df = []
    ## 重复10次;或者
    ## 同时保留正负样本对。
    loop_nums=pd.Series(cate_simi_sample_nums_list).describe()['25%']
    loop_nums=int(loop_nums)
    logger.info(f'Outer Simi Loop nums: {loop_nums}')
    for i in range(loop_nums):
        sub_neg_stand0simi_df, sub_pos_simi_df, stand_simi_new_dict = Outer_Exc_get_pair_df(cate_simi_new_dict,force_simi_comb=False)
        neg_stand0simi_pair_df.append(sub_neg_stand0simi_df)
        pos_stand0simi_pair_df.append(sub_pos_simi_df)
    sub_neg_stand0simi_df, sub_pos_simi_df, stand_simi_new_dict = Outer_Exc_get_pair_df(cate_simi_new_dict,force_simi_comb=True)
    neg_stand0simi_pair_df.append(sub_neg_stand0simi_df)
    pos_stand0simi_pair_df.append(sub_pos_simi_df)
    neg_stand_simi_pair_df = pd.concat(neg_stand0simi_pair_df, axis=0,sort=False)
    pos_stand_simi_pair_df = pd.concat(pos_stand0simi_pair_df, axis=0, sort=False)
    neg_stand_simi_pair_df['label'] = 0
    pos_stand_simi_pair_df['label'] = 1
    train_df = pd.concat([neg_stand_simi_pair_df, pos_stand_simi_pair_df], axis=0,sort=False).reset_index(drop=True)
    train_df.columns=['Stand_text', 'Simi_text', 'label']
    train_df['type']='Outer_Simi'
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    return train_df

#### M2-Outer Dev
## {种类:[种类下面所有相似集List，[标准问下面的相似问List]]
cate_2_simi_dict = {}
## 存储种类下相似样本数量
cate_simi_sample_nums_list=[]
for cate_group_item in data_df.head(100).groupby(['Category']):
    cate_stand_tuple = cate_group_item[0]
    stand_text_list = cate_group_item[1]['Stand_text'].tolist()
    ## 针对种类-相似问，构造负样本对
    cate_simi_text_list=cate_group_item[1]['Simi_text'].tolist()
    for sub_stand_text in stand_text_list:
        if sub_stand_text in cate_simi_text_list:
            pass
        else:
            cate_simi_text_list.append(sub_stand_text)
    cate_simi_sample_nums_list.append(len(cate_simi_text_list))
    ## 不同标准问对应的相似问，用于正样本对
    stand_simi_block_list=[]
    for stand_group_item in cate_group_item[1].groupby(['Stand_text']):
        simi_text_list=stand_group_item[1]['Simi_text'].tolist()
        stand_simi_block_list.append(simi_text_list)
    cate_2_simi_dict.update({cate_stand_tuple:[cate_simi_text_list,stand_simi_block_list]})

pprint(cate_2_simi_dict)
print(pd.Series(cate_simi_sample_nums_list).describe())
print(pd.Series(cate_simi_sample_nums_list).describe()['25%'])

## 存储不同种类中的相似问，构成负样本
cate_simi_text_list = []
## 相似问之间，构成正样本
pos_simi_text_list = []
## 种类-[相似问List,标准问-相似问List]，构成字典
cate_rest_simi_dict = {}
force_simi_comb=False
for sub_category, sub_cate_cont in cate_2_simi_dict.items():
    sub_cate_simi_text_list=sub_cate_cont[0]
    sub_stand_simi_block_list=sub_cate_cont[1]
    rand_simi_text, rest_simi_text_list = data_split_list(sub_cate_simi_text_list)
    if rand_simi_text == None:
        continue
    cate_simi_text_list.append(rand_simi_text)
    sub_stand_simi_block_list_new=[]
    for sub_stand_simi_list in sub_stand_simi_block_list:
        if force_simi_comb:
            ## 强制相似样本组合成对
            rest_simi_text_pair_list = get_positive_pair(sub_stand_simi_list)
            pos_simi_text_list.extend(rest_simi_text_pair_list)
        elif len(sub_stand_simi_list) // 2:
            ##随机逻辑，当为奇数时，相似样本组合成对
            rest_simi_text_pair_list = get_positive_pair(sub_stand_simi_list)
            pos_simi_text_list.extend(rest_simi_text_pair_list)
        ## 相似样本不组合成对
        ## 同样随机删除正相似问一个元素
        _,sub_stand_simi_list2=data_split_list(sub_stand_simi_list)
        sub_stand_simi_block_list_new.append(sub_stand_simi_list2)
    cate_rest_simi_dict.update({sub_category: [rest_simi_text_list,sub_stand_simi_block_list_new]})
## 不同种类的负样本，最终负样本，
neg_cate_pair_list = get_positive_pair(cate_simi_text_list)

neg_cate_simi_df=pd.DataFrame(neg_cate_pair_list)
pos_stand_simi_df=pd.DataFrame(pos_simi_text_list)

logger.info(f'neg_cate_simi_df:\n{neg_cate_simi_df.head()}')
logger.info(f'pos_stand_simi_df:\n{pos_stand_simi_df.head()}')

#### Test

xx=pd.DataFrame({'Category':['A']*5+['H']*5,'Stand_text':['a1']*3+['a2']*2+['h1']*3+['h2']*2,
                 'Simi_text':list('bcdef')+list('ijkmn')})
logger.info(f'xx:\n{xx}')

train_df=Outer_Exc_sample_df_pre(xx)
logger.info(f'train_df:\n{train_df}')


def Outer_Exc_sample_focus_simi_neg_df(data_df,extra_file):
    """
    重点基于种类-相似语料构造负样本
    1. 基于基于不同种类的相似问构造负样本-df的base-method,生成df
    2. 额外样本[假设为标准问]与相似样本[原始相似]，构造负样本集。
    :param data_df:DataFrame,原始数据。
    :param extra_file:额外扩展文件，csv格式
    return:train-df,基于1，2两种思路构造负样本训练集。
    """
    train_df1 = Outer_Exc_sample_df_pre(data_df)
    train_df2 = Outer_Exc_sample_df_pre(data_df)
    train_data_pre = pd.concat([train_df1, train_df2], axis=0)
    label_dist = train_data_pre.label.value_counts()
    if label_dist.shape[0]>1:
        diff = label_dist[1] - label_dist[0]
        #正负样本数量差额
        diff = abs(diff)
        #print('diff: ',diff)
        data = data_df.copy()
        simi_text_series = data.loc[:, 'Simi_text'].sample(diff, replace=True)
    else:
        diff=int(label_dist/2)
        simi_text_series = train_data_pre.loc[:, 'Simi_text'].sample(diff)
    if extra_file:
        corpus_extra = pd.read_csv(extra_file, header=None)
        logger.info('Extra  data shape: {}'.format(corpus_extra.shape))
        if diff < corpus_extra.shape[0]:
            extra_neg_text = corpus_extra.sample(diff).loc[:, 0]
        else:
            extra_neg_text = corpus_extra.sample(diff, replace=True).loc[:, 0]
        extra_neg_df = pd.DataFrame({'Simi_text': simi_text_series.tolist(), 'Stand_text': extra_neg_text.tolist(), 'label': 0})
        extra_neg_df = extra_neg_df.loc[:, ['Stand_text', 'Simi_text', 'label']]
        extra_neg_df['type']='Extra_Simi'
        ## 进一步增加额外样本与相似集构造负样本对，数量为原始数据集一半
        further_sample_nums=int(data_df.shape[0]*0.5)
        simi_raw_text_series=data_df.loc[:, 'Simi_text'].sample(further_sample_nums)
        further_extra_neg_series=corpus_extra.sample(further_sample_nums).loc[:, 0]
        extra_neg_df2 = pd.DataFrame(
            {'Simi_text': simi_raw_text_series.tolist(), 'Stand_text': further_extra_neg_series.tolist(), 'label': 0})
        extra_neg_df2 = extra_neg_df2.loc[:, ['Stand_text', 'Simi_text', 'label']]
        extra_neg_df2['type'] = 'Extra_Simi'
        all_data = pd.concat([train_data_pre, extra_neg_df2], axis=0,sort=False)
    else:
        all_data=train_data_pre
    all_data = all_data.sample(frac=1)
    return all_data

#### Test
xx=pd.DataFrame({'Category':['A']*5+['H']*5,'Stand_text':['a1']*3+['a2']*2+['h1']*3+['h2']*2,
                 'Simi_text':list('bcdef')+list('ijkmn')})
logger.info(f'xx:\n{xx}')

#extra_file='data/severe_simi_neg_extra_corpus.csv'
extra_file=None
train_df=Outer_Exc_sample_focus_simi_neg_df(xx,extra_file)
logger.info(f'train_df:\n{train_df}')


def Outer_Exc_build_sample_df(data_df,extra_file):
    """
    1. 正样本自抽样，扩充正样本数量,构造正样本对
    2. 额外扩展样本与其他标准问,构造负样本对
    3. 当前标准问与其他标准问，包含与其他所有重复笛卡尔和去重标准问笛卡尔成对，构成负样本对
    :param data_df: 只有正样本的原始df,必须包含Category,Stand_text,Simi_text,共3个字段
    :param extra_file: 额外扩展文件，csv格式
    return DataFrame: 抽样组成最终df,必须包含Category,Stand_text,Simi_text,label共4个字段
    """
    ## 1. 正样本自抽样，扩充正样本数量
    data = data_df.copy()
    df_cols=['Stand_text','Simi_text','label','type']
    # 自抽样扩充正样本数量
    data_pos_sample_df1 = data.sample(frac=0.5)
    data_pos_sample_df2 = data.sample(frac=1.5, replace=True)
    data_pos_sample_df3 = data.loc[:, ['Category','Simi_text', 'Stand_text']].sample(frac=0.5)
    data_pos_sample_df3.columns=['Category','Stand_text','Simi_text']
    standard_query = data['Stand_text'].unique().tolist()
    # 2. 额外扩展样本与其他标准问，构造负样本。
    if extra_file:
        data_extra = pd.read_csv(extra_file, header=None)
        data_extra = data_extra[0].unique().tolist()
        ## 标准问与额外样本笛卡尔积获得负样本对
        data_neg_extra_df4 = pd.DataFrame(list(itertools.product(standard_query, data_extra)))
        data_neg_extra_df4 = data_neg_extra_df4.sample(data.shape[0])
        data_neg_extra_df4.columns = ['Stand_text', 'Simi_text']
        data_neg_extra_df4['label']=0
        data_neg_extra_df4['type'] = 'Extra_Stand'
        data_neg_extra_df4=data_neg_extra_df4.loc[:,df_cols]
    else:
        data_neg_extra_df4=pd.DataFrame()
    #3.1 当前标准问与其他标准问，与其他所有重复标准问笛卡尔
    neg_stand_pair_list = []
    for sub_stand in standard_query:
        data2 = data[data['Stand_text'] != sub_stand]
        other_stand = data2['Stand_text'].tolist()
        neg_stand_pair_list.extend(list(itertools.product([sub_stand], other_stand)))
    data_neg_stand_df5 = pd.DataFrame(neg_stand_pair_list)
    data_neg_stand_df5 = data_neg_stand_df5.sample(data.shape[0])
    #3.2 标准问之间去重标准问笛卡尔成对
    stand_unique_list = data['Stand_text'].unique().tolist()
    data_neg_stand_df6_pre = pd.DataFrame(list(itertools.combinations(stand_unique_list, 2)))
    if data_neg_stand_df6_pre.shape[0] > int(data.shape[0] / 2):
        data_neg_stand_df6 = data_neg_stand_df6_pre.sample(int(len(data) / 2))
    else:
        data_neg_stand_df6 = data_neg_stand_df6_pre.sample(int(len(data) / 2), replace=True)
    sample_pos_df = pd.concat([data_pos_sample_df1, data_pos_sample_df2, data_pos_sample_df3],axis=0,sort=False)
    sample_pos_df['label'] = 1
    sample_pos_df['type']='Self_Simi'
    sample_pos_df=sample_pos_df.loc[:,df_cols]
    sample_neg_df=pd.concat([data_neg_stand_df5, data_neg_stand_df6],axis=0,sort=False)
    sample_neg_df.columns=['Stand_text','Simi_text']
    sample_neg_df['label']=0
    sample_neg_df['type'] = 'Stand_Inner'
    sample_neg_df = sample_neg_df.loc[:, df_cols]
    #重点基于相似问构造负样本和一定量的正样本
    train_focus_neg_df = Outer_Exc_sample_focus_simi_neg_df(data_df,extra_file)
    train_focus_neg_df = train_focus_neg_df.sample(len(data))
    logger.info(f'sample_pos_df:\n{sample_pos_df.head()}')
    logger.info(f'data_neg_extra_df4:\n{data_neg_extra_df4.head()}')
    logger.info(f'sample_neg_df:\n{sample_neg_df.head()}')
    logger.info(f'train_focus_neg_df:\n{train_focus_neg_df.head()}')
    final_train_df = pd.concat([sample_pos_df,data_neg_extra_df4,sample_neg_df,train_focus_neg_df], axis=0, sort=False)
    final_train_df = final_train_df.sample(frac=1)
    final_train_df = final_train_df.dropna()
    final_train_df.loc[:, 'label'] = final_train_df.loc[:, 'label'].map(int)
    final_train_df = final_train_df.reset_index(drop=True)
    return final_train_df

#### Test
xx=pd.DataFrame({'Category':['A']*5+['H']*5,'Stand_text':['a1']*3+['a2']*2+['h1']*3+['h2']*2,
                 'Simi_text':list('bcdef')+list('ijkmn')})
logger.info(f'xx:\n{xx}')

#extra_file='data/simi_data/severe_simi_neg_extra_corpus.csv'
extra_file=None
train_df=Outer_Exc_build_sample_df(xx,extra_file)
logger.info(f'train_df:\n{train_df}')


##########################
##  M-2: Data-Augment
##########################
""" 
DA-strategy
1.主要以加入标点符号为主，随机删掉为辅。
2.只针对正样本相似集处理；只针对字符串长度>=3的进行操作。
  insert-punctuation:0.8
  random_delete:0.1 
  keep_original:0.1
"""
"""
EDA-方法
提出了四种简单的数据增强操作，包括：
1.同义词替换(Synonym Replacement, SR)：从句子中随机选取n个不属于停用词集的单词，并随机选择其同义词替换它们；
2.随机插入(Random Insertion, RI)：随机的找出句中某个不属于停用词集的词，并求出其随机的同义词，将该同义词插入句子的一个随机位置。重复n次；
3.随机交换(Random Swap, RS)：随机的选择句中两个单词并交换它们的位置。重复n次；
4.随机删除(Random Deletion, RD)：以  的概率，随机的移除句中的每个单词；
5.同义词替换不使用词表，而是使用词向量或者预训练语言模型；
6.通过在在文本中插入一些符合或者词语，来增加噪声；如原始文本中随机插入一些标点符号
7.将句子通过翻译器翻译成另外一种语言再翻译回来的回译手段
"""
"""
(AEDA)An Easier Data Augmentation Technique for Text Classification
增加噪声；如原始文本中随机插入一些标点符号
Q：插入多少标点符号？A：从1到三分之一句子长度中，随机选择一个数，作为插入标点符号的个数。
Q：为什么是1到三分之一句长？A：作者表示，既想每个句子中有标点符号插入，增加句子的复杂性；又不想加入太多标点符号，
                         过于干扰句子的语义信息，并且太多噪声对模型可能有负面影响。
Q：句子插入标点符号的位置如何选取？A：随机插入。
Q：标点符号共包含哪些？A：主要有6种，“.”、“;”、“?”、“:”、“!”、“,”。
Q：AEDA比EDA效果好的理论基础是什么？A：作者认为，EDA方法，如论是同义词替换，还是随机替换、随机插入、随机删除，都改变了原始文本的序列信息；
                               而AEDA方法，只是插入标点符号，对于原始数据的序列信息修改不明显。个人理解，通过词语修改的方法，
                               与原始语义改变可以更加负面；而仅插入一些标点符号，虽然增加了噪声，但是原始文本的语序并没有改变.
实验结果：当数据集较小时，数据增强倍数越大，效果提升的越明显；
        但是当数据量较大时，数据增强倍数越大，效果提升将会下降。
"""


def eda_insert_punctuation(sentence, punc_ratio=0.3):
    PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
    words = list(sentence)
    new_line = []
    insert_punc_nums = random.randint(1, int(punc_ratio * len(words) + 1))
    insert_idx = random.sample(range(0, len(words)), insert_punc_nums)
    for sub_idx, word in enumerate(words):
      if sub_idx in insert_idx:
        new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
        new_line.append(word)
      else:
        new_line.append(word)
    new_line = ''.join(new_line)
    return new_line

xx="但是当数据量较大时，数据增强倍数越大，效果提升将会下降。"
xx_res=eda_insert_punctuation(xx)
logger.info(f'Input text: {xx}')
logger.info(f'eda_insert_punctuation Result: {xx_res}')


### 随机删除
def eda_random_delete(words, drop_proba=0.2):
    if len(words) == 1:
        return words
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > drop_proba:
            new_words.append(word)
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]
    return ''.join(new_words)

xx="但是当数据量较大时，数据增强倍数越大，效果提升将会下降。"
xx_res=eda_random_delete(xx)
logger.info(f'              Input text: {xx}')
logger.info(f'eda_random_delete Result: {xx_res}')


def EDA_build_sampler_df(data_df,insert_proba=0.8,delete_proba=0.1):
    """
    只针对正样本相似集处理；只针对字符串长度 >= 3的进行操作。
    insert_punctuation: 0.8
    random_delete: 0.1
    keep_original: 0.1
    :param data_df: 只有正样本的原始df,必须包含Category,Stand_text,Simi_text,共3个字段
    :return DataFrame: 抽样组成最终df,必须包含Category,Stand_text,Simi_text,label共4个字段
    """
    def eda_operation(text:str):
        if len(text.strip())>=3:
            rand = random.uniform(0, 1)
            if rand>1-insert_proba:
                eda_text=eda_insert_punctuation(text)
            elif rand>1-delete_proba:
                eda_text = eda_random_delete(text)
            else:
                eda_text=text
        else:
            return text
        return eda_text

    data=data_df.copy()
    data.loc[:,'eda_text']=data.loc[:,'Simi_text'].map(eda_operation)
    data=data.loc[:,['Stand_text','eda_text']]
    data.rename(columns={'eda_text':'Simi_text'},inplace=True)
    data['label']=1
    data['type'] = 'EDA'
    return data


#### Test
xx=pd.DataFrame({'Category':['A']*5+['H']*5,'Stand_text':['a1']*3+['a2']*2+['h1']*3+['h2']*2,
                 'Simi_text':list('bcd')+['adfasdfa','adfafaaad']+list('ijk')+["长句子相对于短句子","存在一个特性"]})
logger.info(f'xx:\n{xx}')

train_df=EDA_build_sampler_df(xx)
logger.info(f'train_df:\n{train_df}')

##########################################################
##  Method-3: 初始Inner-Exclusive
##########################################################
""" 
Method-3.内互斥正负样本抽样成对逻辑:
    正样本逻辑
    M3-1.1 自抽样复制扩充正样本对数量[3次不同随机操作]为主。
    M3-1.2 标准问-正相似问之间构造正样本[原始样本量1.5倍]为辅，占原始正样本的25%。
    M3-1.3 
    负样本逻辑：
    M3-2.1 [相同种类-标准问]之间构造负样本对,包含与其他所有重复笛卡尔和去重标准问笛卡尔成对，构成负样本对
    M3-2.2 [相同种类-标准问]的相似问构造负样本对[默认重复10次；2个样本集]。
    M3-2.3 额外扩展数据集(假设为标准问)与相似问构造负样本[正负样本差额量]
    M3-2.4 额外扩展数据集(假设为相似问)与其他标准问,构造负样本对[原始数据量]
    M3-2.5 
最后每种抽样标记📌抽样方法编号
"""
""" 
keep method: 1. data_split_list; 2. get_positive_pair; 3.get_pair_df

## 目前基于最近pandas>1.0的groupby的特性更改
"""

def Inner_Exc_get_pair_df(cate_2_simi_dict, force_simi_comb=True):
    """
    不同种类中相似问之间,构成负样本对；正相似问之间，构成正样本对
    :param query_dict: {'Category':[List[Cate-Simi_text],List[Stand-Simi_text]]},针对[标准问-正相似问列表]字典
    return:不同种类-相似问之间负样本-df;相似问之间正样本-df;{'Category':[List[rest-Cate-Simi_text],List[rest-Stand-Simi_text]]}
    """
    ## 存储相同种类中的不同标准问-相似问，构成负样本
    neg_diff_stand_2_simi_pair_list = []
    ## 相似问之间，构成正样本
    pos_simi_text_pair_list = []
    ## 种类-[rest-标准问_相似问List]，构成字典
    cate_rest_simi_dict = {}
    for sub_category, sub_stand_simi_cont in cate_2_simi_dict.items():
        sub_cate_stand_simi_block_list_new = []
        ## 存储相同种类中的不同标准问-相似问，构成负样本
        sub_stand_2_simi_text_list = []
        for sub_stand_simi_list in sub_stand_simi_cont:
            rand_simi_text, rest_simi_text_list = data_split_list(sub_stand_simi_list)
            if rand_simi_text == None:
                continue
            sub_stand_2_simi_text_list.append(rand_simi_text)
            if force_simi_comb:
                ## 强制相似样本组合成对
                rest_simi_text_pair_list = get_positive_pair(rest_simi_text_list)
                pos_simi_text_pair_list.extend(rest_simi_text_pair_list)
            elif len(rest_simi_text_list) // 2:
                ##随机逻辑，当为奇数时，相似样本组合成对
                rest_simi_text_pair_list = get_positive_pair(rest_simi_text_list)
                pos_simi_text_pair_list.extend(rest_simi_text_pair_list)
            ## 相似样本不组合成对
            sub_cate_stand_simi_block_list_new.append(rest_simi_text_list)
        cate_rest_simi_dict.update({sub_category: sub_cate_stand_simi_block_list_new})
        ## 相同种类的不同标准问的相似问，构造负样本对，
        neg_cate_pair_list = get_positive_pair(sub_stand_2_simi_text_list)
        logger.info(f'neg_cate_pair_list: {neg_cate_pair_list}')
        neg_diff_stand_2_simi_pair_list.extend(neg_cate_pair_list)
    return pd.DataFrame(neg_diff_stand_2_simi_pair_list), pd.DataFrame(pos_simi_text_pair_list), cate_rest_simi_dict

def Inner_Exc_sample_df_pre(data_df):
    """
    相同种类-标准问的相似问，负样本；标准问-正相似问之间，正样本
    :param: DataFrame,原始数据。
    return: train-df,基于不同标准问的相似问构造负样本-df,相似问正样本-df,组合最终df
    """
    data = data_df.copy()
    ## {种类:[标准问下面的相似问List,...,[]]
    cate_2_simi_dict = {}
    ## 存储种类下相似样本数量
    cate_simi_sample_nums_list = []
    for cate_group_item in data.groupby(['Category']):
        cate_stand_tuple = cate_group_item[0][0]##新版pandas-gropyby
        ## 不同标准问对应的相似问，用于正样本对
        stand_simi_block_list = []
        for stand_group_item in cate_group_item[1].groupby(['Stand_text']):
            stand_text = stand_group_item[0][0]##新版pandas-gropyby
            simi_text_list = stand_group_item[1]['Simi_text'].tolist()
            if stand_text not in simi_text_list:
                simi_text_list.append(stand_text)
            cate_simi_sample_nums_list.append(len(simi_text_list))
            stand_simi_block_list.append(simi_text_list)
        cate_2_simi_dict.update({cate_stand_tuple: stand_simi_block_list})
    cate_simi_new_dict = cate_2_simi_dict.copy()
    neg_stand0simi_pair_df = []
    pos_stand0simi_pair_df = []
    ## 重复10次;或者
    ## 同时保留正负样本对。
    loop_nums=pd.Series(cate_simi_sample_nums_list).describe()['50%']
    loop_nums=int(loop_nums)
    logger.info(f'Outer Simi Loop nums: {loop_nums}')
    for i in range(loop_nums):
        sub_neg_stand0simi_df, sub_pos_simi_df, stand_simi_new_dict = Inner_Exc_get_pair_df(cate_simi_new_dict,force_simi_comb=False)
        logger.info(f'sub_neg_stand0simi_df: {sub_neg_stand0simi_df.head()}')
        neg_stand0simi_pair_df.append(sub_neg_stand0simi_df)
        pos_stand0simi_pair_df.append(sub_pos_simi_df)
    sub_neg_stand0simi_df, sub_pos_simi_df, stand_simi_new_dict = Inner_Exc_get_pair_df(cate_simi_new_dict,force_simi_comb=True)
    neg_stand0simi_pair_df.append(sub_neg_stand0simi_df)
    pos_stand0simi_pair_df.append(sub_pos_simi_df)
    neg_stand_simi_pair_df = pd.concat(neg_stand0simi_pair_df, axis=0,sort=False)
    pos_stand_simi_pair_df = pd.concat(pos_stand0simi_pair_df, axis=0, sort=False)
    neg_stand_simi_pair_df['label'] = 0
    neg_stand_simi_pair_df['type'] = 'Neg_Inner_Diff_Simi'
    pos_stand_simi_pair_df['label'] = 1
    pos_stand_simi_pair_df['type'] = 'Pos_Inner_Simi'
    train_df = pd.concat([neg_stand_simi_pair_df, pos_stand_simi_pair_df], axis=0,sort=False).reset_index(drop=True)
    train_df.columns=['Stand_text', 'Simi_text', 'label','type']
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    return train_df


#### M3-Inner Dev
## {种类:[标准问下面的相似问List,...,[]]
cate_2_simi_dict = {}
## 存储种类下相似样本数量
cate_simi_sample_nums_list=[]
for cate_group_item in data_df.head(20).groupby(['Category']):
    cate_stand_tuple = cate_group_item[0][0]
    ## 不同标准问对应的相似问，用于正样本对
    stand_simi_block_list=[]
    for stand_group_item in cate_group_item[1].groupby(['Stand_text']):
        stand_text=stand_group_item[0][0]
        print("stand_text: ",stand_text)
        simi_text_list=stand_group_item[1]['Simi_text'].tolist()
        if stand_text not in simi_text_list:
            simi_text_list.append(stand_text)
        cate_simi_sample_nums_list.append(len(simi_text_list))
        stand_simi_block_list.append(simi_text_list)
    cate_2_simi_dict.update({cate_stand_tuple:stand_simi_block_list})

pprint(cate_2_simi_dict)
print(pd.Series(cate_simi_sample_nums_list).describe())
print(pd.Series(cate_simi_sample_nums_list).describe()['25%'])


## 存储相同种类中的不同标准问-相似问，构成负样本
neg_diff_stand_2_simi_pair_list = []
## 相似问之间，构成正样本
pos_simi_text_pair_list = []
## 种类-[rest-标准问_相似问List]，构成字典
cate_rest_simi_dict = {}
force_simi_comb=False
for sub_category, sub_stand_simi_cont in cate_2_simi_dict.items():
    sub_cate_stand_simi_block_list_new = []
    ## 存储相同种类中的不同标准问-相似问，构成负样本
    sub_stand_2_simi_text_list = []
    for sub_stand_simi_list in sub_stand_simi_cont:
        rand_simi_text, rest_simi_text_list = data_split_list(sub_stand_simi_list)
        if rand_simi_text == None:
            continue
        sub_stand_2_simi_text_list.append(rand_simi_text)
        if force_simi_comb:
            ## 强制相似样本组合成对
            rest_simi_text_pair_list = get_positive_pair(rest_simi_text_list)
            pos_simi_text_pair_list.extend(rest_simi_text_pair_list)
        elif len(rest_simi_text_list) // 2:
            ##随机逻辑，当为奇数时，相似样本组合成对
            rest_simi_text_pair_list = get_positive_pair(rest_simi_text_list)
            pos_simi_text_pair_list.extend(rest_simi_text_pair_list)
        ## 相似样本不组合成对
        sub_cate_stand_simi_block_list_new.append(rest_simi_text_list)
    cate_rest_simi_dict.update({sub_category: sub_cate_stand_simi_block_list_new})
    ## 相同种类的不同标准问的相似问，构造负样本对，
    neg_cate_pair_list = get_positive_pair(sub_stand_2_simi_text_list)
    neg_diff_stand_2_simi_pair_list.extend(neg_cate_pair_list)

neg_cate_simi_df=pd.DataFrame(neg_diff_stand_2_simi_pair_list)
pos_stand_simi_df=pd.DataFrame(pos_simi_text_pair_list)

logger.info(f'neg_cate_simi_df:\n{neg_cate_simi_df.head()}')
logger.info(f'pos_stand_simi_df:\n{pos_stand_simi_df.head()}')

#### Test

xx=pd.DataFrame({'Category':['A']*5+['H']*5,'Stand_text':['a1']*3+['a2']*2+['h1']*3+['h2']*2,
                 'Simi_text':list('bcdef')+list('ijkmn')})
logger.info(f'xx:\n{xx}')

train_df=Inner_Exc_sample_df_pre(xx)
logger.info(f'train_df:\n{train_df}')


def Inner_Exc_sample_focus_simi_neg_df(data_df,extra_file):
    """
    重点基于相同种类间-相似语料构造负样本
    1. 基于基于相同种类的相似问构造负样本-df的base-method,生成df
    2. 额外样本[假设为标准问]与相似样本[原始相似]，构造负样本集。
    :param data_df:DataFrame,原始数据。
    :param extra_file:额外扩展文件，csv格式
    return:train-df,基于1，2两种思路构造负样本训练集。
    """
    train_df1 = Inner_Exc_sample_df_pre(data_df)
    train_df2 = Inner_Exc_sample_df_pre(data_df)
    train_data_pre = pd.concat([train_df1, train_df2], axis=0)
    label_dist = train_data_pre.label.value_counts()
    if label_dist.shape[0]>1:
        diff = label_dist[1] - label_dist[0]
        #正负样本数量差额
        diff = abs(diff)
        #print('diff: ',diff)
        data = data_df.copy()
        simi_text_series = data.loc[:, 'Simi_text'].sample(diff, replace=True)
    else:
        diff=int(label_dist/2)
        simi_text_series = train_data_pre.loc[:, 'Simi_text'].sample(diff)
    if extra_file:
        corpus_extra = pd.read_csv(extra_file, header=None)
        logger.info('Extra  data shape: {}'.format(corpus_extra.shape))
        if diff < corpus_extra.shape[0]:
            extra_neg_text = corpus_extra.sample(diff).loc[:, 0]
        else:
            extra_neg_text = corpus_extra.sample(diff, replace=True).loc[:, 0]
        extra_neg_df = pd.DataFrame({'Simi_text': simi_text_series.tolist(), 'Stand_text': extra_neg_text.tolist(), 'label': 0})
        extra_neg_df = extra_neg_df.loc[:, ['Stand_text', 'Simi_text', 'label']]
        extra_neg_df['type']='Neg_Extra_Simi_diff'
        ## 进一步增加额外样本与相似集构造负样本对，数量为原始数据集正样本一样
        further_sample_nums=data_df.shape[0]
        logger.info(f'Extra further sample nums: {further_sample_nums}')
        simi_raw_text_series=data_df.loc[:, 'Simi_text'].sample(n=further_sample_nums)
        if further_sample_nums<corpus_extra.shape[0]:
            further_extra_neg_series=corpus_extra.sample(n=further_sample_nums).loc[:, 0]
        else:
            further_extra_neg_series = corpus_extra.sample(n=further_sample_nums,replace=True).loc[:, 0]
        extra_neg_df2 = pd.DataFrame(
            {'Simi_text': simi_raw_text_series.tolist(), 'Stand_text': further_extra_neg_series.tolist(), 'label': 0})
        extra_neg_df2 = extra_neg_df2.loc[:, ['Stand_text', 'Simi_text', 'label']]
        extra_neg_df2['type'] = 'Neg_Extra_Simi_rand'
        logger.info(f'extra_neg_df2 shape: {extra_neg_df2.shape}')
        all_data = pd.concat([train_data_pre,extra_neg_df,extra_neg_df2], axis=0,sort=False)
    else:
        all_data=train_data_pre
    all_data = all_data.sample(frac=1)
    return all_data

#### Test
xx=pd.DataFrame({'Category':['A']*5+['H']*5,'Stand_text':['a1']*3+['a2']*2+['h1']*3+['h2']*2,
                 'Simi_text':list('bcdef')+list('ijkmn')})
logger.info(f'xx:\n{xx}')

extra_file='data/simi_data/severe_simi_neg_extra_corpus.csv'
#extra_file=None
train_df=Inner_Exc_sample_focus_simi_neg_df(xx,extra_file)
logger.info(f'train_df shape: {train_df.shape}')
logger.info(f'train_df:\n{train_df.head()}')


def Inner_Exc_build_sample_df(data_df,extra_file):
    """
    1. 正样本自抽样，扩充正样本数量,构造正样本对
    2. 额外扩展样本与其他标准问,构造负样本对
    3. 当前标准问与其他标准问，包含与其他所有重复笛卡尔和去重标准问笛卡尔成对，构成负样本对
    :param data_df: 只有正样本的原始df,必须包含Category,Stand_text,Simi_text,共3个字段
    :param extra_file: 额外扩展文件，csv格式
    return DataFrame: 抽样组成最终df,必须包含Category,Stand_text,Simi_text,label共4个字段
    """
    ## 1. 正样本自抽样，扩充正样本数量
    data = data_df.copy()
    df_cols=['Stand_text','Simi_text','label','type']
    # 自抽样扩充正样本数量
    data_pos_sample_df1 = data.sample(frac=0.5)
    data_pos_sample_df2 = data.sample(frac=1.5, replace=True)
    data_pos_sample_df3 = data.loc[:, ['Category','Simi_text', 'Stand_text']].sample(frac=0.5)
    data_pos_sample_df3.columns=['Category','Stand_text','Simi_text']
    standard_query = data['Stand_text'].unique().tolist()
    # 2. 额外扩展样本与其他标准问，构造负样本。
    if extra_file:
        data_extra = pd.read_csv(extra_file, header=None)
        data_extra = data_extra[0].unique().tolist()
        ## 标准问与额外样本笛卡尔积获得负样本对
        data_neg_extra_df4 = pd.DataFrame(list(itertools.product(standard_query, data_extra)))
        data_neg_extra_df4 = data_neg_extra_df4.sample(data.shape[0])
        data_neg_extra_df4.columns = ['Stand_text', 'Simi_text']
        data_neg_extra_df4['label']=0
        data_neg_extra_df4['type'] = 'Neg_Extra_Stand'
        data_neg_extra_df4=data_neg_extra_df4.loc[:,df_cols]
    else:
        data_neg_extra_df4=pd.DataFrame()
    #3.1 相同种类中当前标准问与其他标准问，与其他所有重复标准问笛卡尔
    sub_cate_unique_list=data['Category'].unique().tolist()
    sub_cate_inner_neg_stand_pair_list=[]
    # 3.2 相同种类中标准问之间去重标准问笛卡尔成对
    sub_cate_inner_neg_stand_Descartes_pair_list = []
    for sub_cate in sub_cate_unique_list:
        sub_cate_df=data[data['Category']==sub_cate]
        sub_stand_unique_text = data['Stand_text'].unique().tolist()
        ## 去重标准问笛卡尔成对
        stand_unique_Descartes_pair_list = list(itertools.combinations(sub_stand_unique_text, 2))
        sub_cate_inner_neg_stand_Descartes_pair_list.extend(stand_unique_Descartes_pair_list)
        neg_stand_pair_list = []
        for sub_stand in sub_stand_unique_text:
            sub_other_stand_df = sub_cate_df[sub_cate_df['Stand_text'] != sub_stand]
            other_stand = sub_other_stand_df['Stand_text'].tolist()
            neg_stand_pair_list.extend(list(itertools.product([sub_stand], other_stand)))
        sub_cate_inner_neg_stand_pair_list.extend(neg_stand_pair_list)
    data_neg_stand_df5 = pd.DataFrame(sub_cate_inner_neg_stand_pair_list)
    data_neg_stand_df5 = data_neg_stand_df5.sample(data.shape[0])
    ## 重标准问笛卡尔成对DF
    data_neg_stand_df6_pre = pd.DataFrame(sub_cate_inner_neg_stand_Descartes_pair_list)
    if data_neg_stand_df6_pre.shape[0] > int(data.shape[0] / 2):
        data_neg_stand_df6 = data_neg_stand_df6_pre.sample(int(len(data) / 2))
    else:
        data_neg_stand_df6 = data_neg_stand_df6_pre.sample(int(len(data) / 2), replace=True)
    #sample_pos_df = pd.concat([data_pos_sample_df1, data_pos_sample_df2, data_pos_sample_df3],axis=0,sort=False)
    sample_pos_df = pd.concat([data_pos_sample_df1, data_pos_sample_df2], axis=0, sort=False)
    sample_pos_df['label'] = 1
    sample_pos_df['type']='Self_sample'
    sample_pos_df=sample_pos_df.loc[:,df_cols]
    sample_neg_df=pd.concat([data_neg_stand_df5, data_neg_stand_df6],axis=0,sort=False)
    sample_neg_df.columns=['Stand_text','Simi_text']
    sample_neg_df['label']=0
    sample_neg_df['type'] = 'Neg_Stand_Interactive'
    sample_neg_df = sample_neg_df.loc[:, df_cols]
    #重点基于相似问构造负样本和一定量的正样本
    train_focus_neg_df = Inner_Exc_sample_focus_simi_neg_df(data_df,extra_file)
    train_focus_neg_df=train_focus_neg_df.sample(len(data))
    logger.info(f'sample_pos_df shape:\n{sample_pos_df.shape}')
    logger.info(f'data_neg_extra_df4 shape:\n{data_neg_extra_df4.shape}')
    logger.info(f'sample_neg_df shape:\n{sample_neg_df.shape}')
    logger.info(f'train_focus_neg_df shape:\n{train_focus_neg_df.shape}')
    final_train_df = pd.concat([sample_pos_df,data_neg_extra_df4,sample_neg_df,train_focus_neg_df], axis=0, sort=False)
    final_train_df = final_train_df.sample(frac=1)
    final_train_df = final_train_df.dropna()
    final_train_df.loc[:, 'label'] = final_train_df.loc[:, 'label'].map(int)
    final_train_df = final_train_df.reset_index(drop=True)
    return final_train_df


#### Test
xx=pd.DataFrame({'Category':['A']*5+['H']*5,'Stand_text':['a1']*3+['a2']*2+['h1']*3+['h2']*2,
                 'Simi_text':list('bcdef')+list('ijkmn')})
logger.info(f'xx:\n{xx}')


extra_file='data/simi_data/severe_simi_neg_extra_corpus.csv'
#extra_file=None
train_df=Inner_Exc_build_sample_df(xx,extra_file)
logger.info(f'train_df shape:{train_df.shape}')
logger.info(f'train_df:\n{train_df.head()}')
logger.info(f'type distribution:\n{train_df.type.value_counts()}')
logger.info(f'label distribution:\n{train_df.label.value_counts()}')







