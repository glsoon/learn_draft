#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/1/30 15:38
# @Author  : Gongle
# @File    : text_cluster_recursive_offline.py
# @Version : 1.0
# @Desc    : None

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import torch
import math
import logging
from datetime import datetime,timedelta,date
import time
import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import hdbscan
from sklearn.metrics.pairwise import  cosine_distances,cosine_similarity
from pprint import pprint
from typing import List,Dict,Union
from types import MethodType
import ahocorasick
import itertools
import copy
import re
import json
import requests

os.chdir('/home/stops/Work_space/NLP_work/Med_assit_chatglm')


from db_config_taiyi import DB ## load data from pgsql

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger.info('Starting')


#### load model
simi_model_path='/home/stops/Work_space/NLP_models/bge-base-zh-v1.5'
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu' )
simi_model=SentenceTransformer(simi_model_path,device=device)
simi_model.eval()
logger.info('Loading semantic similarity model')

#### load data

consult_file='data/Toxic_instruction_df_2k_240104.xlsx'
logger.info('load file : {}'.format(consult_file))
consult_df=pd.read_excel(consult_file)
logger.info('data shape :{}'.format(consult_df.shape))


consult_df.loc[:,'text_len']=consult_df.loc[:,'instruction'].map(lambda x: len(str(x)))
logger.info('msg length distribution: ')
logger.info('{}'.format(consult_df['text_len'].describe(percentiles=[0.01,0.2,0.25,0.5,0.75,0.8,0.9,0.99])))

consult_df=consult_df.loc[(consult_df['text_len']>=3)&(consult_df['text_len']<=300),:]

query_set_text_list=consult_df.loc[:,'instruction'].unique().tolist()
logger.info('query_set_text_list nums: {}'.format(len(query_set_text_list)))


def output_cluster_result_df(text_list, min_samples=3):
    s_time=time.time()
    corpus_matrix  = []
    batch_size=300
    for i in range(0,len(text_list), batch_size):
        corpus_matrix.append(simi_model.encode(text_list[i:i+batch_size],show_progress_bar=False))
    corpus_matrix =np.concatenate(corpus_matrix)
    logger.info('build embed cost time: {:.2f}s, shape: {}'.format(time.time()-s_time,corpus_matrix.shape))
    logger.info('starting cluster: ',)
    s_time=time.time()
    cluster = hdbscan.HDBSCAN(min_cluster_size = min_samples)
    cluster.fit(corpus_matrix)
    labels_cluster= cluster.labels_
    cluster_result_df = pd.DataFrame({'content':text_list, 'cluster':labels_cluster})
    cluster_label_counts_df=cluster_result_df.cluster.value_counts().reset_index()
    cluster_label_counts_df.columns=['cluster','count_nums']
    label_num=cluster_label_counts_df.shape[0]
    label_max_num=cluster_label_counts_df['count_nums'].max()
    label_min_num=cluster_label_counts_df['count_nums'].min()
    logger.info( 'label nums: {},label_cnt max: {}, min: {}'.format(label_num,label_max_num,label_min_num))
    cluster_data_df=pd.merge(cluster_result_df,cluster_label_counts_df,on=['cluster'],how='left')
    cluster_data_df=cluster_data_df.sort_values('cluster',ascending=False)
    cluster_idx=cluster_data_df.loc[:,'cluster'].drop_duplicates().tolist()
    cluster_idx_map=dict(zip(cluster_idx,range(1,len(cluster_idx)+1)))
    cluster_data_df.loc[:,'cluster']=cluster_data_df.loc[:,'cluster'].map(cluster_idx_map)
    cluster_data_df=cluster_data_df.reset_index(drop=True)
    logger.info('cluster cost time: {:.2f}s'.format(time.time()-s_time))
    logger.info('cluster data :\n{}'.format(cluster_data_df.head()))
    return cluster_data_df,label_max_num


label_max_num=len(query_set_text_list)
min_samples=3
random_min_samples=False
super_cluster_df=pd.DataFrame()
cluster_samples_threshold=20
loop_num=0
c_time=time.time()
judge_dead_loop_data=None
logger.info('loop params:{}, min_samples {},cluster_samples_threshold: {},random_min_samples :{}'.format(loop_num,min_samples,cluster_samples_threshold,random_min_samples))
while label_max_num>cluster_samples_threshold:
    loop_num+=1
    if random_min_samples:
        min_samples =np.random.randint(3,6,1).tolist()[0]
    sub_cluster_df,label_max_num=output_cluster_result_df(query_set_text_list,min_samples)
    if super_cluster_df.shape[0]>0:
        former_cluster_no=super_cluster_df['cluster'].max()
        logger.info('loop num: {}, min_samples: {},former_cluster_no: {}'.format(loop_num,min_samples,former_cluster_no))
    else:
        former_cluster_no=0
    if label_max_num>cluster_samples_threshold:
        sub_keep_df=sub_cluster_df.loc[sub_cluster_df['count_nums']<cluster_samples_threshold,:].copy()
        sub_keep_df.loc[:,'cluster']=sub_keep_df.loc[:,'cluster']+former_cluster_no
        super_cluster_df=super_cluster_df.append(sub_keep_df,sort=False)
        query_set_text_list=sub_cluster_df.loc[sub_cluster_df['count_nums']>=cluster_samples_threshold,'content'].tolist()
    else:
        sub_cluster_df.loc[:,'cluster']=sub_cluster_df.loc[:,'cluster']+former_cluster_no
        super_cluster_df=super_cluster_df.append(sub_cluster_df,sort=False)
        break
    save_file = os.path.join('output_data','Text_Cluster_data_df_v2_loop_' + str(loop_num) + '_' + datetime.now().strftime('%Y-%m-%d') + '.xlsx')
    super_cluster_df.to_excel(save_file)
    logger.info(f'loop save file : {save_file}')
    if judge_dead_loop_data!=sub_cluster_df.shape:
        judge_dead_loop_data=sub_cluster_df.shape
    else:
        logger.info('overlap dead data ,please tune parameters,')
        break

logger.info('Final loop nums: {},cost time: {:.2f}'.format(loop_num,time.time()-c_time))

super_cluster_df=super_cluster_df.reset_index(drop=True)
save_file=os.path.join('output_data','Text_Cluster_data_df_'+datetime.now().strftime('%Y-%m-%d')+'.xlsx')
super_cluster_df.to_excel(save_file)
logger.info(f'save file : {save_file}')


"""

nohup python text_cluster_recursive_offline.py  >  log/cluster_recursive_log_240104.txt  2>&1 & 
tail -F  log/cluster_recursive_log_240104.txt

"""


