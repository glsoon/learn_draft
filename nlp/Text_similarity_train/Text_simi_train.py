#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/13 13:50
# @Author  : Gongle
# @File    : Text_simi_train.py
# @Version : 1.0
# @Desc    : None


#########################################################################################
####  step-0    load packages
#########################################################################################

import torch
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import sentence_transformers.readers.InputExample as InputExample

from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,average_precision_score
from sklearn.metrics import recall_score, roc_curve,f1_score,auc
from scipy.stats import pearsonr, spearmanr
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import argparse
from typing import List
import sys
import os

os.chdir(os.path.abspath('.'))
sys.path.append(os.path.abspath('.'))

from Text_Similarity_Sampler import build_sample_df,Outer_Exc_build_sample_df
from Text_Similarity_Sampler import Inner_Exc_build_sample_df,EDA_build_sampler_df,DataSampler


#### Just some code to logger.info debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger=logging.getLogger(__name__)
logger.info('Starting')

#########################################################################################
####  step-1    define function and class
#########################################################################################

def show_ml_metric(test_labels, predict_labels, predict_prob):
    accuracy =accuracy_score(test_labels, predict_labels)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1_measure = f1_score(test_labels, predict_labels)
    confusionMatrix = confusion_matrix(test_labels, predict_labels)
    fpr, tpr, threshold = roc_curve(test_labels, predict_prob, pos_label=1)
    Auc = auc(fpr, tpr)
    logger.info ("------------------------- ")
    logger.info ("confusion matrix:")
    logger.info ("------------------------- ")
    logger.info ("| TP: %5d | FP: %5d |" % (confusionMatrix[1, 1], confusionMatrix[0, 1]))
    logger.info ("----------------------- ")
    logger.info ("| FN: %5d | TN: %5d |" % (confusionMatrix[1, 0], confusionMatrix[0, 0]))
    logger.info (" ------------------------- ")
    logger.info ("Accuracy:       %.2f%%" % (accuracy * 100))
    logger.info ("Recall:         %.2f%%" % (recall * 100))
    logger.info ("Precision:      %.2f%%" % (precision * 100))
    logger.info ("F1-measure:     %.2f%%" % (f1_measure * 100))
    logger.info ("AUC:            %.2f%%" % (Auc * 100))
    logger.info ("------------------------- ")
    return (Auc)

def show_info(info_str):
    logger.info('**' * 15)
    log_time = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    log_str = '[Info]   {:<25}  time: {:<19}'.format(info_str, log_time)
    logger.info(log_str)
    return log_str


def show_run_info(step, run_all_steps):
    log_time = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    log_str = ('[Info] time :{:<19}  task id :{:>7}    finished {:>4.2%} '.format(log_time, step, step / run_all_steps))
    logger.info(log_str)
    return log_str


class SentBertDataReader(object):
    """
    Reads in the Stanford NLI dataset and the MultiGenre NLI dataset
    """
    def __init__(self,data_df ):
        self.data_df = data_df

    def get_examples(self, max_examples=0):
        data_df=self.data_df
        s1 = data_df.loc[:,'Simi_text'].tolist()
        s2 = data_df.loc[:,'Stand_text'].tolist()
        labels = data_df.loc[:,'label'].tolist()
        examples = []
        id = 0
        for sentence_a, sentence_b, label in zip(s1, s2, labels):
            guid = "%s-%d" % ('faq', id)
            id += 1
            examples.append(InputExample(guid=guid, texts=[sentence_a, sentence_b], label=float(label)))
            if 0 < max_examples <= len(examples):
                break
        return examples

    @staticmethod
    def get_labels():
        return {"true": 0, "fake": 1}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]


###########  sent bert train and evaluation
class sentSimiModel():
    def __init__(self,train_df,dev_df,trained_path=None,pretrain_model='hfl/chinese-bert-wwm-ext',gpu_no=0,output_path='./faq_model/recall'):
        self.train_df=train_df
        self.dev_df=dev_df
        self.trained_path=trained_path
        self.pretrain_model=pretrain_model
        self.output_path=output_path
        self.gpu_no=gpu_no

    def train(self,batch_size=70,epochs=4,output_version='v366'):
        logger.info('current train  epochs: %d  batch_size: %d'%(epochs,batch_size))
        device=torch.device('cuda:{}'.format(self.gpu_no))
        ############# load pretrain model weights
        if not self.trained_path:
            logger.info("Loading pretrained model : {}".format(self.pretrain_model))
            word_embedding_model = models.Transformer(self.pretrain_model)
            # Apply mean pooling to get one fixed sized sentence vector
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model],device=device)
            model.train()
        else:
            logger.info("Loading trained model : {}".format(self.trained_path))
            model=SentenceTransformer(self.trained_path,device=device)
            model.train()
        #### convert dataframe to dataset format
        train_datasets=SentencesDataset(SentBertDataReader(self.train_df).get_examples(), model=model)
        train_dataloader = DataLoader(train_datasets, shuffle=True, batch_size=batch_size)
        train_loss = losses.CosineSimilarityLoss(model=model)
        ####
        logger.info("Read  dev dataset")
        dev_samples = []
        for i,row in self.dev_df.iterrows():
            inp_example = InputExample(texts=[row['Simi_text'], row['Stand_text']], label=row['label'])
            dev_samples.append(inp_example)
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')
        # Configure the training
        logger.info('dataloader steps : %d'%(len(train_dataloader)))
        warmup_steps = math.ceil(len(train_dataloader) * epochs / batch_size * 0.5) #30% of train data for warm-up
        logger.info("Warmup-steps: {}".format(warmup_steps))
        model_save_path = os.path.join(self.output_path,'Medical_simi-'+output_version+'epochs_'+str(epochs)+'-'+datetime.now().strftime("%Y-%m-%d"))
        # Train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=evaluator,
                  epochs=epochs,
                  evaluation_steps=500,
                  warmup_steps=warmup_steps,
                  output_path=model_save_path
                  )
        return model


###########  calculate input text with object_text similarity
def model_pred_result(model, text, obje_embed,top_n=3,dist_mode='cosine'):
    target_nums = len(obje_embed)
    embed_list = model.encode([text],show_progress_bar=False)
    embed_list=np.repeat(embed_list,target_nums,axis=0)
    if dist_mode=='cosine':
        pred_pos_proba = 1 - (paired_cosine_distances(embed_list, obje_embed))
    elif dist_mode == 'manhattan':
        pred_pos_proba = -paired_manhattan_distances(embed_list, obje_embed)
    elif dist_mode == 'euclidean':
        pred_pos_proba = -paired_euclidean_distances(embed_list, obje_embed)
    elif dist_mode == 'dot':
        pred_pos_proba = [np.dot(emb1, emb2) for emb1, emb2 in zip(embed_list, obje_embed)]
        pred_pos_proba = np.array(pred_pos_proba)
    pred_proba_sorted = np.sort(pred_pos_proba)[::-1]
    pred_pos_idx = pred_pos_proba.argsort()[::-1]
    pred_result_idx = pred_pos_idx[:top_n]
    pred_proba = pred_proba_sorted[:top_n]
    return pred_result_idx,pred_proba


def model_pred_eval(model, text, obje_embed,top_n=3,dist_mode='cosine',thre=0.95):
    target_nums = len(obje_embed)
    embed_list = model.encode([text],show_progress_bar=False)
    embed_list=np.repeat(embed_list,target_nums,axis=0)
    if dist_mode=='cosine':
        pred_pos_proba = 1 - (paired_cosine_distances(embed_list, obje_embed))
    elif dist_mode == 'manhattan':
        pred_pos_proba = -paired_manhattan_distances(embed_list, obje_embed)
    elif dist_mode == 'euclidean':
        pred_pos_proba = -paired_euclidean_distances(embed_list, obje_embed)
    elif dist_mode == 'dot':
        pred_pos_proba = [np.dot(emb1, emb2) for emb1, emb2 in zip(embed_list, obje_embed)]
        pred_pos_proba = np.array(pred_pos_proba)
    pred_proba_sorted = np.sort(pred_pos_proba)[::-1]
    pred_pos_idx = pred_pos_proba.argsort()[::-1]
    if thre:
        pred_thre=[(x,y) for x,y in zip(pred_proba_sorted,pred_pos_idx) if x>=thre]
        pred_pos_idx,pred_proba_sorted=zip(*pred_thre)
    pred_result_idx = pred_pos_idx[:top_n]
    pred_proba = pred_proba_sorted[:top_n]
    return pred_result_idx,pred_proba


def model_pair_eval(model, Simi_text,Stand_text,dist_mode='cosine'):
    embed_list = model.encode([Simi_text,Stand_text],show_progress_bar=False)
    if dist_mode=='cosine':
        pred_pos_proba = 1 - (paired_cosine_distances([embed_list[0]], [embed_list[1]]))
    elif dist_mode == 'manhattan':
        pred_pos_proba = -paired_manhattan_distances([embed_list[0]], [embed_list[1]])
    elif dist_mode == 'euclidean':
        pred_pos_proba = -paired_euclidean_distances([embed_list[0]], [embed_list[1]])
    elif dist_mode == 'dot':
        pred_pos_proba = np.dot(embed_list[0], embed_list[1])
    pred_pos_proba=pred_pos_proba[0]
    return pred_pos_proba


def model_pair_eval_proba(model, Simi_text:List,Stand_text:List,dist_mode='cosine',batch_size=300):
    left_embed_list = model.encode(Simi_text,show_progress_bar=False,batch_size=batch_size)
    right_embed_list=model.encode(Stand_text,show_progress_bar=False,batch_size=batch_size)
    if dist_mode=='cosine':
        pred_pos_proba = 1 - (paired_cosine_distances(left_embed_list, right_embed_list))
    elif dist_mode == 'manhattan':
        pred_pos_proba = -paired_manhattan_distances(left_embed_list, right_embed_list)
    elif dist_mode == 'euclidean':
        pred_pos_proba = -paired_euclidean_distances(left_embed_list, right_embed_list)
    elif dist_mode == 'dot':
        pred_pos_proba = np.dot(left_embed_list, right_embed_list)
    return pred_pos_proba

###########  evaluate eval_data with top_3 index only using standard question
def model_pred_evaluate(model, text_df, data_sampler,top_n=3,dist_mode='cosine',verbose=False):
    show_info('objective label encoding')
    stand_ques = data_sampler.label
    logger.info('Standard text nums: {}\nExamples: {}'.format(len(stand_ques),stand_ques[:5]))
    stand_ques = [str(item) for item in stand_ques]
    stand_ques_embed = model.encode(stand_ques,show_progress_bar=False)
    i = 0
    nums = text_df.shape[0]
    eval_result=[]
    for sub_text, true_label  in zip(text_df.loc[:, 'Simi_text'], text_df.loc[:, 'Stand_text']):
        i += 1
        if i % 50 == 0:
            show_run_info(i, nums)
        sub_eval_result=[]
        pred_idx, pred_proba = model_pred_result(model,sub_text,stand_ques_embed,top_n=top_n,dist_mode=dist_mode)
        pred_label = [data_sampler.label[idx] for idx in pred_idx]
        pred_all_bool=1 if true_label in pred_label else 0
        if (not pred_all_bool) and verbose:
            logger.info('***' * 15)
            logger.info('query question: {}'.format(sub_text))
            logger.info('candidate question : ')
            sub_i = 0
            for sub_cand in pred_label:
                sub_i += 1
                logger.info('{}  : {}'.format(sub_i, sub_cand))
            logger.info('true question : \n {}'.format(true_label))
        pred_bool=[1 if true_label in pred_label[:num] else 0 for num in range(1,top_n+1)]
        eval_result.append(pred_bool)
    logger.info('***' * 15)
    logger.info('pred result mode :{} \n'.format(dist_mode))
    eval_result_value=np.array(eval_result).sum(axis=0)
    eval_result_value=eval_result_value/len(eval_result)
    for sub_i,sub_val in enumerate(eval_result_value):
        logger.info('top_{}  model evaluation accuracy : {:.4f}'.format(sub_i+1,sub_val))


def model_pred_eval_threshold(model, text_df,thre_inter=0.01):
    show_info('pair eval threshold')
    i = 0
    eval_result=[]
    for sub_text, true_label  in zip(text_df.loc[:, 'Simi_text'], text_df.loc[:, 'Stand_text']):
        i += 1
        sub_pred_proba = model_pair_eval(model, sub_text,true_label,dist_mode='cosine')
        eval_result.append(sub_pred_proba)
    start = 0.82
    threshold = [round(item, 4) for item in np.linspace(start, 1, int((1 - start) / thre_inter)) if item != 1.0]
    threshold = sorted(list(set(threshold)), reverse=True)
    for i,sub_thre_val in enumerate(threshold):
        sub_thre_bool=[1 if item>sub_thre_val else 0 for item in eval_result]
        sub_match_bool=[1 if x==y else 0 for x,y in zip(sub_thre_bool,text_df['label'].tolist())]
        sub_acc=sum(sub_match_bool)/len(sub_match_bool)
        logger.info('[{}] Threshold :{:.4f} ==>  model evaluation accuracy : {:.4f}'.format(i,sub_thre_val,sub_acc))


def model_pred_eval_output(model, text_df,thre=0.87,output_file='eval_result.xlsx'):
    show_info('eval output')
    eval_result=[]
    eval_proba=[]
    for sub_text_left,sub_text_right,sub_label in zip(text_df['Simi_text'],text_df['Stand_text'],text_df['label']):
        sub_res=model_pair_eval(model,sub_text_left,sub_text_right)
        eval_proba.append(sub_res)
        sub_pred_label = 1 if sub_res>thre else 0
        sub_acc= 1 if sub_pred_label==sub_label else 0
        eval_result.append(sub_acc)
    acc=sum(eval_result)/len(eval_result)
    logger.info('Threshold :{:.2f} ==>  model evaluation accuracy : {:.4f}'.format(thre, acc))
    text_df.loc[:,'simi_proba']=eval_proba
    text_df.loc[:, 'match_label'] = eval_result
    if output_file:
        text_df.to_excel(output_file)


def simi_eval_threshold(true_label,pred_proba,thre_inter=0.01,verbose=True):
    start = 0.7
    threshold = [round(item, 4) for item in np.linspace(start, 1, int((1 - start) / thre_inter)) if item != 1.0]
    threshold = sorted(list(set(threshold)), reverse=True)
    for i,sub_thre_val in enumerate(threshold):
        sub_thre_bool=[1 if item>sub_thre_val else 0 for item in pred_proba]
        sub_match_bool=[1 if x==y else 0 for x,y in zip(sub_thre_bool,true_label)]
        sub_precision_bool=[1 if x==y and x== 1 else 0 for x,y in zip(sub_thre_bool,true_label) ]
        if sum(true_label)!=0:
            sub_recall=sum(sub_precision_bool)/sum(true_label)
        else:
            sub_recall=0
        if sum(sub_thre_bool)!=0:
            sub_precision=sum(sub_precision_bool)/sum(sub_thre_bool)
        else:
            sub_precision=0
        sub_acc=sum(sub_match_bool)/len(sub_match_bool)
        sub_f1_score=f1_score(true_label,sub_thre_bool)
        if verbose:
            logger.info('[{:>2}] Threshold :{:.4f} ==> Acc: {:.4f}; Precision: {:.4f}; recall: {:.4f} ; f1: {:.4f} '.format(i,
                                           sub_thre_val,sub_acc,sub_precision,sub_recall,sub_f1_score))
    return None


def argument_parse():
    parser=argparse.ArgumentParser('setting parameters for sent bert training similarity')
    parser.add_argument('-d', '--data-file', default='data/Med_general_norm_category_231121.xlsx', type=str,help='eval data file ')
    parser.add_argument('-ed','--eval-file',default='data/General_eval_df_0929.xlsx',type=str,help='eval data file ')
    parser.add_argument('-ef', '--extra-file', default='data/simi_data/severe_simi_neg_extra_corpus.csv', type=str,help='extra data file ')
    parser.add_argument('-b','--batch-size', default=120,type=int, help=" batch size")
    parser.add_argument('-e', '--epoch', default=3, type=int, help=" num epochs")
    parser.add_argument('-g', '--gpu-no', default=0, type=int, help="device GPU no ")
    parser.add_argument('-n', '--top-n', default=15, type=int, help=" return top-n results evaluation")
    parser.add_argument('-m', '--mode', default='test', type=str, help=" train mode")
    parser.add_argument('-v', '--data-version', default='all', type=str, help=" data version")
    parser.add_argument('-p', '--pretrain', default="/home/stops/Work_space/NLP_models/Erlangshen-Roberta-330M-Similarity", type=str, help="pretrain model name")
    parser.add_argument('-t', '--trained_path', default="/home/stops/Work_space/NLP_models/Erlangshen-Roberta-330M-Similarity", type=str, help="trained model path")
    parser.add_argument('-o', '--output', default='./simi_model/', type=str, help="save train recall model path")
    parser.add_argument('-l', '--balanced', default=1, type=int, help=" keep balance corpus")
    opt=parser.parse_args()
    return(opt)


if __name__=="__main__":
    model_args = argument_parse()
    data_path = 'data/simi_data/Med_general_norm_category_231121.xlsx'
    logger.info('train file : {}'.format(data_path))
    model_args.eval_file=None
    ## 采用外互斥策略
    data_sampler = DataSampler(data_path=data_path,eval_file=model_args.eval_file,extra_file=model_args.extra_file,
                               proc_mode="outer",eda_mode=True)
    data_sampler.load_data()
    mode=model_args.mode
    balanced = model_args.balanced
    if balanced==1:
        balanced_bool=True
    else:
        balanced_bool=False
    logger.info('balanced_bool: {}'.format(balanced_bool))
    train_data, val_data, test_data = data_sampler.model_corpus_data(mode=mode,balanced_bool=balanced_bool)
    if isinstance(val_data,pd.DataFrame) and isinstance(test_data,pd.DataFrame):
        logger.info(' train shape: {}   val shape: {}  test shape: {}'.format(train_data.shape,val_data.shape,test_data.shape))
        logger.info('train data head:\n {}\n'.format(train_data.head()))
        logger.info('train dist:\n {}   \nval dist:\n {}   \ntest dist:\n {}\n'.format(train_data.label.value_counts(),val_data.label.value_counts(),test_data.label.value_counts()))
    else:
        logger.info(' train shape: {}   val shape: {}  test shape: {}'.format(train_data.shape, len(val_data),
                                                                        len(test_data)))
        val_data=data_sampler.eval_df.copy()
        test_data=data_sampler.eval_df.copy()
        logger.info('only train mode: val_data and test data is same to eval_df')
        logger.info(' train shape: {}   val shape: {}  test shape: {}'.format(train_data.shape, val_data.shape,
                                                                        test_data.shape))
    ####  train
    pretrain_name=model_args.pretrain
    output_path=model_args.output
    batch_size=model_args.batch_size
    epochs=model_args.epoch
    output_version = "_"+model_args.data_version+"_"
    sentBert=sentSimiModel(train_data,val_data,trained_path=model_args.trained_path,gpu_no=model_args.gpu_no,output_path=model_args.output)
    sent_model=sentBert.train(batch_size,epochs,output_version)
    #####  TEST with matching standard question
    logger.info("*****************************************")
    logger.info("****   TEST")
    logger.info("*****************************************")
    pred_thre=0.87
    sent_model.eval()
    top_n=model_args.top_n
    logger.info('****' * 15)
    show_info(' Evaluate  test  Data')
    eval_top_n=model_args.top_n
    test_df=test_data
    logger.info('test df shape: {}'.format(test_data.shape))
    logger.info('test df head: {}'.format(test_data.head()))
    test_nums = test_data.shape[0]
    logger.info('test data shape :{}'.format(test_nums))
    start_time = datetime.now()
    dist_mode_list=['cosine']
    for sub_mode in dist_mode_list:
        model_pred_evaluate(sent_model,test_df,data_sampler,top_n=top_n,dist_mode=sub_mode,verbose=False)
    end_time = datetime.now()
    cost_time = end_time - start_time
    logger.info('cost time : {}'.format(cost_time))
    logger.info('cost time milliseconds per question: {}'.format((cost_time.microseconds/1000) / test_nums))
    test_pred_poba=model_pair_eval_proba(sent_model,test_df['Simi_text'].tolist(),test_df['Stand_text'].tolist())
    eval_thre=0.5
    eval_pred_label=[1 if itm >eval_thre else 0 for itm in test_pred_poba]
    show_ml_metric(test_df['label'],eval_pred_label,test_pred_poba)
    logger.info('Output test simi result: ')
    model_pred_eval_threshold(sent_model, test_df, thre_inter=0.01)
    test_simi_file='test_result'+'_'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.xlsx'
    test_simi_file=os.path.join('output_data',test_simi_file)
    model_pred_eval_output(sent_model,test_df,pred_thre,test_simi_file)
    logger.info("*****************************************")
    logger.info("****   EVAL")
    logger.info("*****************************************")
    logger.info('****' * 15)
    show_info(' Evaluate  eval  Data')
    eval_top_n=model_args.top_n
    eval_df=data_sampler.eval_df
    logger.info('eval df shape: {}'.format(eval_df.shape))
    logger.info('eval df head: {}'.format(eval_df.head()))
    eval_nums = eval_df.shape[0]
    logger.info('test data shape :{}'.format(eval_nums))
    start_time = datetime.now()
    dist_mode_list=['cosine']
    for sub_mode in dist_mode_list:
        model_pred_evaluate(sent_model,eval_df,data_sampler,top_n=top_n,dist_mode=sub_mode,verbose=False)
    end_time = datetime.now()
    cost_time = end_time - start_time
    logger.info('cost time : {}'.format(cost_time))
    logger.info('cost time milliseconds per question: {}'.format((cost_time.microseconds/1000) / eval_nums))
    logger.info('eval metric: ')
    sub_simi_label = eval_df['label'].tolist()
    eval_pred_poba=model_pair_eval_proba(sent_model,eval_df['Simi_text'].tolist(),eval_df['Stand_text'].tolist())
    eval_thre=0.5
    eval_pred_label=[1 if itm >eval_thre else 0 for itm in eval_pred_poba]
    show_ml_metric(eval_df['label'],eval_pred_label,eval_pred_poba)
    logger.info('Output eval simi result: ')
    simi_eval_threshold(sub_simi_label,eval_pred_poba,0.01,True)
    simi_file='eval_result'+'_'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.xlsx'
    simi_file=os.path.join('output_data',simi_file)
    model_pred_eval_output(sent_model,eval_df,pred_thre,simi_file)
    '''
    ## -----------------------------------------------------------
    ## python command  
    
    python Text_simi_train.py  -l 1 -m only_train -e 3   -t hfl/chinese-bert-wwm-ext
    
    python Text_simi_train.py  -l 1 -m train      -e 3  -t "/home/stops/Work_space/NLP_models/Erlangshen-Roberta-330M-Similarity"
    
    python Text_simi_train.py  -l 1 -m train   -v General   -e 3 -b 600  -t "/home/stops/Work_space/NLP_models/bge-base-zh-v1.5" > log_simi_1122.txt 
    
    '''



