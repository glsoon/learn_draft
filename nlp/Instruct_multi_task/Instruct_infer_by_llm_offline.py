#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/18 19:14
# @Author  : Gongle
# @File    : Instruct_infer_by_llm_offline.py
# @Version : 1.0
# @Desc    : None



import os,time
os.environ['CUDA_VISIBLE_DEVICES']='2'
import copy
import logging
import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, \
    default_data_collator, TrainingArguments, Trainer
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer
from transformers import BertConfig, BertTokenizer,BertTokenizerFast
from datasets import Dataset
from typing import List,Union
import tqdm
import re
import json

os.chdir('/home/stops/Work_space/NLP_work/Med_assit_chatglm')


from db_config_taiyi import DB ## load data from pgsql

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger.info('Starting')


#%%

##################################
## Instruct data
##################################

data_file='/home/stops/Work_space/NLP_models/train_2M_CN/train_2M_CN.json'

belle_instruct_data=[]
with open(data_file,'r') as f :
    for sub_data in f.readlines()[:50000]:
        sub_data_dict=json.loads(sub_data)
        belle_instruct_data.append(sub_data_dict)
logger.info(f'belle_instruct_data nums: {len(belle_instruct_data)}')
logger.info(f'examples: {belle_instruct_data[:2]}')

#%%

belle_instruct_df=pd.DataFrame.from_dict(belle_instruct_data)
belle_instruct_df=belle_instruct_df.applymap(lambda x:x.strip())

logger.info(f"{belle_instruct_df.shape}")


#%%

belle_instruct_df.loc[:,'instruct_len']=belle_instruct_df.apply(lambda x:len(x['instruction']),axis=1)
print(belle_instruct_df['instruct_len'].describe(percentiles=[0.01,0.1,0.2,0.25,0.5,0.75,0.8,0.9,0.9999]))


select_instruct_df=belle_instruct_df.loc[belle_instruct_df['instruct_len']<300,['instruction','output']].copy()
logger.info(f"{select_instruct_df.shape}")


#%%

instruct_prompt_text="""对于下面文本，参考下面已有的类别，给出所属的类别，格式为json，{{"类别":[xxx]}}，同时解释说明。
参考类别 [代码，头脑风暴，健康医疗，闲聊，翻译，文本重写，文本提取摘要，文本扩写，信息抽取，文本理解，文本分类，科学，数学，传记，商业，社会，经济，文化，角色扮演，常识类问答，其他]，同时不要出现没有在上面的类别。

文本内容：
"{text}"
"""

check_user_first_text_result_list=select_instruct_df["instruction"].tolist()

user_first_text_department_result_list=[]
for sub_text in check_user_first_text_result_list:
    sub_u_f_text=instruct_prompt_text.format(text=sub_text)
    user_first_text_department_result_list.append(sub_u_f_text)

logger.info(f"user_first_text_department_result_list nums: {len(user_first_text_department_result_list)}")


logger.info(f"{user_first_text_department_result_list[:2]}")



###################
## load model
###################
model_path="/home/stops/Work_space/NLP_models/baichuan_model/global_step_e1_15138"


from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from torch.nn.utils.rnn import pad_sequence
from torch.types import Number

def right_padding(sequences: [torch.Tensor], padding_value) -> torch.Tensor:
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)


def left_padding(sequences: [torch.Tensor], padding_value) -> torch.Tensor:
    return right_padding(
        [seq.flip(0) for seq in sequences],
        padding_value=padding_value,).flip(1)


def init_model():
    #model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True).to(torch.bfloat16).cuda()
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False,trust_remote_code=True)
    logger.info(f"load model path: {model_path}")
    return model, tokenizer


#%%

model, tokenizer = init_model()


#%%

def LLM_batch_infer(batch_text_list:List[str]):
    if isinstance(batch_text_list,str):
        batch_text_list=[batch_text_list]
    user_token_id=195
    assistant_token_id=196
    batch_input_ids=[]
    for sub_text in batch_text_list:
        sub_input_ids = [user_token_id]+tokenizer.encode(text=sub_text)+[assistant_token_id]
        batch_input_ids.append(torch.tensor(sub_input_ids, dtype=torch.long))
    batch_input_len=[len(item) for item in batch_input_ids]
    batch_max_len=max(batch_input_len)
    ## padding-strategy: LEFT
    batch_input_tensor_ids=left_padding(batch_input_ids, padding_value=0)
    test_batch_input_ids=batch_input_tensor_ids.to("cuda")
    batch_outputs = model.generate(test_batch_input_ids)
    batch_response_text_result_list=[]
    for sub_output  in batch_outputs:
        sub_response = tokenizer.decode(sub_output[batch_max_len:], skip_special_tokens=True)
        batch_response_text_result_list.append(sub_response)
    return batch_response_text_result_list



test_text_list=user_first_text_department_result_list[:2]
logger.info("test_text_list: {}".format(test_text_list))

s_time=time.time()
test_infer_texts=LLM_batch_infer(test_text_list)
logger.info("batch cost time : {:.2f}".format(time.time()-s_time))
logger.info("test_infer_texts: {}".format(test_infer_texts))



extract_department_text="""[{].*?[}]"""
test_result=re.findall(extract_department_text,test_infer_texts[0])
logger.info("test_result: {}".format(test_result))


####################
## 科室结果
####################
department_infer_detail_result=[]
department_infer_result=[]
batch_size=2
all_nums=len(user_first_text_department_result_list)
logger.info(f"run nums: {all_nums} , batch_size: {batch_size}")

for idx in range(0,all_nums,batch_size):
    if idx%10==0:
        logger.info("run step: {}, finished: {:.2%}".format(idx,idx/all_nums))
    sub_batch_texts=user_first_text_department_result_list[idx:(idx+batch_size)]
    sub_infer_texts=LLM_batch_infer(sub_batch_texts)
    extract_department_text="""[{].*?[}]"""
    for sub_text in sub_infer_texts:
        sub_result=re.findall(extract_department_text,sub_text)
        department_infer_detail_result.append(sub_text)
        if sub_result:
            sub_result_text=sub_result[0]
        else:
            sub_result_text=None
        department_infer_result.append(sub_result_text)

logger.info(f"department_infer_detail_result nums: {len(department_infer_detail_result)}")
logger.info(f"examples: {department_infer_detail_result[:2]}")


logger.info(f"department_infer_result nums: {len(department_infer_result)}")
logger.info(f"examples: {department_infer_result[:2]}")


final_department_result_list=[]
for sub_text in department_infer_result:
    #print("sub_text: ",sub_text)
    try:
        sub_dict=eval(sub_text)
        #print("sub_dict: ",sub_dict)
        sub_res=sub_dict["类别"][0]
    except:
        sub_res=None
    final_department_result_list.append(sub_res)

logger.info(f"final_department_result_list nums: {len(final_department_result_list)}")
logger.info(f"examples: {final_department_result_list[:2]}")


#%%



save_df=select_instruct_df.head(len(final_department_result_list)).copy()

save_df.loc[:,"detail_result"]=department_infer_detail_result
save_df.loc[:,"dict_result"]=department_infer_result
save_df.loc[:,"result"]=final_department_result_list

logger.info(f"{save_df.shape}")
logger.info(f"{save_df.head()}")



save_mode=True
save_file="output_data/Instruct_prediction_by_llm_5w_0918_df.xlsx"
if save_mode:
    save_df.to_excel(save_file,index=False)
    logger.info(f"data nums: {save_df.shape}, save file: {save_file}")


"""
nohup python Instruct_infer_by_llm_offline.py > log/instruct_log_0918.txt  2>&1 &
tail -F log/instruct_log_0918.txt
"""



