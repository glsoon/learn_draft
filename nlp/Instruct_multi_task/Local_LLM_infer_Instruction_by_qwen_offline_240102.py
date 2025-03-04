#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/6 19:13
# @Author  : Gongle
# @File    : Local_LLM_infer_Instruction_by_qwen_offline_240102.py
# @Version : 1.0
# @Desc    : 使用本地Qwen-14b模型对开源指令进行分类

import os,time
os.environ['CUDA_VISIBLE_DEVICES']='0'

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


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger.info('Starting')

"""
alpaca_file="data/Alpaca_gpt4_zh_chosen_data_5w_1020.xlsx"
alpaca_df=pd.read_excel(alpaca_file)
"""


#data_file="data/Alpaca_gpt4_zh_chosen_data_5w_1020.xlsx"
data_file="data/Alpaca_COT_zh_data_7w_240103.xlsx"
data_df=pd.read_excel(data_file).sample(300)

print(data_df.shape)
print(data_df.head()) ## columns:  instruction ,output
print(data_df.isnull().sum())

###################
## build prompt
###################
instruct_prompt_text="""对于下面指令文本的意图目的，参考下面已有的类别，给出所属的类别。
参考类别：[计算，逻辑与推理，代码，知识与百科，语言理解与抽取，上下文对话，生成与创作，角色扮演，安全指令攻击，任务规划，其他]，同时不要出现没有在上面的类别。

指令文本内容：
"{text}"

任务要求如下：
1.首先给出所属类别的分析过程。
2.再返回结果为JSON格式，{{"类别":[xxx]}}。
"""

check_user_first_text_result_list=data_df["instruction"].tolist()

user_first_text_department_result_list=[]
for sub_text in check_user_first_text_result_list:
    sub_u_f_text=instruct_prompt_text.format(text=sub_text)
    user_first_text_department_result_list.append(sub_u_f_text)

logger.info(f"user_first_text_department_result_list nums: {len(user_first_text_department_result_list)}")
logger.info(f"Examples : {user_first_text_department_result_list[:2]}")


###################
## load model
###################
#model_path="/home/stops/Work_space/NLP_work/Baichuan2_LM_train/output_bc2_7b_med_nlp_dir_0914/global_step_e3_1324"
#model_path="/home/stops/Work_space/NLP_models/baichuan_model/global_step_e2_8922"
model_path="/home/stops/Work_space/NLP_models/Qwen-14B-Chat/"


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
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False,trust_remote_code=True)
    logger.info(f"load model path: {model_path}")
    return model, tokenizer



model, tokenizer = init_model()


def Qwen_FAQ_batch_infer(batch_text_list:List[str]):
    if isinstance(batch_text_list,str):
        batch_text_list=[batch_text_list]
    im_start, im_end = "<|im_start|>", "<|im_end|>"
    im_start_tokens = [tokenizer.im_start_id]
    im_end_tokens = [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")
    batch_input_ids=[]
    def _tokenize_str(role, content):
        return tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())
    system="You are a helpful assistant."
    system_tokens_part = _tokenize_str("system", system)
    system_tokens = im_start_tokens + system_tokens_part + im_end_tokens
    for sub_text in batch_text_list:
        sub_input_ids=system_tokens+nl_tokens+ im_start_tokens+ _tokenize_str("user", sub_text)+ im_end_tokens \
            + nl_tokens+ im_start_tokens+ tokenizer.encode("assistant")+ nl_tokens
        batch_input_ids.append(torch.tensor(sub_input_ids, dtype=torch.long))
    batch_input_len=[len(item) for item in batch_input_ids]
    batch_max_len=max(batch_input_len)
    ## padding-strategy: LEFT, QWEN eos_token_id=pad_token_id=151643
    batch_input_tensor_ids=left_padding(batch_input_ids, padding_value=151643)
    test_batch_input_ids=batch_input_tensor_ids.to("cuda")
    batch_outputs = model.generate(test_batch_input_ids)
    batch_response_text_result_list=[]
    for sub_output  in batch_outputs:
        sub_response = tokenizer.decode(sub_output[batch_max_len:], skip_special_tokens=True)
        batch_response_text_result_list.append(sub_response)
    return batch_response_text_result_list




test_text_list=user_first_text_department_result_list[:1]
logger.info(f"test_text_list: {test_text_list}")

s_time=time.time()
test_infer_texts=Qwen_FAQ_batch_infer(test_text_list)
logger.info("batch cost time : {:.2f}".format(time.time()-s_time))
logger.info(f"test_infer_texts: {test_infer_texts}")

extract_dict_text="""[{]\n?.*?\n?[}]"""
test_result=re.findall(extract_dict_text,test_infer_texts[0],re.DOTALL)
logger.info(f"test_result dict : {test_result}")


####################
## 科室结果
####################

## instruction,output
## 历史数据
text_result_list=[]
input_result_list=[]
output_result_list=[]


##预测数据
llm_infer_result=[]
llm_infer_post_proc_dict_result=[]
llm_infer_post_proc_text_result=[]

log_steps=50
save_steps=50
batch_size=10
all_nums=len(user_first_text_department_result_list)
logger.info(f"run nums: {len(list(range(0,all_nums,batch_size)))} , batch_size: {batch_size}")

for idx in range(0,all_nums,batch_size):
    ## 科室预测结果
    sub_batch_texts=user_first_text_department_result_list[idx:(idx+batch_size)]
    sub_infer_texts=Qwen_FAQ_batch_infer(sub_batch_texts)
    llm_infer_result.extend(sub_infer_texts)

    ## 科室预测后处理结果
    sub_department_proc_dict_result=[]
    sub_department_proc_text_result = []
    extract_department_text = """[{].*?[}]"""
    for sub_text in sub_infer_texts:
        sub_result = re.findall(extract_dict_text,sub_text,re.DOTALL)
        if sub_result:
            sub_result_text = sub_result[0]
        else:
            sub_result_text = None
        try:
            sub_dict = eval(sub_result_text)
            # print("sub_dict: ",sub_dict)
            sub_raw_res = sub_dict["类别"]
            if isinstance(sub_raw_res, List):
                sub_result_string = sub_raw_res[0]
            else:
                sub_result_string = sub_raw_res
        except:
            sub_result_string = None
        sub_department_proc_dict_result.append(sub_result_text)
        sub_department_proc_text_result.append(sub_result_string)
    llm_infer_post_proc_dict_result.extend(sub_department_proc_dict_result)
    llm_infer_post_proc_text_result.extend(sub_department_proc_text_result)

    ## 历史数据
    sub_text_result_list=data_df["instruction"].tolist()[idx:(idx+batch_size)]
    sub_input_result_list = sub_batch_texts
    sub_output_result_list = data_df["output"].tolist()[idx:(idx + batch_size)]


    ## save
    text_result_list.extend(sub_text_result_list)
    input_result_list.extend(sub_input_result_list)
    output_result_list.extend(sub_output_result_list)


    if idx%log_steps==0:
        logger.info("run step: {}, all nums: {} , finished: {:.2%}".format(idx,all_nums,idx/all_nums))

    if idx%save_steps==0:
        logger.info(f"text_result_list nums: {len(text_result_list)}")
        logger.info(f"input_result_list nums: {len(input_result_list)}")
        logger.info(f"output_result_list nums: {len(output_result_list)}")
        logger.info(f"llm_infer_post_proc_text_result nums: {len(llm_infer_post_proc_text_result)}")
        sub_gpt_res_df = pd.DataFrame({'text':text_result_list,
                                       'input': input_result_list,
                                       'output': output_result_list,
                                       'detail_result': llm_infer_result,
                                       'dict_label': llm_infer_post_proc_dict_result,
                                       'label':llm_infer_post_proc_text_result})
        logger.info('sub_gpt_res_df.shape: {}'.format(sub_gpt_res_df.shape))
        sub_save_date = datetime.now().strftime('%Y-%m-%d-%H')
        sub_save_file = 'output_data/department_sub_df_' + sub_save_date + '.xlsx'
        sub_gpt_res_df.to_excel(sub_save_file)
        logger.info('save file: {}'.format(sub_save_file))


save_df=data_df.head(len(user_first_text_department_result_list)).copy()


save_df.loc[:,"input"]=user_first_text_department_result_list
save_df.loc[:,"detail_result"]=llm_infer_result
save_df.loc[:,"dict_label"]=llm_infer_post_proc_dict_result
save_df.loc[:,"label"]=llm_infer_post_proc_text_result

logger.info(f"shape :{save_df.shape}")
logger.info(f"head :\n{save_df.head()}")


save_mode=True
save_file="output_data/Alpaca_gpt4_cot_zh_label_data_7w_240102.xlsx"
if save_mode:
    save_df.to_excel(save_file,index=False)
    logger.info(f"data nums: {save_df.shape}, save file: {save_file}")


"""
nohup python Local_LLM_infer_Instruction_by_qwen_offline_240102.py > log/alpaca_instruct_clf_log_240103.txt  2>&1 &
tail -F log/alpaca_instruct_clf_log_240103.txt
"""





