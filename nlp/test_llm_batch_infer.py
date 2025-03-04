#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/22 16:56
# @Author  : Gongle
# @File    : test_llm_batch_infer.py
# @Version : 1.0
# @Desc    : None

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.types import Number
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig


model_path="/home/stops/Work_space/NLP_work/Baichuan_LM_train/output_plugin_comp_dir_0818/global_step_e3_6184"

def init_model():
    #model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True).to(torch.bfloat16).cuda()
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False,trust_remote_code=True)
    logger.info(f"load model path: {model_path}")
    return model, tokenizer

def right_padding(sequences: [torch.Tensor], padding_value: Number) -> torch.Tensor:
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)


def left_padding(sequences: [torch.Tensor], padding_value: Number) -> torch.Tensor:
    return right_padding(
        [seq.flip(0) for seq in sequences],
        padding_value=padding_value,
    ).flip(1)

def build_batch_faq_input_ids(batch_text_list:List[str]):
    user_token_id=195
    assistant_token_id=196
    batch_input_ids=[]
    for sub_text in batch_text_list:
        sub_input_ids = [user_token_id]+tokenizer.encode(text=sub_text)+[assistant_token_id]
        batch_input_ids.append(torch.tensor(sub_input_ids, dtype=torch.long))
    batch_input_len=[len(item) for item in batch_input_ids]
    batch_max_len=max(batch_input_len)
    #batch_input_tensor_ids=pad_sequence(batch_input_ids, batch_first=True, padding_value=0)
    batch_input_tensor_ids=left_padding(batch_input_ids, padding_value=0)
    return batch_input_tensor_ids,batch_max_len

#%%


test_text_list=["小明有5根铅笔，他想分给他的3位朋友，每人分几根铅笔？",
                "血压70、59，是高血压还是低血压呢？"]

test_input_ids=tokenizer.encode(test_text_list[0])
print(test_input_ids)


#%%

test_batch_input_ids,test_batch_max_len=build_batch_faq_input_ids(test_text_list)
print(test_batch_input_ids)

#%%
s_time=time.time()
generation_config=GenerationConfig.from_pretrained(model_path)
test_batch_input_ids=test_batch_input_ids.to("cuda")
batch_outputs = model.generate(test_batch_input_ids, generation_config=generation_config)
print("generation_config: ",generation_config)
print("batch cost time : {:.2f}".format(time.time()-s_time))

#%%
for sub_output  in batch_outputs:
    sub_response = tokenizer.decode(sub_output[test_batch_max_len:], skip_special_tokens=True)
    print("sub_response: ",sub_response)


#%%

s_time=time.time()
generation_config=GenerationConfig.from_pretrained(model_path)
test_batch_input_ids=test_batch_input_ids.to("cuda")
batch_outputs = model.generate(test_batch_input_ids[:1], generation_config=generation_config)
print("generation_config: ",generation_config)
print("single cost time : {:.2f}".format(time.time()-s_time))

#%%

for sub_output  in batch_outputs:
    sub_response = tokenizer.decode(sub_output[test_batch_max_len:], skip_special_tokens=True)
    print("sub_response: ",sub_response)

