#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/31 10:06
# @Author  : Gongle
# @File    : llama_dataset.py
# @Version : 1.0
# @Desc    : None

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
from torch.utils.data import Dataset


## for Ziya-llama-13b-model
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "im_start"
DEFAULT_BOS_TOKEN = "im_end"
DEFAULT_UNK_TOKEN = "[PAD]"


""" 
{
    "chat_format": "chatml",
    "eos_token_id": 151643,
    "pad_token_id": 151643,
    "max_window_size": 6144,
    "max_new_tokens": 512,
    "do_sample": true,
    "top_k": 0,
    "top_p": 0.5,
    "transformers_version": "4.31.0"
}

pad_token_id :151643 ['<|endoftext|>']

messages=[{'role':'user','content':'你是医生，进行问诊'},
          {'role':'assistant','content':'你好，请问有什么不适'},
          {'role':'user','content':'肚子痛'},
          {'role':'assistant','content':'是间断的，还是连续疼的？'},
          {'role':'user','content':'间断的'},]
          
system_tokens   : im_start_tokens + system_token      + nl_tokens + system_text_token      +  im_end_tokens
query_tokens    : im_start_tokens + user_token        + nl_tokens + user_text_token        +  im_end_tokens
response_tokens : im_start_tokens + assistant_token   + nl_tokens + assistant_text_token   +  im_end_tokens

##loop:
next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens

context_tokens =system_tokens+next_context_tokens
"""

import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class LlamaDataSet(Dataset):
    def __init__(self, data_path, tokenizer,max_len ):
        self.all_data = []
        self.seq_lens = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                input_ids = sample["input_ids"]
                labels = sample["labels"]
                self.seq_lens.append(len(input_ids))
                # Padding-side: Right

                if max_len:
                    pad_len = max_len - len(input_ids)
                    input_ids =  input_ids+[151643] * pad_len
                    labels =  labels+[-100] * pad_len
                #attention_mask = input_ids.ne(tokenizer.pad_token_id)
                attention_mask = [1 if item!=151643 else 0 for item in input_ids]
                self.all_data.append({"input_ids": input_ids, "labels": labels,"attention_mask": attention_mask})

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance


def coll_fn(batch):
    #<pad>=151643
    input_ids_list, labels_list,attention_mask_list = [], [],[]
    for instance in batch:
        input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
        labels_list.append(torch.tensor(instance["labels"], dtype=torch.long))
        attention_mask_list.append(torch.tensor(instance["attention_mask"], dtype=torch.long))
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=151643),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=-100),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0)}


