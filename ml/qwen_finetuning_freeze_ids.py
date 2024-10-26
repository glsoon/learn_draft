#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/09/12 18:28
# @Author  : Gongle
# @File    : LLM_clf_proc_dataset.py
# @Version : 1.0
# @Desc    : None

import pandas as pd
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
import torch
import deepspeed
import argparse
from torch.utils.data import RandomSampler, DataLoader
from qwen_dataset_ids import LlamaDataSet, coll_fn  ## right padding


from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from shutil import copy
import math
from typing import Dict, List, Optional
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger.info('Starting')



def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    last = scalars[0]
    smoothed = list()
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5) # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_loss(steps,metrics,save_dictionary):
    plt.figure()
    plt.plot(steps, metrics, alpha=0.4, label="raw")
    plt.plot(steps, smooth(metrics), label="smooth")
    plt.title("training loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(save_dictionary, "training_loss.png"), format="png", dpi=100)
    print("Figure saved:", os.path.join(save_dictionary, "training_loss.png"))

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/Qwen_train_data_d1230_s936_1218.json', type=str, help='')
    parser.add_argument('--model_dir', default="/home/stops/Work_space/NLP_models/Qwen-14B-Chat", type=str, help='')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='')
    parser.add_argument('--train_batch_size', default=2, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='')
    parser.add_argument('--output_dir', default='output_dir_freeze/', type=str, help='')
    parser.add_argument('--log_steps', type=int, default=20, help='')
    parser.add_argument('--max_len', type=int, default=1000, help='')
    #parser.add_argument('--max_src_len', type=int, default=450, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='')
    return parser.parse_args()


def main():
    args = set_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model.enable_input_require_grads()

    logger.info(f"load model : {args.model_dir}")

    conf = {"train_micro_batch_size_per_gpu": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-5,
                    "betas": [
                        0.9,
                        0.95
                    ],
                    "eps": 1e-8,
                    "weight_decay": 5e-4
                }
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            "steps_per_print": args.log_steps
            }

    for name, param in model.named_parameters():
        if not any(nd in name for nd in ["h.39", "h.38", "h.37","h.36"]):
            param.requires_grad = False


    print_trainable_parameters(model)
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)
    logger.info('load data file: {}'.format(args.train_path))
    train_dataset = LlamaDataSet(args.train_path, tokenizer, args.max_len)
    train_seq_lens_list=train_dataset.seq_lens
    logger.info('train_seq_lens distribution: {}'.format(pd.Series(train_seq_lens_list).describe()))
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=conf["train_micro_batch_size_per_gpu"],
                                  sampler=RandomSampler(train_dataset),
                                  collate_fn=coll_fn,
                                  drop_last=True,
                                  num_workers=0)
    test_ex= list(train_dataloader)[0]
    #logger.info('test_ex: {}'.format(test_ex))
    xx_input_ids = test_ex['input_ids'].cpu().numpy()[0]
    logger.info('test input: {}'.format(tokenizer.decode(xx_input_ids)))

    xx_labels = test_ex['labels'].cpu().numpy()[0]
    logger.info('test label: {}'.format(tokenizer.decode([item for item in xx_labels if item != -100])))

    model_engine, optimizer, _, _ = deepspeed.initialize(config=conf,
                                                         model=model,
                                                         model_parameters=model.parameters())
    model_engine.train()
    global_step = 0
    keep_global_step_list=[]
    keep_loss_result_list=[]
    for i_epoch in range(args.num_train_epochs):
        train_iter = iter(train_dataloader)
        for step, batch in enumerate(train_iter):
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            outputs = model_engine.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            if conf["gradient_accumulation_steps"] > 1:
                loss = loss / conf["gradient_accumulation_steps"]
            model_engine.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if (step + 1) % conf["gradient_accumulation_steps"] == 0:
                model_engine.step()
                global_step += 1
            if global_step % args.log_steps == 0:
                logger.info("loss:{}, global_step:{}".format(float(loss.item()), global_step))
                keep_global_step_list.append(global_step)
                keep_loss_result_list.append(float(loss.item()))
        save_dir = os.path.join(args.output_dir, f"global_step_e{i_epoch}_{global_step}")
        model_engine.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        copy(os.path.join(args.model_dir, "configuration_qwen.py"), os.path.join(save_dir, "configuration_qwen.py"))
        copy(os.path.join(args.model_dir, "modeling_qwen.py"), os.path.join(save_dir, "modeling_qwen.py"))
        copy(os.path.join(args.model_dir, "generation_config.json"), os.path.join(save_dir, "generation_config.json"))
        copy(os.path.join(args.model_dir, "qwen_generation_utils.py"), os.path.join(save_dir, "qwen_generation_utils.py"))
        copy(os.path.join(args.model_dir, "tokenization_qwen.py"), os.path.join(save_dir, "tokenization_qwen.py"))
        logger.info(f"save model: {save_dir}")
        if keep_loss_result_list:
            logger.info(f"keep_loss_result_list: {keep_loss_result_list[-5:]}")
            plot_loss(keep_global_step_list,keep_loss_result_list,args.output_dir)
            logger.info("save loss figure")


if __name__ == "__main__":
    main()


    ## plugin max tokens: 1027


    # CUDA_VISIBLE_DEVICES=0  deepspeed --master_port 6666  qwen_finetuning_freeze_ids.py   --train_path data/Qwen_train_data_d1230_s936_1218.json  --output_dir output_qwen_14b_1k_dir_1218  --max_len 938  --num_train_epochs 5  --train_batch_size 2 --log_steps 20


    """
    
    nohup sh train.sh > log/qwen_14b_train_1k_1218.txt 2>&1 &
    tail -F log/qwen_14b_train_1k_1218.txt
    

    """





