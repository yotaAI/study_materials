import pandas as pd
import numpy as np
import os
import torch
from torch import nn,optim
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')

from config import RNNConfig
from module import RNNLayer, RNNDataset
from trainer import training

if __name__=='__main__':
    config=RNNConfig()
    # Tokenizer
    tokenizer=AutoTokenizer.from_pretrained('gpt2',use_fast=True)
    special_tokens_dict = {
        'bos_token': '<BOS>',
        'eos_token': '<EOS>',
        'pad_token': '<PAD>',
        'unk_token': '<UNK>',
        'sep_token': '<SEP>',
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    config.vocab_size=len(tokenizer)
    config.pad_token_id=tokenizer.pad_token_id
    config.eos_token_id=tokenizer.eos_token_id

    # # Dataset
    # de_en = load_dataset("parquet", data_files="./dataset/train_wmt.parquet")
    # train_ds = RNNDataset(de_en['train'],tokenizer,config.seq_len)
    # train_loader=torch.utils.data.DataLoader(train_ds,batch_size=config.batch_size,shuffle=True)

    # Huggingface Dataset
    dataset = load_dataset("parquet", data_files=config.dataset_pth)
    dataset.set_format("torch", columns=['inputs_ids', 'labels_ids'])
    train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=config.batch_size, shuffle=True)
    # Model
    model = RNNLayer(config).to(device).to(torch.bfloat16)
    if config.pretrained_pth:
        print("Loading Pretrained Model ....")
        state_dict=torch.load(config.pretrained_pth,weights_only=False)
        model.load_state_dict(state_dict['model'])
    optimizer=optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=0.9)
    
    print("Training .....")
    training(config,model,train_loader,(optimizer,scheduler),device)
