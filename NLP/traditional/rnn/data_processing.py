from transformers import AutoTokenizer
from datasets import load_dataset,Dataset
import multiprocessing as mp


## Setting Tokenizer
tokenizer=AutoTokenizer.from_pretrained('gpt2',use_fast=True)
special_tokens_dict = {
    'bos_token': '<BOS>',
    'eos_token': '<EOS>',
    'pad_token': '<PAD>',
    'unk_token': '<UNK>',
    'sep_token': '<SEP>',
}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

## Dataset to Parquet
dataset = load_dataset("wmt/wmt14",'de-en')
dataset = Dataset.from_list(dataset['train']['translation'])
dataset.to_parquet("./dataset/train_wmt.parquet")

## Perquet Dataset Formatting
def formatting_function(examples):
    inputs_text = []
    labels_text = []
    for de_text, en_text in zip(examples['de'], examples['en']):
        inputs = f"{tokenizer.bos_token}{de_text}{tokenizer.sep_token}{en_text}"
        labels = f"{de_text}{tokenizer.sep_token}{en_text}{tokenizer.eos_token}"
        inputs_text.append(inputs)
        labels_text.append(labels)

    return {'inputs': inputs_text,'labels':labels_text}

dataset = load_dataset("parquet", data_files="./dataset/train_wmt.parquet")
formatted_dataset = dataset.map(formatting_function,batched=True,remove_columns=['de', 'en'])
formatted_dataset['train'].to_parquet("./dataset/wmt_dataset_de_en_formatted.parquet")

## Tokenizing Dataset
seq_length=512
params = {'padding':'max_length','max_length':seq_length,'truncation':True,'return_tensors':'pt'}
def tokenize_function(examples):
    inputs_ids = []
    labels_ids = []
    for inputs, labels in zip(examples['inputs'], examples['labels']):
        input_id = tokenizer(inputs, **params).input_ids[0]
        label_id = tokenizer(labels, **params).input_ids[0]
        
        indices = (label_id == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        if len(indices)>0:
            indices=indices[0]
            label_id[:indices]=tokenizer.pad_token_id
            
        inputs_ids.append(input_id)
        labels_ids.append(label_id)

    return {'inputs_ids': inputs_ids,'labels_ids':labels_ids}

dataset = load_dataset("parquet", data_files="./dataset/wmt_dataset_de_en_formatted.parquet")
formatted_dataset = dataset.map(tokenize_function,batched=True,num_proc=mp.cpu_count())
formatted_dataset['train'].to_parquet("./dataset/wmt_dataset_de_en_tokenized.parquet")
