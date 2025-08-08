from config import RNNConfig
import torch
from torch import nn, optim
import torch.nn. functional as F
from torch.nn import init

class RNNDataset(torch.utils.data.Dataset):
    def __init__(self,dataset,tokenizer,sequence_length):
        self.dataset=dataset
        self.tokenizer=tokenizer
        self.params = {'padding':'max_length','max_length':sequence_length,'truncation':True,'return_tensors':'pt'}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        inputs_str= f"{self.tokenizer.bos_token}{self.dataset[idx]['de']}{self.tokenizer.sep_token}{self.dataset[idx]['en']}"
        labels_str= f"{self.dataset[idx]['de']}{self.tokenizer.sep_token}{self.dataset[idx]['en']}{self.tokenizer.eos_token}"
        input_ids = self.tokenizer(inputs_str, **self.params).input_ids[0]
        labels = self.tokenizer(labels_str, **self.params).input_ids[0]

        indices = (labels == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        if len(indices)>0:
            indices=indices[0]
            labels[:indices]=self.tokenizer.pad_token_id  

        return input_ids,labels,inputs_str,labels_str


class RNNLayer(nn.Module):
    def __init__(self,config:RNNConfig):
        super().__init__()
        self.config=config
        self.embedding = nn.Embedding(config.vocab_size,config.hidden_dim)
        self.w_hx = nn.Linear(config.hidden_dim,config.hidden_dim,bias=False)
        self.w_hh = nn.Linear(config.hidden_dim,config.hidden_dim,bias=False)
        self.w_yh = nn.Linear(config.hidden_dim,config.vocab_size,bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # GPT uses normal initialization for linear layers
            init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self,input_ids):
        bsz,seq = input_ids.shape
        assert bsz==1,'For prediction please pass single input at a time.'
        assert seq<self.config.seq_len, "Please remove padding and pass only the sentence which need to translate."
        
        
        x_t = self.embedding(input_ids)
        h_t = torch.zeros((bsz,self.config.hidden_dim)).to(input_ids.device).to(x_t.dtype)

        # Encoding
        for i in range(seq):
            combined = torch.add(self.w_hx(x_t[:,i,:]), self.w_hh(h_t))
            h_t = torch.tanh(combined)
        
        # Decoding
        prev_emb = x_t[:,-1]
        decoded_output = []
        for i in range(self.config.seq_len - seq):
            combined = torch.add(self.w_hx(prev_emb), self.w_hh(h_t))
            h_t = torch.tanh(combined)
            
            o_t = self.w_yh(h_t)
            token=torch.argmax(o_t,dim=-1).detach()
            prev_emb = self.embedding(token)
            decoded_output.append(token)
            if token.item() == self.config.eos_token_id:
                break
            
        decoded_output = torch.stack(decoded_output,dim=1)
        return decoded_output

    def forward(self,input_ids,h_t=None,labels=None):
        bsz,seq = input_ids.shape # X =(batch size, sequence length)
        x_t = self.embedding(input_ids) #(batch size, sequence_length, hidden size)
        if h_t is None:
            h_t = torch.zeros((bsz,self.config.hidden_dim)).to(input_ids.device).to(x_t.dtype)
            
        # RNN Calculation
        outputs = []
        for i in range(seq):
            combined = torch.add(self.w_hx(x_t[:,i,:]), self.w_hh(h_t))
            h_t = F.tanh(combined)
            o_t = self.w_yh(h_t)
            outputs.append(o_t)
        outputs = torch.stack(outputs,dim=1)
        
        loss=None
        if labels is not None:
            loss = F.cross_entropy(outputs.view(-1,outputs.size(-1)),labels.view(-1),ignore_index=self.config.pad_token_id)
            
        return (loss , outputs) if labels is not None else outputs
        