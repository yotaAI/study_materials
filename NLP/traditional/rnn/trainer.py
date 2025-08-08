import os
import json
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model,train_loader,optimizer,scheduler,device,epoch,scaler=None,writer=None):
    model.train()
    loader=tqdm(train_loader)
    total_loss=0

    for idx,data in enumerate(loader):

        optimizer.zero_grad()
        input_ids=data['inputs_ids'].to(device)
        labels=data['labels_ids'].to(device)
        if scaler is not None:
            with autocast():
                loss,output = model(input_ids=input_ids,labels=labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss,output = model(input_ids=input_ids,labels=labels)
            loss.backward()
            optimizer.step()
        total_loss+=loss.item()

        # Logging
        if idx%100==0:
            loader.set_postfix({'loss':loss.item()})
            if writer is not None:
                global_step = epoch * len(train_loader) + idx
                writer.add_scalar('train/loss', loss.item(), global_step)
                if scheduler is not None:
                    writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step)
            
    if scheduler is not None:
        scheduler.step()
    
    loader.set_postfix({'loss' : total_loss / len(train_loader),'lr': scheduler.get_last_lr()[0] if scheduler is not None else 'none'})
    
    return total_loss / len(train_loader)


def save_model(model,optimizer,scheduler,loss,epoch,config,path="./tmp"):
    os.makedirs(path,exist_ok=True)
    with open(os.path.join(config.save_pth, 'config.json'), 'w+') as f:
        json.dump(config.__dict__, f, indent=4)
    state_dict = {
        'model':model.state_dict(),
        'optmizer': optimizer.state_dict(),
        'loss':loss,
        'scheduler':scheduler,
    }
    torch.save(state_dict,os.path.join(path,f'state_dict_e{epoch}.pt'))

# Training
def training(config,model,train_loader,optimizer,device):
    
    writer=SummaryWriter(config.log_dir)

    # Initialize GradScaler for mixed precision
    if config.use_amp:
        scaler = GradScaler()
    else:
        scaler = None

    scheduler=None
    if isinstance(optimizer,tuple):
        scheduler=optimizer[1]
        optimizer=optimizer[0]

    for epoch in range(config.epochs):
        loss = train_one_epoch(model,train_loader,optimizer,scheduler,device,epoch,scaler,writer)
        
        save_model(model,optimizer,scheduler,loss,epoch,config,config.save_pth)
        
        print(f'Loss after Epoch {epoch} is {loss}')
    writer.close()

        