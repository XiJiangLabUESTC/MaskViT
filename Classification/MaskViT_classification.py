# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler,Dataset
from transformers import data
from models.detector import Detector
import util.misc as utils
from util.scheduler import create_scheduler
import os
import timm
def get_args_parser():
    parser = argparse.ArgumentParser('Set YOLOS', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-6, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--eval_size', default=224, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='start epoch')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,help='gradient clipping max norm')

    parser.add_argument('--use_checkpoint', action='store_true',help='use checkpoint.checkpoint to save mem')
    # scheduler
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='warmupcos', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step", options:"step", "warmupcos"')
    ## step
    parser.add_argument('--lr_drop', default=100, type=int)  
    ## warmupcosine

    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # * model setting
    parser.add_argument("--det_token_num", default=50, type=int)
    parser.add_argument('--backbone_name', default='small', type=str)
    parser.add_argument('--num_classes',default=21,type=int)
    parser.add_argument('--init_pe_size', default=(224,224),nargs='+', type=int,help="init pe size (h,w)")
    parser.add_argument('--mid_pe_size', default=(224,224),nargs='+', type=int,help="mid pe size (h,w)")

    parser.add_argument('--pre_trained', default='./few_shot_classification/pretrained/deit_small_pretrain_200epoch.pth')
    parser.add_argument('--resume',default='./few_shot_classification/pretrained/pretrain_finetune_model_10.pth')
    parser.add_argument('--task',default='classification')
    parser.add_argument('--fix_backbone',action='store_true')
    # dataset parameters
    parser.add_argument('--base_train_x',default='/media/D/yazid/YOLOS/few_shot_classification/data/train_base_X.npy')
    parser.add_argument('--base_train_y',default='/media/D/yazid/YOLOS/few_shot_classification/data/train_base_y.npy')
    parser.add_argument('--base_test_x',default='/media/D/yazid/YOLOS/few_shot_classification/data/test_base_X.npy')
    parser.add_argument('--base_test_y',default='/media/D/yazid/YOLOS/few_shot_classification/data/test_base_y.npy')

    parser.add_argument('--novel_train_x',default='/media/D/yazid/YOLOS/few_shot_classification/data/activelearning/novel_X_5_shots.npy')
    parser.add_argument('--novel_train_y',default='/media/D/yazid/YOLOS/few_shot_classification/data/activelearning/novel_y_5_shots.npy')
    parser.add_argument('--novel_test_x',default='/media/D/yazid/YOLOS/few_shot_classification/data/activelearning/test_X_5_shots.npy')
    parser.add_argument('--novel_test_y',default='/media/D/yazid/YOLOS/few_shot_classification/data/activelearning/test_y_5_shots.npy')
    parser.add_argument('--eval',action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=2, type=int)

    # file path setting
    parser.add_argument('--output_dir', default='./few_shot_classification/result/active_5_shots/MaskViT_discrete')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = timm.create_model()
    model=model.to(device)
    model_without_ddp = model
    # fixed the backbone parameters for fine true in noval 
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module
    # change the classifier head for new object detection task
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if args.fix_backbone:
        optimizer = torch.optim.AdamW(model.backbone.head.parameters(), lr=1e-3,weight_decay=args.weight_decay)
        print('Fixed the Backbone!!!')
    else:
        optimizer  = build_optimizer(model_without_ddp, args)
        print('Train on The whole model')
    lr_scheduler, _ = create_scheduler(args, optimizer)
    novel_test_sampler,dataloader_novel_test = creat_dataloader(np.load(args.novel_test_x).tolist(),np.load(args.novel_test_y).tolist(),transform_test,args)
    novel_train_sampler,dataloader_novel_train = creat_dataloader(np.load(args.novel_train_x).tolist(),np.load(args.novel_train_y).tolist(),transform_train,args)
    if args.eval:
        epoch_loss,epoch_acc=evaluate_with_mask_generate(dataloader_novel_test,model,device)
        print('Test Acc: %.4f'/%(epoch_acc))
        return 
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1,reduction='sum')
    output_dir = Path(args.output_dir)
    print(output_dir)
    os.makedirs(output_dir,exist_ok=True)
    # for epoch_init in range(5):
    #     if args.distributed:
    #         base_train_sampler.set_epoch(epoch_init)
    #     optimizer_init=build_optimizer(model_without_ddp, args)
    #     epoch_loss,epoch_acc=train_one_epoch_without_mask(dataloader_base_train,model,device,criterion,optimizer_init)
    #     print('Initial training: Epoch:%d  Base Loss:%.4f  Base TrainAcc:%.4f'%(epoch_init,epoch_loss,epoch_acc))
    # checkpoint_path= './few_shot_classification/pretrain_finetune_model_10.pth'
    # utils.save_on_master({'model': model.state_dict()},checkpoint_path)
    for epoch in range(args.epochs):  
        if epoch % 10 ==0:
            sampler,dataloader_base_train=add_selected_base(model,criterion,device,args)
        if args.distributed:
            sampler.set_epoch(epoch)
            novel_train_sampler.set_epoch(epoch)
        epoch_loss,epoch_acc=train_one_epoch_with_mask(dataloader_base_train,model,device,criterion,optimizer)
        epoch_loss_novel,epoch_acc_novel=train_one_epoch_without_mask(dataloader_novel_train,model,device,criterion,optimizer)
        print('EPOCH:%d   Base Loss:%.4f    Base TrainAcc:%.4f  Novel Loss:%.4f   Novel TrainAcc:%.4f'%(epoch,epoch_loss,epoch_acc,epoch_loss_novel,epoch_acc_novel))
        lr_scheduler.step(epoch)
        if epoch>=20 and (epoch+1)%3==0:
            novel_test_acc=evaluate_with_mask(dataloader_novel_test,model,device)
            print('EPOCH:%d   Loss:%.4f    TrainAcc:%.4f   NovelTestAcc:%.4f'%(epoch,epoch_loss,epoch_acc,novel_test_acc))
            checkpoint_path= output_dir / 'deit_small_{}_epoch.pth'.format(epoch+1)
            utils.save_on_master(
                {'model': model.state_dict(),
                'args': args,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch},checkpoint_path)
            log_stats = {
                'EPOCH':epoch,
                'Loss':epoch_loss,
                'TrainAcc':epoch_acc,
                'NovelTestAcc':novel_test_acc}
            if args.output_dir and utils.is_main_process():
                with (output_dir / 'logs_checkpoints.txt').open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

from PIL import Image
import torchvision.transforms as transforms
transform_train = transforms.Compose([
    transforms.PILToTensor(),
    transforms.RandAugment(),
    transforms.RandomResizedCrop((224,224),scale=(0.08,1.0),ratio=(0.75,1.33),interpolation=2),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomErasing(),
    ])
transform_test = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
class MyDataset(Dataset):
    def __init__(self, X,y,transform):
        self.X=X
        self.y=y
        self.transform = transform
        self.path2class={
            'apple':0,
            'apple_tree':1,
            'avocado':2,
            'capsicum':3,
            'Chinee_apple':4,
            'grass_clover':5,
            'Lantana':6,
            'mango':7,
            'orange':8,
            'Parkinsonia':9,
            'Parthenium':10,
            'Prickly_acacia':11,
            'rockmelon':12,
            'Rubber_vine':13,
            'Siam_weed':14,
            'Snake_weed':15,
            'strawberry':16,
            'deep_seeding':17,
            'lettuce':18,
            'maize':19,
            'tomato':20

        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img_path = self.X[idx]
        if isinstance(img_path,list):
            img_path=img_path[0]
        img_name = self.y[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        label = self.path2class[img_name]
        img_transformed = self.transform(img)
        return img_transformed,label,img_path

def train_one_epoch_without_mask(dataloader,model,device,criterion,optimizer,max_norm=-1):
    model.train()
    epoch_accuracy,epoch_loss=0,0
    for data,label,imgpath in dataloader:
        label=torch.tensor(label).to(device)
        data = data.to(device)
        B=label.shape[0]
        mask=torch.full((B,196),True)
        output = model.forward_with_mask(data,mask)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        acc = (output.argmax(dim=1) == label).float().mean().item()
        epoch_accuracy += acc / len(dataloader)
        epoch_loss += (loss.item()) / len(dataloader)
    return epoch_loss,epoch_accuracy

def train_one_epoch_with_mask(dataloader,model,device,criterion,optimizer,max_norm=-1):
    def backward_hook(module,grad_in,grad_out):
        grad_block.append(grad_out[0].detach().cpu())
    model.train()
    criterion.train()
    criterion=criterion.to(device)
    model=model.to(device)
    model.backbone.patch_embed.register_backward_hook(backward_hook)
    epoch_accuracy,epoch_loss=0,0
    for data,label,imgpath in dataloader:
        grad_block=list()
        label=torch.tensor(label).to(device)
        data=data.to(device)
        output=model(data)
        loss = criterion(output,label)
        model.zero_grad()
        loss.backward()
        a=grad_block[0].abs().sum(dim=2)#patch_num*patch_dim
        #even the random has s fixed number.
        #study for paper not for problem
        #a worker
        values,index=torch.topk(a,dim=1,k=49)
        mask=(a.T>=values[:,-1].T).T
        #mask=mask_filter(mask)
        mask=mask.to(device)
        out=model.forward_with_mask(data,mask)
        loss = criterion(out,label)
        model.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        acc = (output.argmax(dim=1) == label).float().mean().item()
        epoch_accuracy += acc / len(dataloader)
        epoch_loss += (loss.item()) / len(dataloader)
    return epoch_loss,epoch_accuracy

def add_selected_base(model,criterion,device,args):
    base_path='/media/D/yazid/argin/AgriNet/base'
    novel_train_x=np.load(args.novel_train_x).tolist()
    shots=len(novel_train_x)//4
    novel_train_x,novel_train_y=[],[]
    # This line will remove the mask for novel dataset.
    #novel_train_y=np.load(args.novel_train_y).tolist()
    classname=os.listdir(base_path)
    for item in classname:
        images=os.listdir(os.path.join(base_path,item))
        images=[os.path.join(base_path,item,img) for img in images]
        y=[item for _ in range(len(images))]
        selected=get_loss_dict(model,criterion,images,y,device,shots)
        novel_train_x.extend(selected)
        novel_train_y.extend([item for _ in range(len(selected))])
    return creat_dataloader(novel_train_x,novel_train_y,transform_train,args)
    
def mask_filter(mask):
    B=mask.shape[0]
    mask=mask.reshape(B,14,14)
    x_sum=torch.sum(mask,dim=2)
    y_sum=torch.sum(mask,dim=1)
    arg_x=torch.argmax(x_sum,dim=1)
    arg_y=torch.argmax(y_sum,dim=1)
    arg_x_center=torch.where(arg_x-3>=0,arg_x,3)
    arg_x_center=torch.where(arg_x+3<=13,arg_x_center,10)
    arg_y_center=torch.where(arg_y-3>=0,arg_y,3)
    arg_y_center=torch.where(arg_y+3<=13,arg_y_center,10)
    mask_proj=torch.full((B,14,14),False)
    for i in range(B):
        x,y=arg_x_center[i],arg_y_center[i]
        mask_proj[i,x-3:x+4,y-3:y+4]=True
    mask_proj=mask_proj.reshape(B,-1)
    return mask_proj
    
def evaluate_with_mask(dataloader,model,device):
    model.eval()
    with torch.no_grad():
        epoch_accuracy = 0
        for data, label,imgpath in dataloader:
            label=torch.tensor([item for item in label]).to(device)
            data = data.to(device)
            B=data.tensors.shape[0]
            mask=torch.full((B,196),True)
            mask=mask.to(device)
            output = model.forward_with_mask(data,mask)
            acc = (output.argmax(dim=1) == label).float().mean().item()
            epoch_accuracy += acc / len(dataloader)
    return epoch_accuracy

def evaluate_with_mask_generate(dataloader,model,device):
    def backward_hook(module,grad_in,grad_out):
        grad_block.append(grad_out[0].detach().cpu())
    criterion=torch.nn.CrossEntropyLoss(label_smoothing=0.1,reduction='sum')
    criterion=criterion.to(device)
    model=model.to(device)
    model.backbone.patch_embed.register_backward_hook(backward_hook)
    epoch_accuracy,epoch_loss=0,0
    for data,label,imgpath in dataloader:
        model.train()
        grad_block=list()
        label=torch.tensor(label).to(device)
        data=data.to(device)
        output=model(data)
        loss = criterion(output,label)
        model.zero_grad()
        loss.backward()
        a=grad_block[0].abs().sum(dim=2)#patch_num*patch_dim
        #even the random has s fixed number.
        #study for paper not for problem
        #a worker
        values,index=torch.topk(a,dim=1,k=49)
        mask=(a.T>=values[:,-1].T).T
        mask=mask_filter(mask)
        mask=mask.to(device)
        with torch.no_grad():
            model.eval()
            out=model.forward_with_mask(data,mask)
            loss = criterion(out,label)
            acc = (output.argmax(dim=1) == label).float().mean().item()
        epoch_accuracy += acc / len(dataloader)
        epoch_loss += (loss.item()) / len(dataloader)
    return epoch_loss,epoch_accuracy
    

def build_optimizer(model, args):
    if hasattr(model.backbone, 'no_weight_decay'):
        skip = model.backbone.no_weight_decay()
    head = []
    backbone_decay = []
    backbone_no_decay = []
    for name, param in model.named_parameters():
        if "backbone" not in name and param.requires_grad:
            head.append(param)
        if "backbone" in name and param.requires_grad:
            if len(param.shape) == 1 or name.endswith(".bias") or name.split('.')[-1] in skip:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)
    param_dicts = [
        {"params": head},
        {"params": backbone_no_decay, "weight_decay": 0., "lr": args.lr},
        {"params": backbone_decay, "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                weight_decay=args.weight_decay)
    return optimizer

def creat_dataloader(X,y,transform,args,batch_size=-1):
    dataset=MyDataset(X,y,transform)
    if args.distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
    if batch_size<0:
        batch_size=args.batch_size
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler,collate_fn=utils.collate_fn, num_workers=args.num_workers)
    return sampler,data_loader

def get_loss_dict(model,criterion,X,y, device,shots):
    model.eval()
    minLoss=99999
    dataset=MyDataset(X,y,transform=transform_test)
    dataloader=DataLoader(dataset,shots)
    with torch.no_grad():
        for samples, targets, paths in dataloader:
            samples = samples.to(device)
            targets = torch.tensor(targets).to(device)
            outputs = model(samples)
            loss = criterion(outputs, targets)
            if loss.item()<minLoss:
                minLoss=loss.item()
                selected_path=paths
    return selected_path
if __name__ == '__main__':
    parser = argparse.ArgumentParser('YOLOS training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)