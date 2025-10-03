# -*- coding: utf-8 -*-
# @Author: Kevin Kamm
# @Date:   2025-09-25 08:35:23
# @Last Modified by:   Kevin Kamm
# @Last Modified time: 2025-09-25 08:35:40
from tqdm import tqdm
from processes import *
import torch
from typing import List, Tuple, Optional, Callable, Generator

# assumption of process shape = (batch,time,[states,payoff])
class DeepOSNet(torch.nn.Module):
    def __init__(self,
                 d:int = 1, # number of states
                 N:int = 2, # number of time steps
                 latent_dim: List[int]=[51], 
                 outputDims: int=1, 
                 batch_size:int =2**12):
        super().__init__()
        self.outputDims = outputDims
        self.batch_size=batch_size
        self.d = d+1 # since price adds a dimension
        self.N = N-1 # for "continuation" decision
        self.latent_dim = latent_dim
        self.latent_dim.insert(0,self.d)

        deepOSNet = [self._deepOSBlock_in()]
        for i in range(1,len(self.latent_dim)):
            deepOSNet.append(self._deepOSBlock(self.latent_dim[i-1],self.latent_dim[i]))
        deepOSNet.append(self._deepOSBlock_out(self.latent_dim[-1]))
        self.deepOSNet = torch.nn.Sequential(*deepOSNet)
        
    def _deepOSBlock_in(self):
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(self.N*self.d,eps=1e-6,momentum=0.9),
            torch.nn.Unflatten(1,(self.N,self.d))
        )    

    def _deepOSBlock(self,latent_last,latent_curr):
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=latent_last,out_features=latent_curr),
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(self.N*latent_curr,eps=1e-6,momentum=0.9),
            torch.nn.Unflatten(1,(self.N,latent_curr)),
            torch.nn.ReLU()
        )
    
    def _deepOSBlock_out(self,latent_last):
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=latent_last,out_features=self.outputDims),
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(self.N*self.outputDims,eps=1e-6,momentum=0.9),
            torch.nn.Unflatten(1,(self.N,self.outputDims)),
            torch.nn.Sigmoid()
        )  

    def forward(self,x):
        return self.deepOSNet(x)
    
    def train_loop(self,model:Callable[[int],Generator],train_steps:int = 1500,milestones:Optional[List[int]]=None,lr_value:Optional[float]=None,gamma:Optional[float]=None):
        if milestones is None:
            milestones = [int(train_steps/6 + self.d / 5), int(train_steps/2 + 3 * self.d / 5)]

        if lr_value is None:
            self.lr_value = 0.05

        if gamma is None:
            gamma = 0.1

        opt = torch.optim.Adam(self.parameters(),lr=self.lr_value,betas=(0.9,0.999),eps=1e-8)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,milestones,gamma=gamma)

        gen = model(self.batch_size) #state and price process, price last on last axis
        self.train()
        trainErr=[]
        trainPrice=[]
        for _ in (pbar:=tqdm(range(train_steps))):
            sp = next(gen)
            opt.zero_grad()
            loss,payoff = self.train_step(sp)
            loss.backward()
            opt.step()
            scheduler.step()
            pbar.set_postfix({'loss':loss.item(),'payoff':payoff.item(),'learning rate':scheduler.get_last_lr()})
            trainErr.append(loss.item())
            trainPrice.append(payoff.item())
        return trainErr, trainPrice

    def train_step(self,sp:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        p=sp[:,:,-1] # process and price process

        nets = self(sp[:,:-1,:])
        nets = nets.permute((0,2,1))
        u_list = [nets[:, :, 0]]
        u_sum = u_list[-1]
        for k in range(1, self.N):
            u_list.append(nets[:, :, k] * (1. - u_sum))
            u_sum = u_sum + u_list[-1] 

        u_list.append(1. - u_sum)
        u_stack = torch.concatenate(u_list, axis=1)
        loss = ((-u_stack * p).sum(1)).mean()

        idx = torch.argmax(torch.cumsum(u_stack.detach(), axis=1) + (u_stack.detach() >= 1).to(torch.uint8),axis=1).to(torch.int64).reshape((-1,1))
        price = torch.gather(p.detach(),1,idx).mean()

        return loss, price

    def evalStopping(self,t:torch.Tensor,sp:torch.Tensor):
        # self.eval() # sic! do not use eval mode here
        self.train()
        with torch.no_grad():
            p=sp[:,:,-1]

            nets = self(sp[:,:-1,:])
            nets = nets.permute((0,2,1))
            u_list = [nets[:, :, 0]]
            u_sum = u_list[-1]
            for k in range(1, self.N):
                u_list.append(nets[:, :, k] * (1. - u_sum))
                u_sum = u_sum + u_list[-1]

            u_list.append(1. - u_sum)
            u_stack = torch.concatenate(u_list, axis=1)

            idx = torch.argmax(torch.cumsum(u_stack.detach(), axis=1) + (u_stack.detach() >= 1).to(torch.uint8),axis=1).to(torch.int64).reshape((-1,1))
            stopped_payoffs = torch.gather(p.detach(),1,idx)
            tau = t.ravel()[idx]

        return tau,stopped_payoffs