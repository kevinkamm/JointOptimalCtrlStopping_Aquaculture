# -*- coding: utf-8 -*-
# @Author: Kevin Kamm
# @Date:   2025-09-19 11:19:24
# @Last Modified by:   Kevin Kamm
# @Last Modified time: 2025-10-03 13:38:07
from typing import Tuple
from processes import *
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
fontsize = 16

class CtrlStoppingSolver(ABC):
    def __init__(self,
                 r:float,
                 k:Callable[[ArrayLike], ArrayLike],
                 g:Callable[[ArrayLike], ArrayLike],
                 bnds:List[Union[Tuple[float,float,int],Tuple[float,float]]],
                 processes:List[Process],
                 **kwargs):
        pass
        

class JointOptCtrlStopping():
    def __init__(self,
                 r:float,
                 k:Callable[[ArrayLike], ArrayLike],
                 g:Callable[[ArrayLike], ArrayLike],
                 bnds:List[Union[Tuple[float,float,int],Tuple[float,float]]],
                 processes:List[Process],
                 solver:CtrlStoppingSolver,
                 solverArgs:dict,
                 device:torch.device=torch.device('cpu'),
                 dtype:torch.dtype=torch.float32):
        self.r = r
        self.k = k
        self.g = g
        self.processes = processes
        self.solver = solver(r,k,g,bnds,processes,**solverArgs,device=device,dtype=dtype)
        self.solverArgs = solverArgs
        self.V = None
        self.U = None
    
    def solve(self,**kwargs):
        self.V, self.U = self.solver.solve(**kwargs)
        return self.V, self.U

    def simulate(self,t, x0:ArrayLike, dWt, U:Optional[Callable]=None, V:Optional[Callable]=None, fuzzy:float=0.0):
        if U is None:
            if self.U is None:
                ValueError("No control function provided")
            U = self.U
        if V is None:
            V = self.V
        M = dWt.shape[0]
        t = t.reshape(-1)
        r = self.r
        T = t[-1]
        device = t.device
        dtype = t.dtype
        Nt = t.size(0)
        dt = (t[1]-t[0])
        d = x0.numel()
        assert d == len(self.processes), "Dimension of x0 does not match number of processes"
        assert d==dWt.shape[-1], "Dimension of dWt does not match number of processes"

        
        xt = x0.reshape((1,1,d)) * torch.ones((M,Nt,d),dtype=dtype,device=device)

        discount = torch.exp(-r * dt)

        Vsim = torch.zeros((M,Nt),dtype=dtype,device=device)

         
        tt = T*torch.ones((M,Nt),dtype=dtype,device=device)
        ut = torch.zeros((M,Nt),dtype=dtype,device=device)

        for i in range(0,Nt-1):
            X =torch.concatenate([t[i]*torch.ones((M,1),dtype=dtype,device=device),xt[:,i,:].view(-1,d)],axis=1)
            utmp = U(X).view(-1,1)
            
            ut[:,i] = utmp.view(-1)
            if V is not None:
                vtmp = V(X).view(-1,1) 
                tt[:,i] = torch.where(vtmp.view(-1)-fuzzy <= self.g(t[i]*torch.ones((M,),dtype=dtype,device=device), *[xt[:,i,j] for j in range(d)]).view(-1), t[i], T) 


            for j in range(len(self.processes)):
                if self.processes[j].isControllable:
                    drift = self.processes[j].drift(t[i]*torch.ones((M,),dtype=dtype,device=device), xt[:,i,j], utmp.view(-1))
                    diffusion = self.processes[j].diffusion(t[i]*torch.ones((M,),dtype=dtype,device=device), xt[:,i,j], utmp.view(-1))
                else:
                    drift = self.processes[j].drift(t[i]*torch.ones((M,),dtype=dtype,device=device), xt[:,i,j])
                    diffusion = self.processes[j].diffusion(t[i]*torch.ones((M,),dtype=dtype,device=device), xt[:,i,j])
                xt[:,i+1,j] = xt[:,i,j] + drift * dt + diffusion * dWt[:,i,j]

            kt = self.k(t[i+1]*torch.ones((M,),dtype=dtype,device=device), *[xt[:,i+1,j] for j in range(d)], utmp.view(-1))

            Vsim[:,i+1] = discount * kt * dt

        Vsim = torch.cumsum(Vsim, dim=1) + torch.exp(-r * t[None,:]) * self.g(t[None,:], *[xt[:,:,j] for j in range(d)])
        
        ut[:,-1] = ut[:,-2]
        out = torch.concatenate([xt,ut[...,None],Vsim[...,None]], axis=2)
        tauInd = torch.min(torch.abs(t-tt), dim=1, keepdim=False)[1]
        return out.detach(), t[tauInd].detach(), tauInd.detach()

def plotComparison(t:torch.Tensor,Vsim_sim:List[ArrayLike], U_sim:List[ArrayLike], tau_sim:List[ArrayLike], wi:List[int]=[0], labels:List[str]=[]):
    fig = plt.figure(figsize=(16,9))
    # plt.xticks(fontsize=22)
    # plt.yticks(fontsize=22)
    num_lines = len(Vsim_sim)
    if len(labels)==0:
        labels = [f"Method {i+1}" for i in range(num_lines)]
    handles = []
    lbls = []
    # reds = plt.cm.Reds(np.linspace(0.4, 1, num_lines)) # Avoid very light colors
    # blues = plt.cm.Blues(np.linspace(0.4, 1, num_lines)) # Avoid very light colors
    # greens = plt.cm.Greens(np.linspace(0.4, 1, num_lines)) # Avoid very light colors
    colors = plt.cm.rainbow(np.linspace(0, 1, num_lines))
    
    for i in range(len(wi)):
        ax_a = fig.add_subplot(1,len(wi),i+1)
        ax_b = ax_a.twinx()
        for j in range(len(Vsim_sim)):
            c = colors[j]
            tau = tau_sim[j].view(-1)[wi[i]].detach().cpu().numpy()
            # tauInd = np.argmin(np.abs(t.detach().cpu().numpy().ravel()-tau))
            ax_a.plot(t.detach().cpu().numpy().ravel(), Vsim_sim[j][wi[i],:].detach().cpu().numpy().ravel(), label=labels[j]+" V", color=c, linestyle=':')
            ax_b.plot(t.detach().cpu().numpy().ravel(), U_sim[j][wi[i],:].detach().cpu().numpy().ravel(), label=labels[j]+" U", color=c, linestyle='--')
            ax_a.axvline(tau, label=labels[j]+" $\\tau$", color=c, linestyle='-')
        ax_a.set_xlabel("Time", fontsize=fontsize)
        ax_a.set_ylabel("Value Function", fontsize=fontsize)
        ax_b.set_ylabel("Control", fontsize=fontsize)
        ax_a.tick_params(axis='both', which='both', labelsize=fontsize)
        ax_b.tick_params(axis='both', which='both', labelsize=fontsize)
        
        
        h, l = ax_a.get_legend_handles_labels()
        handles.extend(h)
        lbls.extend(l)
        h, l = ax_b.get_legend_handles_labels()
        handles.extend(h)
        lbls.extend(l)

    unique = dict(zip(lbls, handles))
    plt.figlegend(unique.values(), unique.keys(), loc = 'lower center', ncol=3, labelspacing=0.1, fontsize=fontsize)
    plt.tight_layout(rect=[0, 0.15, 1, 1.15]) 
    # plt.tight_layout(rect=[0, 0.00, 1, 0.5])
    
    return fig

def plotMethod(t, sys, tau, tauInd, wi:List[int]=[0], labels:List[str]=[], title:str="Method"):
    sys = sys.detach().cpu().numpy()
    t = t.reshape(-1).detach().cpu().numpy()
    tau = tau.reshape(-1).detach().cpu().numpy()
    tauInd = tauInd.reshape(-1).detach().cpu().numpy()
    num_lines = sys.shape[-1]
    if len(labels)==0:
        labels = [f"Var {i+1}" for i in range(num_lines)]
        labels[-2] = "Control"
        labels[-1] = "Value"
    colors = plt.cm.rainbow(np.linspace(0, 1, num_lines))
    ctrlColor = 'g'
    stopColor = 'tab:gray'

    fig = plt.figure(figsize=(16,9))
    fig.suptitle(title, fontsize=16)
    columns = 2
    quotient, remainder = divmod(num_lines-1, columns)
    rows = quotient
    if remainder:
        rows = rows + 1

    plot_idx = [i for i in range(0, num_lines)]
    plot_idx.remove(num_lines-2) # control
    handles = []
    lbls = []
    for i,n in enumerate(plot_idx):
        ax_a = plt.subplot(rows, columns, i+1)
        ax_a.set_xlabel('Time', fontsize=fontsize)
        ax_a.set_ylabel(labels[n], color=colors[n], fontsize=fontsize)
        ax_b = ax_a.twinx()
        ax_b.set_ylabel('Control', color=ctrlColor, fontsize=fontsize)

        for j,w in enumerate(wi):
            tau_wi = tauInd[w]
            tau_w = tau[w]
            ax_a.plot(t[:tau_wi], sys[w,:tau_wi,n], label=labels[n], color=colors[n], linestyle='-')
            ax_b.plot(t[:tau_wi], sys[w,:tau_wi,-2], label=labels[-2], color=ctrlColor, linestyle='-') 
            ax_a.axvline(tau_w, label=f'Stopping Time $\\tau$={tau_w:.2f}', color=stopColor, linestyle=':')

            if j==0:
                h, l = ax_a.get_legend_handles_labels()
                handles.extend(h)
                lbls.extend(l)
                h, l = ax_b.get_legend_handles_labels()
                handles.extend(h)
                lbls.extend(l)
            ax_a.plot(t[tau_wi:], sys[w,tau_wi:,n], color=colors[n], linestyle='-', alpha=0.5)
            ax_b.plot(t[tau_wi:], sys[w,tau_wi:,-2], color=ctrlColor, linestyle='-', alpha=0.5) 
        ax_a.tick_params(axis='both', which='both', labelsize=fontsize)
        ax_b.tick_params(axis='both', which='both', labelsize=fontsize)

    unique = dict(zip(lbls, handles))
    plt.figlegend(unique.values(), unique.keys(), loc = 'lower center', ncol=3, labelspacing=0.1, fontsize=fontsize)
    plt.tight_layout(rect=[0, 0.05, 1, 1]) 
    
    return fig
