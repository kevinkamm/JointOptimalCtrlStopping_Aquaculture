# -*- coding: utf-8 -*-
# @Author: Kevin Kamm
# @Date:   2025-09-19 08:51:11
# @Last Modified by:   Kevin Kamm
# @Last Modified time: 2025-10-03 13:38:30
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Callable, Dict,Any
from numpy.typing import DTypeLike, ArrayLike
import numpy as np
import torch
import matplotlib.pyplot as plt

def BrownianMotion(t:Union[np.ndarray, torch.Tensor], 
                   batch:int, 
                   onlyIncrements:bool=False,
                   dim:int=1, 
                   dtype:Optional[DTypeLike]=None, 
                   device:Optional[torch.device]=None,
                   rng:Optional[np.random.Generator]=None,
                   seed:Optional[int]=None)->Union[np.ndarray, torch.Tensor]:
    """Generate Brownian motion increments.

    Parameters
    ----------
    t : Union[np.ndarray, torch.Tensor]
        Time grid.
    batch : int 
        Number of paths to generate.
    onlyIncrements : bool, optional
        If True, only return increments, by default False.  
    dim : int, optional
        Dimension of the Brownian motion, by default 1.
    dtype : Optional[DTypeLike], optional
        Data type of the output, by default None.
    device : Optional[torch.device], optional
        Device for torch tensors, by default None.
    rng : Optional[Generator], optional
        Random number generator for numpy, by default None.
    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        If onlyIncrements is True, returns increments of shape (batch, N-1, dim).
        If onlyIncrements is False, returns a tuple (increments, paths) where
        increments has shape (batch, N-1, dim) and paths has shape (batch, N, dim).
    """
    if dtype is None:
        dtype = t.dtype
    
    if isinstance(t, np.ndarray):
        t=t.reshape(1,-1,1)
        N = np.prod(t.shape)
        if rng is None:
            if seed is not None:
                rng = np.random.default_rng(seed)
            else:
                rng = np.random.default_rng()
        dt = np.diff(t, axis=1)
        dW = np.sqrt(dt)*rng.standard_normal((batch, N-1, dim),dtype=dtype)
        if onlyIncrements:
            return dW
        else:
            W = np.concatenate([np.zeros((batch, 1, dim), dtype=dtype), np.cumsum(dW, axis=1)],axis=1)
        return dW, W
    
    elif isinstance(t, torch.Tensor):
        N = t.numel()
        t = t.view(1, -1, 1)
        if device is None:
            device = t.device
        if rng is None:
            rng = torch.Generator(device)
            if seed is not None:
                rng.manual_seed(seed)
        dt = torch.diff(t, dim=1)
        dW = torch.randn((batch, N-1, dim), generator=rng, dtype=dtype, device=device) * torch.sqrt(dt)
        if onlyIncrements:
            return dW
        else:
            W = torch.cat([torch.zeros((batch, 1, dim), dtype=dtype, device=device), torch.cumsum(dW, dim=1)], dim=1)
        return dW, W
    else:
        raise TypeError("Input must be a numpy array or a torch tensor.")

class Process(ABC):
    """Abstract base class for stochastic processes.
    Conventions
    -----------
    - shape: (batch_size, time, state1, state2, ... , ctrl)
    - t: time
    - x: state variable
    - u: control variable

    Parameters
    ----------
    ABC : _type_
        _description_
    """
    def __init__(self, 
                 params:Optional[Dict[str,Any]] = None, 
                 dtype:Optional[DTypeLike] = None):
        self.params = params
        self.dtype = dtype
        self.isControllable = False

    @abstractmethod
    def drift(self, t:ArrayLike,x:ArrayLike, **kwargs)->ArrayLike:
        pass

    @abstractmethod
    def diffusion(self, t:ArrayLike,x:ArrayLike, **kwargs)->ArrayLike:
        pass
            
    def _simulate_numpy(self, t:ArrayLike,x0:ArrayLike,dW:Optional[ArrayLike]=None, **kwargs)->ArrayLike:
        N = np.prod(t.shape)
        t = t.reshape(1,-1)
        if dW is not None:
            batch = dW.shape[0]
            Xt = np.zeros((batch, N), dtype=t.dtype)
            Xt[:,0] = x0
            for i in range(N-1):
                Xt[:,i+1] = Xt[:,i] + self.drift(t[:,i], Xt[:,i], **kwargs) * (t[0,i+1]-t[0,i]) + self.diffusion(t[:,i], Xt[:,i], **kwargs) * dW[:,i].reshape(-1)
        else:
            Xt = np.zeros((1, N), dtype=t.dtype)
            Xt[:,0] = x0
            for i in range(N-1):
                Xt[:,i+1] = Xt[:,i] + self.drift(t[:,i], Xt[:,i], **kwargs) * (t[0,i+1]-t[0,i])

        return Xt
    
    def _simulate_torch(self, t:ArrayLike,x0:ArrayLike,dW:Optional[ArrayLike]=None, **kwargs)->ArrayLike:
        N = t.numel()
        t = t.view(1,-1)
        if dW is not None:
            batch = dW.size(0)
            Xt = torch.zeros((batch, N), dtype=t.dtype, device=t.device)
            Xt[:,0] = x0
            for i in range(N-1):
                Xt[:,i+1] = Xt[:,i] + self.drift(t[:,i], Xt[:,i], **kwargs) * (t[0,i+1]-t[0,i]) + self.diffusion(t[:,i], Xt[:,i], **kwargs) * dW[:,i].view(-1)
        else:
            Xt = torch.zeros((1, N), dtype=t.dtype, device=t.device)
            Xt[:,0] = x0
            for i in range(N-1):
                Xt[:,i+1] = Xt[:,i] + self.drift(t[:,i], Xt[:,i], **kwargs) * (t[0,i+1]-t[0,i])

        return Xt
    # @abstractmethod
    def simulate(self, t:ArrayLike,x0:ArrayLike,dW:Optional[ArrayLike]=None,batch:Optional[int]=None, seed:Optional[int]=None, **kwargs)->ArrayLike:
        dtype = t.dtype
        device=None if not isinstance(t, torch.Tensor) else t.device
        t = t.reshape(1,-1)
        
        if dW is not None:
            dW = dW.squeeze(-1)
        elif batch is not None:
            dW = BrownianMotion(t, batch, onlyIncrements=True, dim=1, dtype=dtype, device=device, seed=seed)
        else:
            pass
        
        if isinstance(t, np.ndarray):
            Xt = self._simulate_numpy(t, x0, dW, **kwargs)
        elif isinstance(t, torch.Tensor):
            Xt = self._simulate_torch(t, x0, dW, **kwargs)
        else:
            raise TypeError("Input must be a numpy array or a torch tensor.")

        return Xt

    # @abstractmethod
    def plot(self, t:ArrayLike,Xt:ArrayLike,wi:List[int]=[0], xlabel:str="Time", ylabel:str="State", title:str="Process Simulation", **kwargs):
        fig = plt.figure(figsize=(16,9))
        if isinstance(t, np.ndarray):
            plt.plot(t.ravel(), Xt[wi,...], **kwargs)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
        elif isinstance(t, torch.Tensor):
            plt.plot(t.cpu().numpy().ravel(), Xt[wi,...].cpu().numpy().T, **kwargs)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
        else:
            raise TypeError("Input must be a numpy array or a torch tensor.")
        return fig

class ProcessCtrl(Process):
    def __init__(self, 
                 params:Optional[Dict[str,Any]] = None, 
                 dtype:Optional[DTypeLike] = None):
        super().__init__(params, dtype)
        self.isControllable = True
        
    def drift(self, t, x, u, **kwargs):
        pass
    
    def diffusion(self, t, x, u, **kwargs):
        pass

    def _simulate_numpy(self, t:ArrayLike,x0:ArrayLike,U:Callable[[ArrayLike], ArrayLike],dW:Optional[ArrayLike]=None,returnControl:bool=False, **kwargs)->ArrayLike:
        N = np.prod(t.shape)
        t = t.reshape(1,-1)
        batch = dW.shape[0] if dW is not None else 1
        Xut = np.zeros((batch, N, 2), dtype=t.dtype)
        Xut[:,0,0] = x0
        if dW is not None:
            for i in range(N-1):
                u = U(np.concatenate([t[:,i].reshape(-1,1), Xut[:,i,0].reshape(-1,1)], axis=1)).reshape(-1,1)
                Xut[:,i+1,1] = u.ravel()
                Xut[:,i+1,0] = Xut[:,i,0] + self.drift(t[:,i], Xut[:,i,0], u, **kwargs) * (t[0,i+1]-t[0,i]) + self.diffusion(t[:,i], Xut[:,i,0], u, **kwargs) * dW[:,i]
        else:
            for i in range(N-1):
                u = U(np.concatenate([t[:,i].reshape(-1,1), Xut[:,i,0].reshape(-1,1)], axis=1)).reshape(-1,1)
                Xut[:,i+1,1] = u.ravel()
                Xut[:,i+1,0] = Xut[:,i,0] + self.drift(t[:,i], Xut[:,i,0], u, **kwargs) * (t[0,i+1]-t[0,i]) 
        Xut[:,-1,1] = u.ravel()
        if returnControl:
            return Xut
        else:
            return Xut[:,:,0]

    def _simulate_torch(self, t:ArrayLike,x0:ArrayLike,U:Callable[[ArrayLike], ArrayLike],dW:Optional[ArrayLike]=None,returnControl:bool=False, **kwargs)->ArrayLike:
        N = t.numel()
        t = t.view(1,-1)
        batch = dW.size(0) if dW is not None else 1
        Xut = torch.zeros((batch, N, 2), dtype=t.dtype, device=t.device)
        Xut[:,0,0] = x0
        if dW is not None:
            for i in range(N-1):
                u = U(torch.cat([t[:,i].view(-1,1), Xut[:,i,0].view(-1,1)], axis=1)).view(-1,1)
                Xut[:,i+1,1] = u.view(-1)
                Xut[:,i+1,0] = Xut[:,i,0] + self.drift(t[:,i], Xut[:,i,0], u, **kwargs) * (t[0,i+1]-t[0,i]) + self.diffusion(t[:,i], Xut[:,i,0], u, **kwargs) * dW[:,i]
        else:
            for i in range(N-1):
                u = U(torch.cat([t[:,i].view(-1,1), Xut[:,i,0].view(-1,1)], axis=1)).view(-1,1)
                Xut[:,i+1,1] = u.view(-1)
                Xut[:,i+1,0] = Xut[:,i,0] + self.drift(t[:,i], Xut[:,i,0], u, **kwargs) * (t[0,i+1]-t[0,i]) 
        Xut[:,-1,1] = u.view(-1)
        if returnControl:
            return Xut
        else:
            return Xut[:,:,0]

    def simulate(self, t:ArrayLike,x0:ArrayLike,U:Callable[[ArrayLike], ArrayLike],dW:Optional[ArrayLike]=None,batch:Optional[int]=None, seed:Optional[int]=None,returnControl:bool=False, **kwargs)->ArrayLike:
        dtype = t.dtype
        device=None if not isinstance(t, torch.Tensor) else t.device
        t = t.reshape(1,-1)
        
        if dW is not None:
            dW = dW.squeeze(-1)
        elif batch is not None:
            dW = BrownianMotion(t, batch, onlyIncrements=True, dim=1, dtype=dtype, device=device, seed=seed)
        else:
            pass
        
        if isinstance(t, np.ndarray):
            Xt = self._simulate_numpy(t, x0, U, dW, returnControl=returnControl, **kwargs)
        elif isinstance(t, torch.Tensor):
            Xt = self._simulate_torch(t, x0, U, dW, returnControl=returnControl, **kwargs)
        else:
            raise TypeError("Input must be a numpy array or a torch tensor.")

        return Xt

    def plot(self, t:ArrayLike,Xut:ArrayLike,wi:List[int]=[0], xlabel:str="Time", ylabel:str="State", ylabel_b:str="Control", title:str="Process Simulation", **kwargs):
        if len(Xut.shape) < 3 or (len(Xut.shape) == 3 and Xut.shape[-1] == 1):
            return super().plot(t, Xut, wi, xlabel, ylabel, title, **kwargs)
        else:
            Xt = Xut[...,0]
            ut = Xut[...,-1]
            fig = plt.figure(figsize=(16,9))
            ax = fig.add_subplot(111)
            ax_b = ax.twinx()
            if isinstance(t, np.ndarray):
                ax.plot(t.ravel(), Xt[wi,...], **kwargs)
                ax.plot(t.ravel(), ut[wi,...], 'r--', **kwargs)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax_b.set_ylabel(ylabel_b)
            elif isinstance(t, torch.Tensor):
                ax.plot(t.cpu().numpy().ravel(), Xt[wi,...].cpu().numpy().T, **kwargs)
                ax.plot(t.cpu().numpy().ravel(), ut[wi,...].cpu().numpy().T, 'r--', **kwargs)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax_b.set_ylabel(ylabel_b)
            else:
                raise TypeError("Input must be a numpy array or a torch tensor.")
            return fig

class LinearDeterm(Process):
    def __init__(self, 
                 eta:float,  
                 dtype:Optional[DTypeLike] = None):
        params = {r"\eta": eta}
        super().__init__(params, dtype)
        self.eta = eta

    def drift(self, t, x):
        return self.eta

    def diffusion(self, t, x):
        return 0.0
    
    def simulate(self, t, x0):
        return x0 + self.eta * t

class LogisticDeterm(Process):
    def __init__(self, 
                 alpha:Callable[[ArrayLike], ArrayLike], 
                 beta:Callable[[ArrayLike], ArrayLike], 
                 nu:Optional[float]=1.0,
                 dtype:Optional[DTypeLike] = None):
        params = {r"\alpha": alpha, r"\beta": beta, r"\nu": nu}
        super().__init__(params=params, dtype=dtype)
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        
    def drift(self, t, x):
        return self.alpha(t, x) * (1 - (x / self.beta(t, x))**self.nu) * x
    
    def diffusion(self, t, x):
        return 0.0
    
class LogisticDetermCtrl(LogisticDeterm,ProcessCtrl):
    def __init__(self, 
                 alpha:Callable[[ArrayLike], ArrayLike], 
                 beta:Callable[[ArrayLike], ArrayLike],
                 nu:Optional[float]=1.0,
                 dtype:Optional[DTypeLike] = None):
        super().__init__(alpha, beta, nu=nu, dtype=dtype)

    def drift(self, t, x, u):
        return self.alpha(t, x, u) * (1 - (x / self.beta(t, x , u))**self.nu) * x
    
    def diffusion(self, t, x, u):
        return 0.0
    
class ExpDeterm(Process):
    def __init__(self, 
                 mu:Callable[[ArrayLike], ArrayLike], 
                 dtype:Optional[DTypeLike] = None):
        params = {r"\mu": mu}
        super().__init__(params, dtype)
        self.mu = mu

    def drift(self, t, x):
        return self.mu(t, x) * x

    def diffusion(self, t, x):
        return 0.0
    
class ExpDetermCtrl(ExpDeterm,ProcessCtrl):
    def __init__(self, 
                 mu:Callable[[ArrayLike], ArrayLike], 
                 dtype:Optional[DTypeLike] = None):
        super().__init__(mu, dtype)
        
    def drift(self, t, x, u):
        return self.mu(t, x, u) * x

    def diffusion(self, t, x , u):
        return 0.0

class GBM(Process):
    def __init__(self, 
                 mu:Callable[[ArrayLike], ArrayLike], 
                 sigma:Callable[[ArrayLike], ArrayLike], 
                 dtype:Optional[DTypeLike] = None):
        params = {r"\mu": mu, r"\sigma": sigma}
        super().__init__(params, dtype)
        self.mu = mu
        self.sigma = sigma

    def drift(self, t, x):
        return self.mu(t, x) * x

    def diffusion(self, t, x):
        return self.sigma(t, x) * x
        