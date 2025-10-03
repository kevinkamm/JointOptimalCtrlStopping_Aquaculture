# -*- coding: utf-8 -*-
# @Author: Kevin Kamm
# @Date:   2025-09-20 08:58:37
# @Last Modified by:   Kevin Kamm
# @Last Modified time: 2025-09-29 13:02:21
from tqdm import tqdm
from processes import *
import torch
from typing import List, Tuple, Optional, Callable, Generator

class DGMCell(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, n_layers=3, output_dim=1):
        super(DGMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n = n_layers


class DGMCellLSTM(DGMCell): # Paper
    def __init__(self, input_dim, hidden_dim, n_layers=3, output_dim=1):
        super(DGMCellLSTM, self).__init__(input_dim, hidden_dim, n_layers, output_dim)

        self.act1 = torch.nn.Sigmoid()
        self.act2 = torch.nn.Tanh() # Paper says Linear for unbounded networks
        # self.act2 = lambda x:x# Paper says Linear for unbounded networks


        self.Sw = torch.nn.Linear(self.input_dim, self.hidden_dim)

        self.Uz = torch.nn.ModuleList([torch.nn.Linear(self.input_dim, self.hidden_dim,bias=False) for _ in range(n_layers)])
        self.Wsz = torch.nn.ModuleList([torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(n_layers)])

        self.Ug = torch.nn.ModuleList([torch.nn.Linear(self.input_dim, self.hidden_dim,bias=False) for _ in range(n_layers)])
        self.Wsg = torch.nn.ModuleList([torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(n_layers)])

        self.Ur = torch.nn.ModuleList([torch.nn.Linear(self.input_dim, self.hidden_dim,bias=False) for _ in range(n_layers)])
        self.Wsr = torch.nn.ModuleList([torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(n_layers)])

        self.Uh = torch.nn.ModuleList([torch.nn.Linear(self.input_dim, self.hidden_dim,bias=False) for _ in range(n_layers)])
        self.Wsh = torch.nn.ModuleList([torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(n_layers)])

        self.Wf = torch.nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        S = self.Sw(x)
        for i in range(self.n):
            S = self.act1(S)
            Z = self.act1(self.Uz[i](x) + self.Wsz[i](S))
            G = self.act1(self.Ug[i](x) + self.Wsg[i](S))
            R = self.act1(self.Ur[i](x) + self.Wsr[i](S))
            H = self.act2(self.Uh[i](x) + self.Wsh[i](S*R))
            S = (1-G)*H + Z*S
        out = self.Wf(S)
        return out

class DGMCellFF(DGMCell): # Feedforward
    def __init__(self, input_dim, hidden_dim, n_layers=3, output_dim=1):
        super(DGMCellFF, self).__init__(input_dim, hidden_dim, n_layers, output_dim)

        self.act = torch.nn.Tanh()
        # self.act = torch.nn.ReLU()
        # self.act = torch.nn.Sigmoid()

        self.fin = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fhidden = torch.nn.ModuleList([torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(n_layers)])
        self.fout = torch.nn.Linear(self.hidden_dim, self.output_dim)


    def forward(self, x):
        x=self.act(self.fin(x))
        for i in range(self.n):
            x=self.act(self.fhidden[i](x))
        x=self.fout(x)
        return x
    
class DGMCellFFg(DGMCell): # Feedforward
    def __init__(self,g, input_dim, hidden_dim, n_layers=3, output_dim=1):
        super(DGMCellFFg, self).__init__(input_dim, hidden_dim, n_layers, output_dim)

        self.act = torch.nn.Tanh()
        # self.act = torch.nn.ReLU()
        # self.act = torch.nn.Sigmoid()
        self.g = g
        self.fin = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fhidden = torch.nn.ModuleList([torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(n_layers)])
        self.fout = torch.nn.Linear(self.hidden_dim, self.output_dim)


    def forward(self, x):
        y=self.act(self.fin(x))
        for i in range(self.n):
            y=self.act(self.fhidden[i](y))
        y=self.fout(y)
        return self.g(x)+y
    
class ControlProcess(ABC):
    def __init__(self, params):
        self.params = params
        
    def trains(self,*args, **kwargs):
        pass

    @abstractmethod
    def eval(self,X,dV):
        pass

class ControlProcessNN(torch.nn.Module, ControlProcess):
    def __init__(self, params, input_dim=1, hidden_dim=32, n_layers=3, output_dim=1, lr=1e-2):
        super(ControlProcessNN, self).__init__()
        self.params = params
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n = n_layers
        # self.act = torch.nn.Tanh()
        self.act = torch.nn.ReLU()
        self.lr = lr
        # self.act = torch.nn.Sigmoid()

        self.fin = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fhidden = torch.nn.ModuleList([torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(n_layers)])
        self.fout = torch.nn.Linear(self.hidden_dim, self.output_dim)


    def forward(self, x):
        x=self.act(self.fin(x))
        for i in range(self.n):
            x=self.act(self.fhidden[i](x))
        x=self.fout(x)
        return torch.abs(x) # assuming control is in [0,1]
    
    def trains(self,X,dV,PDE, epochs=100, lr=1e-2):
        # self.train(True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr,betas=(0.9, 0.999), eps=1e-5)
        # x = torch.cat([X,dV],dim=1)
        for epoch in range(epochs):
            optimizer.zero_grad()
            u_hat = self(X).view(-1,1)
            y = PDE(X,dV,u_hat).view(-1,1)
            loss = (-y).mean()
            loss.backward()
            optimizer.step()
            # pbar.set_postfix({'loss': loss.item()})

    def eval(self,X,dV):
        # self.train(False)
        with torch.no_grad():
            # x = torch.cat([X,dV],dim=1)
            u = self.forward(X).view(-1,1)
        return u

class ControlProcessNN2(torch.nn.Module, ControlProcess):
    def __init__(self, params, input_dim=1, hidden_dim=32, n_layers=3, output_dim=1, lr=1e-2):
        super(ControlProcessNN2, self).__init__()
        self.params = params
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n = n_layers
        # self.act = torch.nn.Tanh()
        self.act = torch.nn.ReLU()
        self.lr = lr
        # self.act = torch.nn.Sigmoid()

        self.fin = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fhidden = torch.nn.ModuleList([torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(n_layers)])
        self.fout = torch.nn.Linear(self.hidden_dim, self.output_dim)


    def forward(self, x):
        x=self.act(self.fin(x))
        for i in range(self.n):
            x=self.act(self.fhidden[i](x))
        x=self.fout(x)
        return torch.abs(x) # assuming control is in [0,1]
    
    def trains(self,X,dV,PDE, epochs=100, lr=1e-2):
        # self.train(True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr,betas=(0.9, 0.999), eps=1e-5)
        # x = torch.cat([X,dV],dim=1)
        u = torch.linspace(0,1,steps=100,dtype=X.dtype,device=X.device).view(1,1,-1)
        ytmp = PDE(X[...,None],dV[...,None],u)
        u_opt = torch.argmax(ytmp,dim=2).view(-1)
        y_opt = torch.gather(ytmp,2,u_opt.view(-1,1,1)).view(-1,1)
        for epoch in range(epochs):
            optimizer.zero_grad()
            u_hat = self(X).view(-1,1)
            y = PDE(X,dV,u_hat).view(-1,1)
            loss = torch.nn.functional.relu(-y.view(-1,1)  + y_opt.view(-1,1) ).mean()
            loss.backward()
            optimizer.step()
            # pbar.set_postfix({'loss': loss.item()})

    def eval(self,X,dV):
        # self.train(False)
        with torch.no_grad():
            # x = torch.cat([X,dV],dim=1)
            u = self.forward(X).view(-1,1)
        return u

class ControlProcessAquaculture(ControlProcess):
    def __init__(self, params, feedingCurve:Callable[[torch.Tensor],torch.Tensor]):
        super(ControlProcessAquaculture, self).__init__(params)
        self.f0 = params['f0']
        self.T = params['T']
        self.w_inf = params['w_inf']
        self.gamma_F = params['gamma_F']
        self.mu_F = params['mu_F']
        self.nu = params.get('nu',1.0)
        self.feedingCurve = feedingCurve

    def eval(self,X,dV):
        dVdw = dV[:,1]
        dVdh = dV[:,2]
        t = X[:,0]
        w = X[:,1]
        h = X[:,2]
        pf = X[:,3]
        ft = self.feedingCurve(t)
        # tmp = ft - (h * pf * self.w_inf)/(2*dVdw * w * (self.w_inf-w) *self.gamma_F + 2 * h * dVdh * self.w_inf * self.mu_F)
        tmp = ft - (h * pf )/(-2*dVdw * w * ((w/self.w_inf)**self.nu-1) *self.gamma_F + 2 * h * dVdh * self.mu_F)
        return torch.clamp(tmp.view(-1), min=torch.zeros_like(ft).view(-1), max=ft.view(-1))

class DeepSolver():
    def __init__(self, r, k, g, bnds, processes, dgmCell:DGMCell, u:ControlProcess, lr=1e-2, dtype=torch.float32, device=torch.device('cpu')):
        self.r = r
        self.bnds = bnds
        self.processes = processes
        self.k = k
        self.g = g
        self.u = u
        self.dtype = dtype
        self.device = device

        self.input_dim = len(bnds)
        self.output_dim = 1 

        # self.net = dgmCell(self.input_dim, hidden_dim, n_layers, self.output_dim).to(device).to(dtype)
        self.net = dgmCell.to(device).to(dtype)
        # self.net = DGMCellFF(self.input_dim, hidden_dim, n_layers, self.output_dim).to(device).to(dtype)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr,betas=(0.9, 0.999), eps=1e-5)

        # self.loss_fn = torch.nn.MSELoss()
        self.pde_losses = []
        self.ivp_losses = []
        self.bvp_losses = []
        self.losses = []

    def V(self,X):
        self.net.train(False)
        with torch.no_grad():
            return self.net(X).view(-1,1)
        
    def U(self,X):
        self.net.train(False)
        X.requires_grad_(True)
        V = self.net(X)
        dV = torch.autograd.grad(V, X, grad_outputs=torch.ones_like(V),retain_graph=True)[0]
        ut = self.u.eval(X,dV).detach().view(-1,1)
        dV=None
        X.requires_grad_(False)
        return ut
    
    def get_diff_data(self,n):
        x = [self.bnds[i][0]+(self.bnds[i][1]-self.bnds[i][0])*torch.rand((n, 1),dtype=self.dtype,device=self.device) for i in range(len(self.bnds))]
        X = torch.concatenate(x,axis=1)
        y= self.g(*x).reshape(-1, 1)
        return X, y#, x

    def get_tvp_data(self,n):
        x = [torch.full((n, 1),self.bnds[0][1],dtype=self.dtype,device=self.device)]+\
            [self.bnds[i][0]+(self.bnds[i][1]-self.bnds[i][0])*torch.rand((n, 1),dtype=self.dtype,device=self.device) for i in range(1,len(self.bnds))]
        X = torch.concatenate(x,axis=1)
        y= self.g(*x).reshape(-1, 1)
        return X, y#, x
    

    def PDE(self,X,V,dV,d2V,u1):
        x1u = [X[:,i].view(-1,1) for i in range(self.input_dim)]
        kf = self.k(*x1u,u1)
        pde = dV[:,0].view(-1,1) - self.r * V + kf
        for i in range(len(self.processes)):
            if self.processes[i].isControllable:
                pde = pde + self.processes[i].drift(X[:,0].view(-1,1),X[:,i+1].view(-1,1),u1) * dV[:,i+1].view(-1,1) +\
                        0.5 * self.processes[i].diffusion(X[:,0].view(-1,1),X[:,i+1].view(-1,1),u1)**2 * d2V[:,i+1].view(-1,1)
            else:
                pde = pde + self.processes[i].drift(X[:,0].view(-1,1),X[:,i+1].view(-1,1)) * dV[:,i+1].view(-1,1) +\
                        0.5 * self.processes[i].diffusion(X[:,0].view(-1,1),X[:,i+1].view(-1,1))**2 * d2V[:,i+1].view(-1,1)
        return pde
    
    def CrtlDrift(self,X,dV,u1):
        shape = [1]*len(u1.shape)
        shape[0] = -1
        x1u = [X[:,i,...].view(shape) for i in range(self.input_dim)]
        kf = self.k(*x1u,u1)
        pde = kf
        for i in range(len(self.processes)):
            if self.processes[i].isControllable:
                pde = pde + self.processes[i].drift(X[:,0,...].view(shape),X[:,i+1,...].view(shape),u1) * dV[:,i+1,...].view(shape)
        return pde


    def solve(self, epochs=10000, batch_size=4096, stopping=True, balance:bool=True, fuzzy:float=1e-2):
        self.net.train(True)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,epochs//10+1,gamma=0.5)
        for epoch in (pbar:=tqdm(range(epochs))):
            
            self.optimizer.zero_grad()
                
            if balance:
                n=0
                X1u = torch.tensor([],dtype=self.dtype,device=self.device).view(-1,self.input_dim)
                y1u = torch.tensor([],dtype=self.dtype,device=self.device).view(-1,1)
                numu = 0
                X1l = torch.tensor([],dtype=self.dtype,device=self.device).view(-1,self.input_dim)
                y1l = torch.tensor([],dtype=self.dtype,device=self.device).view(-1,1)
                numl = 0
                while n < 10 and numu < batch_size:
                    Xb, yb = self.get_diff_data(batch_size)
                    ind1 = self.net(Xb).detach() > yb-fuzzy * torch.rand_like(yb) # a bit fuzzier
                    if ind1.sum() > 0:
                        X1u = torch.concat([X1u, Xb[ind1.view(-1),:]],dim=0)
                        y1u = torch.concat([y1u, yb[ind1.view(-1),:]],dim=0)
                        numu += ind1.sum().item()
                    n+=1
                if X1u.size(0) > batch_size:
                    X1u = X1u[:batch_size,:]
                    y1u = y1u[:batch_size,:]
                n=0
                while n < 30 and numl < batch_size:
                    Xb, yb = self.get_diff_data(batch_size)
                    ind1 = self.net(Xb).detach() <= yb+fuzzy * torch.rand_like(yb) # a bit fuzzier
                    if ind1.sum() > 0:
                        X1l = torch.concat([X1l, Xb[ind1.view(-1),:]],dim=0)
                        y1l = torch.concat([y1l, yb[ind1.view(-1),:]],dim=0)
                        numl += ind1.sum().item()
                    n+=1
                if X1l.size(0) > batch_size:
                    X1l = X1l[:batch_size,:]
                    y1l = y1l[:batch_size,:]
            else:
                X1u, y1u = self.get_diff_data(batch_size)
                X1l = X1u
                y1l = y1u
            X1u.requires_grad_(True)
            y1_hat = self.net(X1u)

            # Derivatives

            dV = torch.autograd.grad(y1_hat, X1u, grad_outputs=torch.ones_like(y1_hat),retain_graph=True, create_graph=True)[0]
            d2V = torch.autograd.grad(dV,X1u,grad_outputs=torch.ones_like(dV),retain_graph=True)[0]
            X1u.requires_grad_(False)

            obj = lambda X1u,dV,u1: self.CrtlDrift(X1u.detach(),dV.detach(),u1)

            self.u.trains(X1u.detach(),dV.detach(),obj, epochs=5,lr=5e-4)
            u1 = self.u.eval(X1u.detach(),dV.detach()).view(-1,1).detach()
            pde = self.PDE(X1u,y1_hat,dV,d2V,u1)

            pde_loss = (pde**2).mean()

            # Free Boundary Condition (Soft)
            if stopping:
                y1_hat = self.net(X1l)
                y_freeBD = y1l
                bvp_loss = (torch.relu(-y1_hat + y_freeBD)**2).mean()
            else:
                bvp_loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)


            # Terminal Value
            X3, y3 = self.get_tvp_data(batch_size)
            y3_hat = self.net(X3)
            tvp_loss = (torch.abs(y3- y3_hat)**2).mean()

            # Backpropagation and Update
            combined_loss = pde_loss + tvp_loss + bvp_loss 
            combined_loss.backward()
            self.optimizer.step()
            scheduler.step()

            self.pde_losses.append(pde_loss.item())
            self.bvp_losses.append(bvp_loss.item())
            self.ivp_losses.append(tvp_loss.item())
            self.losses.append(combined_loss.item())

            # pbar.set_postfix({'PDE Loss': self.pde_losses[-1],'Free BD Loss': self.bvp_losses[-1],'TVP Loss': self.ivp_losses[-1], 'Ctrl':u1.mean().item(), 'LR': scheduler.get_last_lr()})
            pbar.set_postfix({'PDE Loss': self.pde_losses[-1],'Free BD Loss': self.bvp_losses[-1],'TVP Loss': self.ivp_losses[-1], 'y<=fB':y1l.numel()/(2*batch_size), 'Ctrl':u1.mean().item(),'LR': scheduler.get_last_lr()})
        self.net.train(False)   
        return self.V, self.U

    def simulate(self,t:torch.Tensor, x0:torch.Tensor, M:int, stopping:bool=False, includeCtrl:bool=False,fuzzy:float=0.0):
        device = t.device
        dtype = t.dtype
        Nt = t.size(0)
        dt = (t[1]-t[0])
        d = x0.numel()
        dWt = BrownianMotion(t, M, onlyIncrements=True, dim=d)

        
        xt = x0.reshape((1,1,d)) * torch.ones((M,Nt,d),dtype=dtype,device=device)

        discount = torch.exp(-self.r * dt)

        Vsim = torch.zeros((M,Nt),dtype=dtype,device=device)
        if stopping:
            tt = t[-1]*torch.ones((M,Nt),dtype=dtype,device=device)

        if includeCtrl:
            u = torch.zeros((M,Nt),dtype=dtype,device=device)

        for i in range(0,Nt-1):
            X =torch.concatenate([t[i]*torch.ones((M,1),dtype=dtype,device=device),xt[:,i,:].view(-1,d)],axis=1)
            X.requires_grad_(True)
            V = self.net(X)
            dV = torch.autograd.grad(V, X, grad_outputs=torch.ones(V.shape,dtype=dtype,device=device),retain_graph=True)[0]
            ut = self.u.eval(X,dV).detach().view(-1,1)
            dV=None
            X.requires_grad_(False)

            if includeCtrl:
                u[:,i] = ut.view(-1)

            if stopping:
                tt[:,i] = torch.where(V.view(-1)-fuzzy <= self.g(t[i]*torch.ones((M,),dtype=dtype,device=device), *[xt[:,i,j] for j in range(d)]).view(-1), t[i], T) # Does not work well


            for j in range(len(self.processes)):
                if self.processes[j].isControllable:
                    drift = self.processes[j].drift(t[i]*torch.ones((M,),dtype=dtype,device=device), xt[:,i,j], ut.view(-1))
                    diffusion = self.processes[j].diffusion(t[i]*torch.ones((M,),dtype=dtype,device=device), xt[:,i,j], ut.view(-1))
                else:
                    drift = self.processes[j].drift(t[i]*torch.ones((M,),dtype=dtype,device=device), xt[:,i,j])
                    diffusion = self.processes[j].diffusion(t[i]*torch.ones((M,),dtype=dtype,device=device), xt[:,i,j])
                xt[:,i+1,j] = xt[:,i,j] + drift * dt + diffusion * dWt[:,i,j]

            kt = self.k(t[i+1]*torch.ones((M,),dtype=dtype,device=device), *[xt[:,i+1,j] for j in range(d)], ut.view(-1))

            Vsim[:,i+1] = discount * kt * dt

        Vsim = torch.cumsum(Vsim, dim=1) + torch.exp(-r * t[None,:]) * self.g(t[None,:], *[xt[:,:,j] for j in range(d)])
        
        if includeCtrl:
            u[:,-1] = u[:,-2]
            out = torch.concatenate([xt,u[...,None],Vsim[...,None]], axis=2)
        else:
            out = torch.concatenate([xt,Vsim[...,None]], axis=2)

        if stopping:
            # tt = torch.min(tt, dim=1, keepdim=False)[0]
            ttInd = torch.min(torch.abs(t-tt), dim=1, keepdim=False)[1]
            return out[:,ttInd], t.view(-1)[ttInd]
        else:
            return out
        

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    from aquaculture import *
    from fdSolver import *
    from deepOS import *
    dtype = torch.float32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed = 0
    torch.manual_seed(seed)
    T = 3.0 # time horizon
    r = 0.01 # interest rate

    # Feeding Curve
    f0 = 0.1
    F = LinearFeeding(f0,T)
    ft = F(np.linspace(0, T, 100))

    # Host parameters
    mu = 0.1
    mu_F = 1.0
    h0 = 1

    hModel = HostConstant(mu, mu_F, F)

    # Weight parameters
    gamma = 5.0
    gamma_F = 8.0
    w_inf = 3.0
    w0 = 0.01 

    wModel = GrowthConst(gamma, gamma_F, w_inf, F)
    
    # Feed price parameters
    sigmaF = 0.25
    PF0 = 0.075

    pFModel = FeedPriceGBM(r, sigmaF)
    # Biomass price parameters
    sigmaB = 0.1
    PB0 = 0.1

    pBModel = BiomassPriceGBM(r, sigmaB)

    # Grids
    N=64
    Nt=2048
    M = 1024*8

    pFt = pFModel.simulate(np.linspace(0,T,Nt), PF0, batch=M)
    pBt = pBModel.simulate(np.linspace(0,T,Nt), PB0, batch=M)

    bnds = [(0.0, T, Nt), # time
            (0.5*w0, 1.1*w_inf, N), # weight
            (0.1*h0, 1.1*h0, N), # host
            (0.1*pFt.min(), 1.5*pFt.max(), N//2), # feed price
            (0.1*pBt.min(), 1.5*pBt.max(), N//2), # biomass price
            (0.0, ft.max(), N//2)] # control
    del pFt, pBt
    k = lambda t, w, h, pf, pb, u: - h * u * pf
    g = lambda t, w, h, pf, pb: w * h * pb


    u = ControlProcessAquaculture({'f0':f0,'T':T,'w_inf':w_inf,'gamma_F':gamma_F,'mu_F':mu_F}, F)
    # u = ControlProcessNN({'f0':f0,'T':T,'w_inf':w_inf,'gamma_F':gamma_F,'mu_F':mu_F}, input_dim=len(bnds[:-1]), hidden_dim=32, n_layers=3, output_dim=1, lr=1e-2).to(device).to(dtype)
    u = ControlProcessNN2({}, input_dim=len(bnds[:-1]), hidden_dim=32, n_layers=3, output_dim=1, lr=1e-2).to(device).to(dtype)
    
    def gNN(X):
        # t = X[:,0]
        w = X[:,1].view(-1,1)
        h = X[:,2].view(-1,1)
        # pf = X[:,3]
        pb = X[:,4].view(-1,1)
        return w * h * pb
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # dgmCell = DGMCellFFg(gNN,input_dim=len(bnds[:-1]), hidden_dim=32, n_layers=3, output_dim=1)
    dgmCell = DGMCellFF(input_dim=len(bnds[:-1]), hidden_dim=32, n_layers=3, output_dim=1)
    # dgmCell = DGMCellLSTM(input_dim=len(bnds[:-1]), hidden_dim=64, n_layers=3, output_dim=1)
    opt = DeepSolver(r,k,g, bnds[:-1],[wModel, hModel, pFModel, pBModel], dgmCell,u,lr=5e-3,dtype=dtype,device=device)
    opt.solve(epochs=15000,batch_size=4096,stopping=True)
    # V, U, tau = opt.solve(epochs=10000,batch_size=4096,stopping=True)
    Vopt0 = opt.net(torch.tensor([0,w0,h0,PF0,PB0],dtype=dtype,device=device).view(1,-1)).item()
    print("Opt Crtl gives value ",Vopt0)
    
    out,tau = opt.simulate(torch.linspace(0,T,Nt,dtype=dtype,device=device),torch.tensor([w0,h0,PF0,PB0],dtype=dtype,device=device),M=8*1024,stopping=True,includeCtrl=True,fuzzy=1e-1)
    print("Opt Crtl Sim gives value ",out[:,-1,-1].mean().item(), " with stopping time ",tau.mean().item(), " and control ",out[:,:, -2].mean().item())
    out,tau = opt.simulate(torch.linspace(0,T,Nt,dtype=dtype,device=device),torch.tensor([w0,h0,PF0,PB0],dtype=dtype,device=device),M=8*1024,stopping=True,includeCtrl=True,fuzzy=5e-2)
    print("Opt Crtl Sim gives value ",out[:,-1,-1].mean().item(), " with stopping time ",tau.mean().item(), " and control ",out[:,:, -2].mean().item())
    out,tau = opt.simulate(torch.linspace(0,T,Nt,dtype=dtype,device=device),torch.tensor([w0,h0,PF0,PB0],dtype=dtype,device=device),M=8*1024,stopping=True,includeCtrl=True,fuzzy=1e-2)
    print("Opt Crtl Sim gives value ",out[:,-1,-1].mean().item(), " with stopping time ",tau.mean().item(), " and control ",out[:,:, -2].mean().item())
    out,tau = opt.simulate(torch.linspace(0,T,Nt,dtype=dtype,device=device),torch.tensor([w0,h0,PF0,PB0],dtype=dtype,device=device),M=8*1024,stopping=True,includeCtrl=True,fuzzy=0.0)
    print("Opt Crtl Sim gives value ",out[:,-1,-1].mean().item(), " with stopping time ",tau.mean().item(), " and control ",out[:,:, -2].mean().item())

    Nos:int = int(T*128) # number of time steps, don't use too many, it will slow down the method a lot
    t:torch.Tensor = torch.linspace(0.0,T,Nos,dtype=dtype,device=device)

    def modelGen(M,batch_factor=64): # You can change this and still use the same DeepOS network!
        while True:
            data = opt.simulate(t, torch.tensor([w0,h0,PF0,PB0],dtype=dtype,device=device),M * batch_factor, includeCtrl=True).detach()
            for i in range(batch_factor):
                yield data[i*M:(i+1)*M,...]

    d:int = 5 # Dimension of process
    latent_dim: List[int] = [d+50,d+50] # latent dimensions of neural network
    dOS = DeepOSNet(d,Nos,latent_dim=latent_dim,outputDims=1).to(device)
    # print(dOS.deepOSNet)
    trainErr, trainPrice=dOS.train_loop(modelGen)

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    # with plt.style.context('dark_background'):
    #     fig, ax= plt.subplots(figsize=(1600*px, 900*px))
    #     ax.plot(trainErr,label='Training Error')
    #     ax.plot(trainPrice,label='Option Price')
    #     ax.legend(fancybox=True, framealpha=0.0)
    #     # ax.set_yscale('log')
    #     plt.show()

    #########################################################################
    ### Testing
    #########################################################################
    test = next(modelGen(1024*32,batch_factor=1))
    tau,payoff=dOS.evalStopping(t,test)
    print(f'Mean stopping time {tau.mean().item()} with price {payoff.mean().item()}') # reference 0.2047