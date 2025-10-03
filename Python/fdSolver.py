# -*- coding: utf-8 -*-
# @Author: Kevin Kamm
# @Date:   2025-09-19 11:19:24
# @Last Modified by:   Kevin Kamm
# @Last Modified time: 2025-10-03 13:37:55
from typing import Tuple
from processes import *
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

def posPart(x):
    return torch.nn.functional.relu(x)

def negPart(x):
    return torch.nn.functional.relu(-x)

class Grids:
    def __init__(self, bnds:List[Tuple[float, float, int]], dtype:DTypeLike=torch.float32, device:torch.device=torch.device('cpu')):
        self.dtype = dtype
        self.grids = []
        self.dx = []
        for i in range(len(bnds)):
            lower, upper, N = bnds[i]
            grid = torch.linspace(lower, upper, N, dtype=dtype, device=device)
            self.grids.append(grid)
            self.dx.append(grid[1] - grid[0])
    
    def get_sizes(self):
        return [grid.size(0) for grid in self.grids]

    def get_steps(self):
        return [dx for dx in self.dx]
    
    def get_grid(self):
        shape = [1 for _ in self.grids]
        return [ grid.view( *shape[:i], -1, *shape[i+1:]) for i, grid in enumerate(self.grids) ]
    
    def get_grid_host(self, inner:bool=False):
        out = [self.grids[0].detach().cpu().numpy()]
        if inner:
            out += [grid[1:-1].detach().cpu().numpy() for grid in self.grids[1:]]
        else:
            out += [grid.detach().cpu().numpy() for grid in self.grids[1:]]
        return out

    def get_meshgrid(self):
        return torch.meshgrid(*[self.grids[i].view(-1) for i in range(len(self.grids)-1)], indexing='ij') # Assumes control grid is on last dimension
    
    def findIndices(self, x0:ArrayLike):
        return [torch.argmin(torch.abs(grid - x0[i])).item() for i, grid in enumerate(self.grids)]
    
    def estimateMemory(self):
        fpPrecision = self.grids[0].element_size()
        sizes = np.array(self.get_sizes())
        memHost = (fpPrecision * np.prod(sizes[:-1])) / 1000**3
        memDevice = (fpPrecision * np.prod(sizes[1:])) / 1000**3
        print(f"Memory (Host): {memHost:.2f} GB", f" Memory (Device): {memDevice:.2f} GB")
        return memHost, memDevice
    
class FDSolver:
    """
    Solve the HJB-VI equation with explicit finite difference method and upwind scheme for first derivatives.
    """

    def __init__(self,
                 r:float,
                 k:Callable[[torch.Tensor], torch.Tensor],
                 g:Callable[[torch.Tensor], torch.Tensor],
                 bnds:List[Tuple[float, float, int]],
                 processes:List[Process],
                 tIndSave:Optional[torch.Tensor]=None,
                 dtype=torch.float32, 
                 device=torch.device('cpu')):
        
        self.k=k
        self.g=g
        self.processes = processes

        self.G = Grids(bnds, dtype=dtype, device=device)
        self.grids = self.G.get_grid()
        self.t = self.grids[0]
        self.u = self.grids[-1]

        assert len(processes) == len(self.grids)-2, "Number of processes must match number of state dimensions"

        self.dx = self.G.get_steps()
        self.dt = self.dx[0]
        self.du = self.dx[-1]

        self.N = self.G.get_sizes()
        self.Nt = self.N[0]
        self.Nu = self.N[-1]
        
        self.dtype = self.t.dtype
        self.device = self.t.device

        self.r=r
        # self.discount = 1 + r * self.dt
        self.discount = torch.exp(self.r * self.dt)

        self.bf = []
        self.sf = []

        if tIndSave is None:
            self.tIndSave = torch.arange(0, self.Nt, 1, dtype=torch.int32, device=self.device)
        else:
            self.tIndSave = tIndSave
        self.Nt_save = self.tIndSave.size(0)

        for i in range(0, len(self.grids)-2):
            if self.processes[i].isControllable:
                self.bf.append(self.processes[i].drift(self.t, self.grids[i+1], self.u))
                self.sf.append(self.processes[i].diffusion(self.t, self.grids[i+1], self.u)**2)
            else:
                self.bf.append(self.processes[i].drift(self.t, self.grids[i+1]))
                self.sf.append(self.processes[i].diffusion(self.t, self.grids[i+1])**2)

        self.V_RI = None
        self.U_RI = None
        self.tau_RI = None

    def _compute_step(self,ti):
        bf = [bf[np.minimum(ti,bf.shape[0]-1),...] for bf in self.bf]
        sf = [sf if isinstance(sf, float) else sf[np.minimum(ti,sf.shape[0]-1),...] for sf in self.sf]

        # Upwind scheme for first derivatives
        prob_xx = 1.0
        prob_xu = []
        prob_xd = []
        for i in range(len(bf)):
            bi = bf[i] 
            si = sf[i] 
            di = self.dx[i+1] #because first grid is time grid
            pos_bi = posPart(bi)
            neg_bi = negPart(bi)
            prob_xx = prob_xx - si * self.dt/(di**2) - torch.abs(bi) * self.dt/di
            prob_xu.append( (si/(2*di**2) + pos_bi/di) * self.dt )
            prob_xd.append( (si/(2*di**2) + neg_bi/di) * self.dt )
            

        assert torch.all(prob_xx >= 0), "probability term should be non-negative: "+str(torch.min(prob_xx).item())

        stage_cost = self.k(self.grids[0][ti,...],*[grid[0,...] for grid in self.grids[1:]])
        exercise_value = self.g(self.grids[0][ti,...],*[grid[0,...] for grid in self.grids[1:-1]])

        return prob_xx, prob_xu, prob_xd, stage_cost, exercise_value

    def _indices(self):
        xi = [torch.arange(0, self.N[i], dtype=torch.long, device=self.device).reshape(self.grids[i].shape[1:-1]) for i in range(1,len(self.N)-1)] # Exclude time and control grid
        xi_u= [torch.cat((xi[i-1].view(-1)[1:], torch.tensor([self.N[i]-1], device=self.device))).reshape(self.grids[i].shape[1:-1]) for i in range(1,len(self.N)-1)] # Exclude time and control grid
        xi_d= [torch.cat((torch.tensor([0], device=self.device),xi[i-1].view(-1)[:-1])).reshape(self.grids[i].shape[1:-1]) for i in range(1,len(self.N)-1)] # Exclude time and control grid

        return xi, xi_u, xi_d

    def interpolate(self, F, method='linear'):
        """Interpolate the function"""
        F = F.detach().cpu().numpy()
        grids = self.G.get_grid_host(inner=True)
        t = grids[0]
        Ntcurr = F.shape[0]
        # if Ntcurr > 1:
        t = t[self.tIndSave.cpu().numpy()][:Ntcurr]
        curr = [t, *grids[1:-1]]
        return RegularGridInterpolator(tuple(curr), F, bounds_error=False, fill_value=None,method=method)
        # else:
        #     RI = RegularGridInterpolator(tuple(grids[1:-1]), F[0,...], bounds_error=False, fill_value=None,method=method)
        #     def tmpRI(Xt):
        #         # tt,ww,hh,pp,qq = inp
        #         return RI(tuple([Xt[:,i].detach().cpu().numpy() for i in range(Xt.shape[1])]))
        #     return lambda Xt: torch.tensor(tmpRI(Xt[:,1:]), dtype=self.dtype, device=self.device)
    
    def U(self,Xt):
        dtype = Xt.dtype
        device = Xt.device
        return torch.tensor(self.U_RI(tuple([Xt[:,i].detach().cpu().numpy() for i in range(Xt.shape[1])])), dtype=dtype, device=device)

    def V(self,Xt):
        dtype = Xt.dtype
        device = Xt.device
        return torch.tensor(self.V_RI(tuple([Xt[:,i].detach().cpu().numpy() for i in range(Xt.shape[1])])), dtype=dtype, device=device)

    def tau(self,Xt):
        dtype = Xt.dtype
        device = Xt.device
        return torch.tensor(self.tau_RI(tuple([Xt[:,i].detach().cpu().numpy() for i in range(Xt.shape[1])])), dtype=dtype, device=device)
    
    def solve(self,tau=False,stopping:bool=False): 
        if tau and not stopping:
            print("Warning: tau can only be computed if stopping=True. Setting tau=False.")
            tau=False
        xi, xi_u, xi_d = self._indices()
        Vshape = [self.Nt_save] + [self.N[i] for i in range(1,len(self.N)-1)] 
        Ushape = [self.Nt_save-1] + [self.N[i] for i in range(1,len(self.N)-1)] 
        Vtmpshape = [self.N[i] for i in range(1,len(self.N)-1)] + [1]
        Utmpshape = [self.N[i] for i in range(1,len(self.N)-1)]
        
        V = torch.zeros(Vshape, dtype=self.dtype, device='cpu')
        U = torch.zeros(Ushape, dtype=self.dtype, device='cpu')

        Vtmp = torch.zeros(Vtmpshape, dtype=self.dtype, device=self.device)
        Utmp = torch.zeros(Utmpshape, dtype=self.dtype, device=self.device)
        if stopping and tau:
            tau = torch.zeros(Vshape,dtype=self.dtype, device='cpu')
            tauTmp = torch.full(Utmpshape, self.t.view(-1)[-1], dtype=self.dtype, device=self.device)
        Vtmp[...,0] = self.g(self.t[-1, ...].unsqueeze(0), *self.grids[1:-1]).squeeze(0).squeeze(-1)
        
        tiSave = self.Nt_save - 1
        if self.Nt-1 in self.tIndSave:
            V[-1, ...] = Vtmp[...,0].detach().cpu()
            tiSave -= 1

        for ti in tqdm(reversed(range(self.Nt - 1)), desc="Backward time steps"):
            prob_xx, prob_xu, prob_xd, stage_cost, exercise_value = self._compute_step(ti)

            expected_V = prob_xx * Vtmp[tuple(i for i in xi)]
            for i in range(len(prob_xu)):
                expected_V += prob_xu[i] * Vtmp[tuple(xi_u[j] if j==i else xi[j] for j in range(len(xi)))]
                expected_V += prob_xd[i] * Vtmp[tuple(xi_d[j] if j==i else xi[j] for j in range(len(xi)))]

            total_profit = (self.dt * stage_cost + expected_V) / self.discount

            ui = torch.argmax(total_profit, dim=-1)
            
            if stopping:
                Vtmp[..., 0]  = torch.maximum(torch.squeeze(torch.take_along_dim(total_profit, ui[...,None], dim=-1)), exercise_value.squeeze(-1))
                if tau:
                    tauTmp[...]  = torch.where(torch.isclose(exercise_value.squeeze(-1),Vtmp[...,0]) , ti * self.dt, self.t.view(-1)[-1])
            else:
                Vtmp[..., 0] = torch.take_along_dim(total_profit, ui[..., None], dim=-1).squeeze(-1)

            Utmp[...] = self.u.view(-1)[ui]

            if ti == self.tIndSave[tiSave]:
                V[tiSave, ...] = Vtmp[...,0].detach().cpu()
                U[tiSave, ...] = Utmp.detach().cpu()
                if tau and stopping:
                    tau[tiSave, ...] = tauTmp.detach().cpu()
                tiSave -= 1
                
        # shtmp = [1 for _ in range(len(xi))]
        shttmp = [1 for _ in range(len(xi)+1)]
        # ind = tuple([x.view(-1)[1:-1].reshape(*shtmp[:i],-1,*shtmp[i+1:]) for i,x in enumerate(xi)])
        indVt = tuple([torch.arange(0,self.Nt_save,1,dtype=torch.long,device='cpu').reshape(-1,*shttmp[1:])] + [x.cpu().view(-1)[1:-1].reshape(*shttmp[:i+1],-1,*shttmp[i+2:]) for i,x in enumerate(xi)])
        indUt = tuple([torch.arange(0,self.Nt_save-1,1,dtype=torch.long,device='cpu').reshape(-1,*shttmp[1:])] + [x.cpu().view(-1)[1:-1].reshape(*shttmp[:i+1],-1,*shttmp[i+2:]) for i,x in enumerate(xi)])
        self.V_RI = self.interpolate(V[indVt])
        self.U_RI = self.interpolate(U[indUt])
        if stopping:
            if tau:
                self.tau_RI = self.interpolate(tau[indVt],method='nearest')
                return  self.V, self.U, self.tau #lambda x: self.V(x), lambda x: self.U(x), lambda x: self.tau(x)
            else:
                return self.V, self.U #lambda x: self.V(x), lambda x: self.U(x)
        else:
            return self.V, self.U # lambda x: self.V(x), lambda x: self.U(x)

    def simulate(self,t:torch.Tensor, x0:torch.Tensor, M:int, stopping:bool=False, tau:bool=False, includeCtrl:bool=False):
        if tau and not stopping:
            print("Warning: tau can only be computed if stopping=True. Setting tau=False.")
            tau=False
        if tau and self.tau is None:
            print("Warning: tau function not available. Setting tau=False.")
            tau=False
            
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
            tt = torch.full((M,Nt),t[-1],dtype=dtype,device=device)

        if includeCtrl:
            u = torch.zeros((M,Nt),dtype=dtype,device=device)
        else:
            u = torch.zeros((M,1),dtype=dtype,device=device)

        for i in range(0,Nt-1):
            inp = torch.cat([ t[i]*torch.ones((M,1),dtype=dtype,device=device), xt[:,i,:].squeeze() ], dim=1)
            # inp = tuple([(t[i]*torch.ones((M,),dtype=dtype,device=device)).detach().cpu().numpy()]+[xt[:,i,j].detach().cpu().numpy() for j in range(d)])
            ut = torch.tensor(self.U(inp).reshape(-1,1), dtype=dtype, device=device)


            if includeCtrl:
                u[:,i] = ut.reshape(-1) 

            if stopping:
                if tau:
                    tt[:,i] = torch.tensor(self.tau(inp).reshape(-1), dtype=dtype, device=device)
                else:
                    tt[:,i] = torch.where(self.V(inp).detach().cpu() <= self.g(*tuple([(t[i]*torch.ones((M,),dtype=dtype,device=device))]+[xt[:,i,j] for j in range(d)])), t[i], T) 


            for j in range(len(self.processes)):
                if self.processes[j].isControllable:
                    drift = self.processes[j].drift(t[i]*torch.ones((M,),dtype=dtype,device=device), xt[:,i,j], ut.reshape(-1))
                    diffusion = self.processes[j].diffusion(t[i]*torch.ones((M,),dtype=dtype,device=device), xt[:,i,j], ut.reshape(-1))
                else:
                    drift = self.processes[j].drift(t[i]*torch.ones((M,),dtype=dtype,device=device), xt[:,i,j])
                    diffusion = self.processes[j].diffusion(t[i]*torch.ones((M,),dtype=dtype,device=device), xt[:,i,j])
                xt[:,i+1,j] = xt[:,i,j] + drift * dt + diffusion * dWt[:,i,j]

            kt = self.k(t[i+1]*torch.ones((M,),dtype=dtype,device=device), *[xt[:,i+1,j] for j in range(d)], ut.reshape(-1))

            Vsim[:,i+1] = discount * kt * dt

        Vsim = torch.cumsum(Vsim, dim=1) + torch.exp(-r * t[None,:]) * self.g(t[None,:], *[xt[:,:,j] for j in range(d)])
        
        if includeCtrl:
            u[:,-1] = u[:,-2]
            out = torch.concatenate([xt,u[...,None],Vsim[...,None]], axis=2)
        else:
            out = torch.concatenate([xt,Vsim[...,None]], axis=2)

        if stopping:
            ttInd = torch.min(torch.abs(t-tt), dim=1, keepdim=False)[1]
            return out, t.view(-1)[ttInd.view(-1)], ttInd
        else:
            return out

