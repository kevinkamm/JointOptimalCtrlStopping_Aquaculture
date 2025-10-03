# -*- coding: utf-8 -*-
# @Author: Kevin Kamm
# @Date:   2025-09-19 10:15:53
# @Last Modified by:   Kevin Kamm
# @Last Modified time: 2025-10-03 11:11:08
from processes import *
import pandas as pd

class Feeding(ABC):
    def __init__(self, params:Optional[Dict[str,Any]]=None, dtype:Optional[DTypeLike]=None):
        self.params = params
        self.dtype = dtype
    
    @abstractmethod
    def __call__(self, t, **kwargs):
        pass 
    
class LinearFeeding(Feeding):
    def __init__(self, f0:float,T:float, dtype:Optional[DTypeLike]=None):
        params = {"f_0": f0, "T": T}
        super().__init__(params, dtype)
        self.Ft = LinearDeterm( (1-f0)/T)
        params.update(self.Ft.params)
        self.params = params
    
    def __call__(self, t, **kwargs):
        return self.Ft.simulate(t, self.params["f_0"])
    
class ExpFeeding(Feeding):
    def __init__(self, f0:float,T:float, dtype:Optional[DTypeLike]=None):
        self.lam = np.log(1/f0)/T
        params = {"f_0": f0, "\lambda": self.lam, "T": T}
        super().__init__(params, dtype)
    
    def __call__(self, t, **kwargs):
        return self.params["f_0"] * (np.exp(self.lam*t) if isinstance(t, (float, int, np.ndarray)) else torch.exp(self.lam*t))
    
class LogisticFeeding(Feeding):
    def __init__(self, f0:float, k:float,tI:float, T:float, L:float=1, dtype:Optional[DTypeLike]=None):
        params = {"f_0": f0, "k": k, "t_I": tI, "T": T, "L": L}
        self.f0=f0
        self.k=k
        self.tI=tI
        self.L=L
        self.T=T    
        super().__init__(params, dtype)
        
    
    def __call__(self, t, **kwargs):
        exp = np.exp(-self.k*(t-self.tI)) if isinstance(t, (float, int, np.ndarray)) else torch.exp(-self.k*(t-self.tI))
        return self.f0 + (self.L-self.f0)/(1+exp)
    
class SinusoidalFeeding(Feeding):
    def __init__(self, f0:float, a:float,tp:float, T:float, b:Optional[float]=None,dtype:Optional[DTypeLike]=None):
        if b is None:
            b = (1 - a - f0) / T
        params = {"f_0": f0, "a": a, "b": b, "tp": tp, "T": T}

        self.f0 = f0
        self.a = a
        self.b = b
        self.tp = tp
        self.T = T
        super().__init__(params, dtype)
        
    
    def __call__(self, t, **kwargs):
        sin = np.sin(2 * np.pi * t / self.tp) if isinstance(t, (float, int, np.ndarray)) else torch.sin(2 * torch.pi * t / self.tp)
        return self.f0 + self.a * sin + self.b * t

class HostConstant(ExpDetermCtrl):
    def __init__(self, mu_0, mu_F,F:Feeding, dtype:Optional[DTypeLike]=None):
        def mu(t, x, u):
            return (-mu_0 - mu_F * (F(t) - u)**2) 

        super().__init__(mu, dtype)
        self.params = {r"\mu_0": mu_0, r"\mu_F": mu_F, "F": str(F.__class__.__name__)}
    
class GrowthConst(LogisticDetermCtrl):
    def __init__(self, alpha_0, alpha_F, w_Inf, F:Feeding, dtype:Optional[DTypeLike]=None,nu:Optional[float]=1.0):
        def alpha(t, x, u):
            return (alpha_0 - alpha_F * (F(t) - u)**2) 

        def beta(t, x, u):
            return w_Inf

        super().__init__(alpha, beta, nu=nu, dtype=dtype)
        self.params = {r"\gamma_0": alpha_0, r"\gamma_F": alpha_F, r"w_{\infty}": w_Inf, "F": str(F.__class__.__name__), r"\nu": nu}

class FeedPriceGBM(GBM):
    def __init__(self, r ,sigmaF, dtype:Optional[DTypeLike]=None):
        params = {"r": r, r"\sigma^{p^F}": sigmaF}
        mu = lambda t, x: r
        sigma = lambda t, x: sigmaF
        super().__init__(mu, sigma, dtype)
        self.params = params

class BiomassPriceGBM(GBM):
    def __init__(self, r, sigmaB, dtype:Optional[DTypeLike]=None):
        params = {"r": r, r"\sigma^{p^B}": sigmaB}
        mu = lambda t, x: r
        sigma = lambda t, x: sigmaB
        super().__init__(mu, sigma, dtype)
        self.params = params

def params_to_df(models:List[Union[Process,Feeding]],initialDatum:Optional[Dict[str,float]]=None, float_format:str=".6f") -> pd.DataFrame:
    all_params = {}
    if initialDatum:
        for k, v in initialDatum.items():
            key = r"$" + r"{0:s}".format(k) + r"$"
            all_params[key] = r"$" + f"{v:{float_format}}" + r"$"
    for model in models:
        for k, v in model.params.items():
            # Wrap key and value in display math
            key = r"$" + r"{0:s}".format(k) + r"$"
            if isinstance(v, float):
                value = r"$" + f"{v:{float_format}}" + r"$"
            else:
                value =  str(v) 
            all_params[key] = value
    return pd.DataFrame([all_params])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    T = 3.0 # time horizon
    r = 0.01 # interest rate

    # Feeding Curve
    f0 = 0.1
    F = LinearFeeding(f0,T)

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

    pFModel = FeedPriceGBM(r, sigmaF, PF0)

    # Biomass price parameters
    sigmaB = 0.1
    PB0 = 0.1

    pBModel = BiomassPriceGBM(r, sigmaB, PB0)

    # t = np.linspace(0, T, 100)
    t = torch.linspace(0, T, 100, dtype=torch.float32, device='cuda')
    ft = F(t)
    print(hModel.drift(0, h0, f0))
    print(hModel.diffusion(0, h0, f0))
    print(wModel.drift(0, w0, f0))
    print(wModel.diffusion(0, w0, f0))
    # plt.plot(t, ft, 'k-', label="Feeding Level")
    plt.plot(t.detach().cpu().ravel(), pFModel.simulate(t, PF0).detach().cpu().ravel(), 'b-', label="Feed Price")
    plt.plot(t.detach().cpu().ravel(), pBModel.simulate(t, PB0).detach().cpu().ravel(), 'r-', label="Biomass Price")
    plt.xlabel("Time")
    plt.ylabel("Feeding Level")
    plt.title("Feeding Level Over Time")
    plt.legend()
    plt.show()
