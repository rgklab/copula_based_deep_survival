import torch
from Nets import *
import numpy as np 
#from sksurv.linear_model.coxph import BreslowEstimator
from os import path
import os



def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

class Weibull_log_linear:
    def __init__(self, nf, mu, sigma, device) -> None:
        #torch.manual_seed(0)
        self.nf = nf
        self.mu = torch.tensor([mu], device=device).type(torch.float32)
        self.sigma = torch.tensor([sigma], device=device).type(torch.float32)
        self.coeff = torch.rand((nf,), device=device)
    
    def survival(self,t,x):
        return torch.exp(-1*torch.exp((LOG(t)-self.mu-torch.matmul(x, self.coeff))/torch.exp(self.sigma)))
    
    def cum_hazard(self, t,x):
        return torch.exp((LOG(t)-self.mu-torch.matmul(x, self.coeff))/torch.exp(self.sigma))
    
    def hazard(self, t,x):
        return self.cum_hazard(t,x)/(t*torch.exp(self.sigma))
    
    def PDF(self,t,x):
        return self.survival(t,x) * self.hazard(t,x)
    
    def CDF(self, t,x ):
        return 1 - self.survival(t,x)
    
    def enable_grad(self):
        self.sigma.requires_grad = True
        self.mu.requires_grad = True
        self.coeff.requires_grad = True
    
    def parameters(self):
        return [self.sigma, self.mu, self.coeff]
    
    def rvs(self, x, u):
        tmp = LOG(-1*LOG(u))*torch.exp(self.sigma)
        tmp1 = torch.matmul(x, self.coeff) + self.mu
        return torch.exp(tmp+tmp1)

    
    
    

    
class Weibull_log:
    def __init__(self, nf, mu, sigma, hidden_layers, device) -> None:
        self.nf = nf
        self.mu = torch.tensor([mu],device=device).type(torch.float32)
        self.sigma = torch.tensor([sigma], device=device).type(torch.float32)
        self.net = Risk_Net(nf, hidden_layers, device)
    
    def survival(self,t,x):
        return torch.exp(-1*torch.exp((LOG(t)-self.mu-self.net(x))/torch.exp(self.sigma)))
    
    def cum_hazard(self, t,x):
        return torch.exp((LOG(t)-self.mu-self.net(x))/torch.exp(self.sigma))
    
    def hazard(self, t,x):
        return self.cum_hazard(t,x)/(t*torch.exp(self.sigma))
    
    def PDF(self,t,x):
        return self.survival(t,x) * self.hazard(t,x)
    
    def CDF(self, t,x ):
        return 1 - self.survival(t,x)
    
    def enable_grad(self):
        self.sigma.requires_grad = True
        self.mu.requires_grad = True
    
    def parameters(self):
        return [self.sigma, self.mu]+ list(self.net.parameters())
    
    def rvs(self, x, u):
        tmp = LOG(-1*LOG(u))*torch.exp(self.sigma)
        tmp1 = self.net(x) + self.mu
        return torch.exp(tmp+tmp1)





    
if __name__ == "__main__":
    w = Weibull_log(10, 1,1,[10,1])
    t = torch.ones(1000)
    x = torch.ones((1000,10))
    

    