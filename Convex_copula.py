from Clayton import Clayton
from Frank import Frank
import torch

class Convex_Arch:
    def __init__(self, theta1,theta2,device, beta=0):
        self.beta = torch.tensor([beta], device=device).type(torch.float32)
        self.clayton = Clayton(torch.tensor([theta1], device=device).type(torch.float32), device=device)
        self.frank = Frank(torch.tensor([theta2], device=device).type(torch.float32), device=device)
        self.device = device
    def PDF(self,uv):
        p = 1.0/(1+torch.exp(-self.beta))
        return self.clayton.PDF(uv)* p + (1.0-p)*self.frank.PDF(uv)
    
    def CDF(self, uv):
        p = 1.0/(1+torch.exp(-self.beta))
        return self.clayton.CDF(uv)* p + (1.0-p)*self.frank.CDF(uv)
    
    def conditional_cdf(self, condition_on, uv):
        p = 1.0/(1+torch.exp(-self.beta))
        return self.clayton.conditional_cdf(condition_on, uv)* p + (1.0-p)*self.frank.conditional_cdf(condition_on, uv)
    
    def enable_grad(self):
        self.clayton.theta.requires_grad = True
        self.frank.theta.requires_grad = True
        self.beta.requires_grad = True

    
    def disable_grad(self):
        self.clayton.theta.requires_grad = False
        self.frank.theta.requires_grad = False
        self.beta.requires_grad = False
    
    def parameters(self):
        return [self.beta, self.clayton.theta, self.frank.theta]
    
    def rvs(self, n_samples):
        p = 1.0/(1+torch.exp(-self.beta))
        z = torch.distributions.bernoulli.Bernoulli(p).sample((n_samples,)).repeat(1,2)
        uv1 = self.clayton.rvs(n_samples)
        uv2 = self.frank.rvs(n_samples)
        uv = z * uv1 + (1-z)*uv2
        return uv
if __name__ == "__main__":
    con = Convex_Arch(2)
    print(con.rvs(10))

    