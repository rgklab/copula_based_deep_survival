import torch 
import matplotlib.pyplot as plt
#from lifelines import KaplanMeierFitter
from utils import *
#from Clayton import Clayton

def risk_1(x, coeff):
    return torch.square(torch.matmul(x, coeff))/x.shape[1]

def risk_poly(x, coeff):
    #return torch.matmul(x, torch.ones((10,)))/5
    #return (torch.matmul(x,coeff)/3)**4
    return 0.3*torch.matmul(torch.sigmoid(x), coeff)**2
    #return torch.sin(torch.matmul(x, coeff))
    #return (torch.matmul(x, coeff) + torch.matmul(x**2, coeff) + torch.matmul((x**3)+(x**4)+(x**5)+(x**6), coeff))*0.2
    #return (torch.matmul(x, coeff) + torch.matmul(x**2, coeff) + torch.matmul((x**3)+(x**4), coeff))*0.1

#def risk_m(x, coeff):
#    return torch.square(torch.matmul(x, coeff))

def risk_m(x, coeff):
    
    return torch.square(torch.matmul(x, coeff))/10

def risk_2(x, coeff):
    return 2.5*torch.sum(torch.sigmoid(x*coeff), dim=1)/x.shape[1]

class Exp_linear:
    def __init__(self, bh, nf) -> None:
        self.nf = nf
        self.bh = torch.tensor([bh]).type(torch.float32)
        self.coeff = torch.rand((nf,))
    
    def hazard(self, t, x):
        return self.bh * torch.exp(torch.matmul(x, self.coeff))
    
    def cum_hazard(self, t, x):
        return self.hazard(t, x) * t
    
    def survival(self, t, x):
        return torch.exp(-self.cum_hazard(t, x))
    
    def CDF(self, t, x):
        return 1.0 - self.survival(t, x)
    
    def PDF(self, t, x):
        return self.survival(t,x)*self.hazard(t,x)
    
    def rvs(self, x, u):
        return -LOG(u)/self.hazard(t=None, x=x)
    
class EXP_nonlinear:
    def __init__(self, bh, nf, risk_function) -> None:
        self.nf = nf
        self.bh = torch.tensor([bh]).type(torch.float32)
        self.coeff = torch.rand((nf,))
        self.risk_function = risk_function
    
    def hazard(self, t, x):
        return self.bh * torch.exp(self.risk_function(x, self.coeff ))
    
    def cum_hazard(self, t, x):
        return self.hazard(t, x) * t
    
    def survival(self, t, x):
        return torch.exp(-self.cum_hazard(t, x))
    
    def CDF(self, t, x):
        return 1.0 - self.survival(t, x)

    def PDF(self, t, x):
        return self.survival(t,x)*self.hazard(t,x)
    
    def rvs(self, x, u):
        return -LOG(u)/self.hazard(t=None, x=x)


class Weibull_linear:
    def __init__(self, nf, alpha, gamma, device):
        #torch.manual_seed(0)
        self.nf = nf
        self.alpha = torch.tensor([alpha], device=device).type(torch.float32)
        self.gamma = torch.tensor([gamma], device=device).type(torch.float32)
        
        self.coeff = torch.rand((nf,), device=device).type(torch.float32)#.clamp(0.1,1.0)
        

    def PDF(self ,t ,x):
        return self.hazard(t, x) * self.survival(t,x)
    
    def CDF(self ,t ,x):   
        return 1 - self.survival(t,x)
    
    def survival(self ,t ,x):   
        return torch.exp(-self.cum_hazard(t,x))
    
    def hazard(self, t, x):
        return ((self.gamma/self.alpha)*((t/self.alpha)**(self.gamma-1))) * torch.exp(torch.matmul(x, self.coeff))
        

    def cum_hazard(self, t, x):
        return ((t/self.alpha)**self.gamma) * torch.exp(torch.matmul(x, self.coeff))
    
    
    def rvs(self, x, u):
        return ((-LOG(u)/torch.exp(torch.matmul(x, self.coeff)))**(1/self.gamma))*self.alpha

class Weibull_nonlinear:
    #torch.manual_seed(0)
    def __init__(self, nf, alpha, gamma, risk_function, device):
        #torch.manual_seed(0)
        self.nf = nf
        self.alpha = torch.tensor([alpha],device=device).type(torch.float32)
        self.gamma = torch.tensor([gamma], device=device).type(torch.float32)
        self.coeff = torch.rand((nf,),device=device)
        self.risk_function = risk_function
    def PDF(self ,t ,x):
        return self.hazard(t, x) * self.survival(t, x)
    
    def CDF(self ,t ,x):    
        return 1 - self.survival(t, x)
    
    def survival(self ,t ,x):   
        return torch.exp(-self.cum_hazard(t, x))
    
    def hazard(self, t, x):
        
        return ((self.gamma/self.alpha)*((t/self.alpha)**(self.gamma-1))) * torch.exp(self.risk_function(x,self.coeff))
        

    def cum_hazard(self, t, x):
        return ((t/self.alpha)**self.gamma) * torch.exp(self.risk_function(x,self.coeff))
    
    def rvs(self, x, u):
        return ((-LOG(u)/torch.exp(self.risk_function(x, self.coeff)))**(1/self.gamma))*self.alpha


if __name__ == "__main__":
    
    dgp1 =Weibull_linear(2, 14, 3)
    dgp1.coeff = torch.tensor([0.3990, 0.5167])
    x = torch.rand((1000,2))
    t = dgp1.rvs(x, torch.rand((1000,)))
    print(torch.min(t), torch.max(t))

    