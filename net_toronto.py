import torch
from Frank import Frank
from Clayton import Clayton
from DGP import *
from models import *
import matplotlib.pyplot as plt
from utils import loss_function
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#DEVICE = 'cpu'
"""def risk1(x, coeff):
    return 2*torch.sin(torch.pi * x*2).reshape(-1,)
    #return torch.sum((2*x)**2, dim=1).reshape(-1,)

def risk2(x, coeff):
    return 2*torch.sin(torch.pi * (x+0.1)*2).reshape(-1,)"""

def risk1(x, coeff):
    tmp = torch.sum(x[:,0:2]**2, dim=1)
    #tmp = torch.sum(torch.sqrt(x), dim=1)
    return tmp.reshape(-1,)
    #return torch.sum((2*x)**2, dim=1).reshape(-1,)

def risk2(x, coeff):
    tmp = torch.matmul(x[:,:3]**2, coeff[:3])
    #tmp = torch.sum(torch.sigmoid(x), dim=1)
    return tmp.reshape(-1,)


def safe_log(x):
    return torch.log(x+1e-6*(x<1e-6))

def deep_optimization_loop(model1, model2, train_data, val_data,copula, n_itr, folder_name):
    model1.enable_grad() 
    model2.enable_grad()
    copula.enable_grad()
    min_val_loss = 1000
    stop_itr = 0
    copula_log = []
    optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters())+[copula.theta], lr=1e-3)
    

    sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=100, verbose=False)
    for itr in range(n_itr):
        
        optimizer.zero_grad()
        model1.net.train()
        model2.net.train()
        loss = loss_function(model1, model2, train_data, copula)
        for p in model1.net.parameters():
            loss += 0.001 * p.norm(2).sum()
        for p in model2.net.parameters():
            loss += 0.001 * p.norm(2).sum()
        loss.backward()
        copula.theta.grad = (copula.theta.grad * 1000).clamp(-0.1,0.1)
        optimizer.step()
        if copula.theta <= 0:
            with torch.no_grad():
                copula.theta[:] = torch.clamp(copula.theta,0.001, 100)

        with torch.no_grad():
            model1.net.eval()
            model2.net.eval()
            val_loss = loss_function(model1, model2, val_data, copula)
            if not torch.isnan(val_loss) and val_loss < min_val_loss:
                stop_itr = 0
                min_val_loss = val_loss.detach().clone()
                best_mu1 = model1.mu.detach().clone()
                best_mu2 = model2.mu.detach().clone()
                best_sig1 = model1.sigma.detach().clone()
                best_sig2 = model2.sigma.detach().clone()
                torch.save(model1.net.state_dict(), folder_name+'/mig_mig1.pt')
                torch.save(model2.net.state_dict(), folder_name+'/mig_mig2.pt')
                best_theta = copula.theta.detach().clone()
            else:
                stop_itr +=1 
                if stop_itr == 3000:
                    break
    model1.mu = best_mu1
    model2.mu = best_mu2
    model1.sigma = best_sig1
    model2.sigma = best_sig2
    model1.net.load_state_dict(torch.load(folder_name+'/mig_mig1.pt'))
    model2.net.load_state_dict(torch.load(folder_name+'/mig_mig2.pt'))
    copula.theta = best_theta
    model1.net.eval()
    model2.net.eval()
    return model1, model2, copula

def deep_optimization_loop_indep(model1, model2, train_data, val_data, n_itr, folder_name):
    model1.enable_grad() 
    model2.enable_grad()
    
    min_val_loss = 1000
    stop_itr = 0
    optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=1e-3)
    for itr in range(n_itr):
        
        optimizer.zero_grad()
        loss = loss_function(model1, model2, train_data, None)
        for p in model1.net.parameters():
            loss += 0.001 * p.norm(2).sum()
        for p in model2.net.parameters():
            loss += 0.001 * p.norm(2).sum()
        loss.backward()
        
        
        optimizer.step()

        with torch.no_grad():
            val_loss = loss_function(model1, model2, val_data, None)
            
            if not torch.isnan(val_loss) and val_loss < min_val_loss:
                stop_itr = 0
                min_val_loss = val_loss.detach().clone()
                best_mu1 = model1.mu.detach().clone()
                best_mu2 = model2.mu.detach().clone()
                best_sig1 = model1.sigma.detach().clone()
                best_sig2 = model2.sigma.detach().clone()
                torch.save(model1.net.state_dict(), folder_name+'/mig_mig1.pt')
                torch.save(model2.net.state_dict(), folder_name+'/mig_mig2.pt')
                
            else:
                stop_itr +=1 
                if stop_itr == 2000:
                    break
    model1.mu = best_mu1
    model2.mu = best_mu2
    model1.sigma = best_sig1
    model2.sigma = best_sig2
    model1.net.load_state_dict(torch.load(folder_name+'/mig_mig1.pt'))
    model2.net.load_state_dict(torch.load(folder_name+'/mig_mig2.pt'))
    
    return model1, model2

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    nf = 10
    coeff1 = torch.rand((nf,), device=DEVICE)
    coeff2 = torch.rand((nf,), device=DEVICE)
    n_train =20000
    n_val = 10000
    n_test = 10000
    
    alpha1 = 17
    gamma1 = 4
    alpha2 = 16
    gamma2 = 3
    dgp1 = Weibull_nonlinear(nf, alpha1, gamma1, risk1, DEVICE)
    dgp2 = Weibull_nonlinear(nf, alpha2, gamma2, risk2, DEVICE)
    #dgp1.coeff = coeff1*2
    #dgp2.coeff = coeff2
    

    copula = Frank(torch.tensor([0.1], device= DEVICE).type(torch.float32), DEVICE)
    x_dict = synthetic_x(n_train, n_val, n_test, nf, DEVICE)
    train_dict, val_dict, test_dict = \
                generate_data(x_dict, dgp1, dgp2,  DEVICE,copula)
    print(torch.mean(train_dict['E']))
    print('dgp_val_loss',loss_function(dgp1, dgp2, val_dict, copula))
    print('dgp_test_loss', loss_function(dgp1, dgp2, test_dict, copula))
    #assert 0
    
    #copula = Clayton(torch.tensor([4]).type(torch.float32))
    #model1 = Weibull_log(nf, 2,2,[4,4,1], device=DEVICE)
    #model2 = Weibull_log(nf, 2,2,[4,4,1], device=DEVICE)
    model1 = Weibull_log(nf, 1,1,[10,4,4,4,2,1], device=DEVICE)
    model2 = Weibull_log(nf, 1,1,[10,4,4,4,2,1], device=DEVICE)
    #
   
    #model = Weibull_log_linear(nf, 2,2)
    

    copula = Frank(torch.tensor([5],device=DEVICE).type(torch.float32), device=DEVICE)
    print(loss_function(model1, model2, val_dict, copula))
    
    
    model1.enable_grad() 
    model2.enable_grad()
    copula.enable_grad()
    min_val_loss = 1000
    
    
    model1, model2, copula = deep_optimization_loop(model1, model2, copula, train_dict,\
                                                    val_dict, 40000)
    print(loss_function(model1, model2, val_dict, copula),loss_function(model1, model2, test_dict, copula), copula.theta)
    print(surv_diff(dgp1, model1, test_dict['X'], 200), surv_diff(dgp2, model2, test_dict['X'], 200))

    assert 0
    model1 = Weibull_log(nf, 2,2,[16,16,1])
    model2 = Weibull_log(nf, 2,2,[16,16,1])
    model1, model2  = deep_optimization_loop_indep(model1, model2, train_dict,\
                                                    val_dict, 40000)
    print(loss_function(model1, model2, val_dict, copula),loss_function(model1, model2, test_dict, copula))
    print(surv_diff(dgp1, model1, test_dict['X'], 200), surv_diff(dgp2, model2, test_dict['X'], 200))
    print(copula.theta)
    """for itr in range(25000):
        if itr%1000==0:
            print(itr, min_val_loss)
        optimizer.zero_grad()
        loss = loss_function(model1, model2, train_dict, copula)
        loss.backward()
        copula.theta.grad = (copula.theta.grad * 50).clamp(-1,1)
        copula_grad.append(copula.theta.grad.detach().clone())
        optimizer.step()

        with torch.no_grad():
            val_loss = loss_function(model1, model2, val_dict, copula)
            #val_loss.append(loss_function(model1, model2, val_dict, copula))
            if not torch.isnan(val_loss) and val_loss < min_val_loss:
                min_val_loss = val_loss.detach().clone()
                best_mu1 = model1.mu.detach().clone()
                best_mu2 = model2.mu.detach().clone()
                best_sig1 = model1.sigma.detach().clone()
                best_sig2 = model2.sigma.detach().clone()
                torch.save(model1.net.state_dict(), 'mig_mig1.pt')
                torch.save(model2.net.state_dict(), 'mig_mig2.pt')
                best_theta = copula.theta.detach().clone()"""

    #plt.plot(copula_grad)
    #print(best_theta)

    #print(surv_diff(dgp1, model1, test_dict['X'],200))
    #print(surv_diff(dgp2, model2, test_dict['X'],200))
    """model1.mu = best_mu1
    model2.mu = best_mu2
    model1.sigma = best_sig1
    model2.sigma = best_sig2
    model1.net.load_state_dict(torch.load('mig_mig1.pt'))
    model2.net.load_state_dict(torch.load('mig_mig2.pt'))
    copula.theta = best_theta"""
    #print(min_val_loss)
    #print('learned val loss', loss_function(model1, model2, val_dict, copula))
    #print('learned theta',copula.theta)
    #print('learned test loss',loss_function(model1, model2, test_dict, copula))
    #t = torch.linspace(0,40,200)
    #x = torch.ones((200, nf))*0.5
    
    #plt.plot(t, dgp1.survival(t, x), label = 'dgp1')
    #plt.plot(t, model1.survival(t, x).detach(), label = 'm1')

    #plt.plot(t, dgp2.survival(t, x), label = 'dgp2')
    #plt.plot(t, model2.survival(t, x).detach(), label = 'm2')
    #plt.plot(val_loss)
    #print(min(val_loss))
    #plt.show()
    #x = torch.linspace(0,1,100).reshape(-1,1)
    """ plt.subplot(2,1,1)
    plt.plot(x, risk1(x, 0), label='sine_1_dgp')
    plt.plot(x, risk2(x, 0), label='sine_2_dgp')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(x, -model1.net(x).detach(), label='sine_1')
    plt.plot(x, -model2.net(x).detach(), label='sine_2')"""
    #plt.legend()
    #plt.plot(x, risk1(x, 0))
    #plt.plot(x, risk2(x, 0))
    #plt.plot(x, -(model.net(x)/torch.exp(model.sigma)).detach())
    #plt.legend()
    #plt.show()
        
    



    

