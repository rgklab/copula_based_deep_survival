
import torch
import torch.optim as optim 
import torch.nn as nn 

import numpy as np 

from models import *
from evaluation import surv_diff,C_index, IBS_plain

from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import kendalltau

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

def generate_events(dgp1, dgp2, x, device,copula=None):
    if copula is None:
        uv = torch.rand((x.shape[0],2), device=device)
    else:
        uv = copula.rvs(x.shape[0])
    t1 = dgp1.rvs(x, uv[:,0])
    t2 = dgp2.rvs(x, uv[:,1])
    E = (t1 < t2).type(torch.float32)
    T = E * t1 + t2 *(1-E)
    return {'X':x,'E':E, 'T':T, 't1':t1, 't2':t2}


def synthetic_x(n_train, n_val, n_test, nf, device):
    x_train = torch.rand((n_train, nf), device=device)
    x_val = torch.rand((n_val, nf), device=device)
    x_test = torch.rand((n_test, nf), device=device)
    return {"x_train":x_train, "x_val":x_val, "x_test":x_test}

def generate_data(x_dict, dgp1, dgp2,device, copula=None):
    train_dict = generate_events(dgp1, dgp2, x_dict['x_train'],device, copula)
    val_dict = generate_events(dgp1, dgp2, x_dict['x_val'],device, copula)
    test_dict = generate_events(dgp1, dgp2, x_dict['x_test'],device, copula)
    return train_dict, val_dict, test_dict

def loss_function(model1, model2, data, copula=None):
    s1 = model1.survival(data['T'], data['X'])
    s2 = model2.survival(data['T'], data['X'])
    f1 = model1.PDF(data['T'], data['X'])
    f2 = model2.PDF(data['T'], data['X'])
    w = torch.mean(data['E'])
    if copula is None:
        p1 = LOG(f1) + LOG(s2)
        p2 = LOG(f2) + LOG(s1)
    else:
        
        S = torch.cat([s1.reshape(-1,1), s2.reshape(-1,1)], dim=1).clamp(0.001,0.999)
        p1 = LOG(f1) + LOG(copula.conditional_cdf("u", S))
        p2 = LOG(f2) + LOG(copula.conditional_cdf("v", S))
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    return -torch.mean(p1 * data['E'] + (1-data['E'])*p2)


def dependent_train_loop(model1, model2,train_data, val_data, copula, n_itr, optimizer1='Adam', optimizer2='Adam', lr1=1e-3, lr2=1e-2, sub_itr=5, verbose=False):
    train_loss_log = []
    val_loss_log = []
    copula_log = torch.zeros((n_itr,))
    model1.enable_grad()
    model2.enable_grad()
    copula.enable_grad()
    copula_grad_log = []
    mu_grad_log = [[], []]
    sigma_grad_log = [[], []]
    coeff_grad_log = [[], []]
    train_loss = []
    val_loss = []
    min_val_loss = 1000
    stop_itr = 0
    if optimizer1 == 'Adam':
        model_optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters())+[copula.theta], lr=lr1, weight_decay=0.0)
    if optimizer2 == 'Adam':
        copula_optimizer = torch.optim.Adam([copula.theta], lr=lr2)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, mode='min', factor=0.9, patience=1000, threshold=0.0001, verbose=True)
    for itr in tqdm(range(n_itr)):
        
        model_optimizer.zero_grad()
        loss = loss_function(model1, model2, train_data, copula)
        loss.backward()
        
        copula.theta.grad = copula.theta.grad * 100
        copula.theta.grad = copula.theta.grad.clamp(-1,1)
        
        if torch.isnan(copula.theta.grad):
            print(copula.theta)
            assert 0
        model_optimizer.step()
        if copula.theta <= 0:
            with torch.no_grad():
                copula.theta[:] = torch.clamp(copula.theta,0.001,30)
        
        #train_loss_log.append(loss.detach().clone())
        #copula_log[itr] = copula.theta.detach().clone()
        #if itr % 1000 == 0:
        #    print(min_val_loss)

    ##########################
        with torch.no_grad():
            val_loss = loss_function(model1, model2, val_data, copula)
            #scheduler.step(val_loss)
            #print(val_loss)
            #val_loss_log.append(val_loss.detach().clone())
            if not torch.isnan(val_loss) and val_loss < min_val_loss:
                stop_itr =0
                #best_perf1 = surv_diff(dgp1, model1, test_data['X'],200)
                #best_perf2 = surv_diff(dgp2, model2, test_data['X'],200)
                best_c1 = model1.coeff.detach().clone()
                best_c2 = model2.coeff.detach().clone()
                best_mu1 = model1.mu.detach().clone()
                best_mu2 = model2.mu.detach().clone()
                best_sig1 = model1.sigma.detach().clone()
                best_sig2 = model2.sigma.detach().clone()
                min_val_loss = val_loss.detach().clone()
                #model_1_dict = save_model(model1)
                #model_2_dict = save_model(model2)
                best_theta = copula.theta.detach().clone()
                #print(val_loss, copula.theta, itr)
            else:
                stop_itr += 1
                if stop_itr == 3000:  
                    break
    model1.mu = best_mu1
    model2.mu = best_mu2
    model1.sigma = best_sig1
    model2.sigma = best_sig2
    model1.coeff = best_c1
    model2.coeff = best_c2
    copula.set_theta(best_theta)
    
    return model1, model2, copula
    
    #return 1, 1, train_loss_log, val_loss_log, copula_grad_log, best_theta

def independent_train_loop_linear(model1, model2,train_data, val_data,dgp1, dgp2,test_data, n_itr, optimizer1='Adam', optimizer2='Adam', lr1=1e-3, lr2=1e-2, sub_itr=5, verbose=False):
    train_loss_log = []
    val_loss_log = []
    copula_log = torch.zeros((n_itr,))
    model1.enable_grad()
    model2.enable_grad()
    
    copula_grad_log = []
    mu_grad_log = [[], []]
    sigma_grad_log = [[], []]
    coeff_grad_log = [[], []]
    train_loss = []
    val_loss = []
    min_val_loss = 1000
    stop_itr = 0
    if optimizer1 == 'Adam':
        model_optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=lr1, weight_decay=0.0)
    
        
    for itr in tqdm(range(n_itr)):
        model_optimizer.zero_grad()
        loss = loss_function(model1, model2, train_data, None)
        loss.backward()
        model_optimizer.step() 
        train_loss_log.append(loss.detach().clone())
    ##########################
        with torch.no_grad():
            val_loss = loss_function(model1, model2, val_data, None)
            val_loss_log.append(val_loss.detach().clone())
            if not torch.isnan(val_loss) and val_loss < min_val_loss:
                stop_itr =0
                best_c1 = model1.coeff.detach().clone()
                best_c2 = model2.coeff.detach().clone()
                best_mu1 = model1.mu.detach().clone()
                best_mu2 = model2.mu.detach().clone()
                best_sig1 = model1.sigma.detach().clone()
                best_sig2 = model2.sigma.detach().clone()
                min_val_loss = val_loss.detach().clone()
            else:
                stop_itr += 1
                if stop_itr == 2000:  
                    break
    model1.mu = best_mu1
    model2.mu = best_mu2
    model1.sigma = best_sig1
    model2.sigma = best_sig2
    model1.coeff = best_c1
    model2.coeff = best_c2
    return model1, model2


def train_no_cens(model, tr_dict, val_dict, n_itr, reg):
    is_deep = isinstance(model, Weibull_log)
    model.enable_grad()
    min_val = 10000
    stop = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for itr in range(n_itr):
        optimizer.zero_grad()
        loss = -torch.mean(LOG(model.PDF(tr_dict['T'], tr_dict['X'])))
        if is_deep:
            for p in model.net.parameters():
                loss += reg * p.norm(2).sum()
        else:
            loss += reg * model.coeff.norm(2)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_loss = -torch.mean(LOG(model.PDF(val_dict['T'], val_dict['X'])))
            if val_loss < min_val:
                stop = 0
                min_val = val_loss.detach().clone()
                b_mu = model.mu.detach().clone()
                b_sig = model.sigma.detach().clone()
                if is_deep:
                    torch.save(model.net.load_state_dict(), 'no_cens.pt')
                else:
                    b_coeff = model.coeff.detach().clone()
            else:
                stop += 1
                if stop == 2000:
                    break
    model.mu = b_mu
    model.sigma = b_sig
    if is_deep:
        model.net.load_state_dict(torch.load('no_cens.pt'))
    else:
        model.coeff = b_coeff
    return model

def train_indep_model(model, tr_dict, val_dict, n_itr, reg):
    is_deep = isinstance(model, Weibull_log)
    model.enable_grad()
    min_val = 10000
    stop = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for itr in range(n_itr):
        optimizer.zero_grad()
        l1 = LOG(model.PDF(tr_dict['T'], tr_dict['X']))
        l2 = LOG(model.survival(tr_dict['T'], tr_dict['X']))
        loss = -torch.mean(tr_dict['E']*l1 + (1-tr_dict['E'])*l2)
        if is_deep:
            for p in model.net.parameters():
                loss += reg * p.norm(2).sum()
        else:
            loss += reg * model.coeff.norm(2)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            l1 = LOG(model.PDF(val_dict['T'], val_dict['X']))
            l2 = LOG(model.survival(val_dict['T'], val_dict['X']))
            val_loss = -torch.mean(val_dict['E']*l1 + (1-val_dict['E'])*l2)
            if val_loss < min_val:
                stop = 0
                min_val = val_loss.detach().clone()
                b_mu = model.mu.detach().clone()
                b_sig = model.sigma.detach().clone()
                if is_deep:
                    torch.save(model.net.load_state_dict(), 'no_cens.pt')
                else:
                    b_coeff = model.coeff.detach().clone()
            else:
                stop += 1
                if stop == 2000:
                    break
    model.mu = b_mu
    model.sigma = b_sig
    if is_deep:
        model.net.load_state_dict(torch.load('no_cens.pt'))
    else:
        model.coeff = b_coeff
    return model
    

def create_data_dict(x, t, e, t1, t2):
    return {'X':x, 'T':t, 'E':e, 't1':t1, 't2':t2}

def claculate_r2(model, x, y):
    with torch.no_grad():
        y_hat = model.rvs(x, torch.ones_like(y)*0.5).detach().clone().cpu().numpy()
    return r2_score(y.detach().cpu().numpy(), y_hat)

def censor_data_random(data_dict, cens_perc):
    data_dict_ = {'X': data_dict['X'], 't1':data_dict['T']}
    idx = torch.randperm(data_dict['T'].shape[0]).to(DEVICE)
    t_cens = data_dict['T'] * torch.rand(idx.shape, device=DEVICE)
    e = torch.ones_like(idx, device=DEVICE)
    e[idx[:int(idx.shape[0]*cens_perc)]] = 0
    T = e * data_dict['T'] + (1-e)*t_cens
    
    data_dict_['T'] = T
    data_dict_['E'] = e
    
    return data_dict_

def dep_censoring(event_model, cens_model, data_dict,  copula):
    data_dict_ = {'X':data_dict['X'], 't1': data_dict['T']}
    u = event_model.survival(data_dict['T'], data_dict['X'])
    
    v = cond_sampling(u, copula).reshape(-1,)
    print("uv",kendalltau(u.detach().clone().cpu().numpy(),v.detach().clone().cpu().numpy()))
    
    t2 = cens_model.rvs(data_dict['X'], v)
    e = (data_dict['T'] < t2).type(torch.float32)
    T = e * data_dict['T'] + (1-e)*t2
    
    data_dict_['T'] = T
    data_dict_['t2'] = t2
    data_dict_['E'] = e
    return data_dict_


def cond_sampling(u, copula):#cond samling from a copula
    v = torch.linspace(0,1, 1000, device=DEVICE)
    v = v.repeat(u.shape[0],1)
    
    cond_cdf = torch.empty_like(v, device=DEVICE)
    for i in range(v.shape[1]):
        uv_ = torch.cat((u.reshape(-1,1), v[:,i].reshape(-1,1)), dim=1)
        cond_cdf[:,i] = copula.conditional_cdf('u', uv_)
    v_ = torch.rand((u.shape[0],), device=DEVICE).reshape(-1,1).repeat(1, cond_cdf.shape[1])
    idx = torch.sum((cond_cdf < v_).type(torch.float32), dim=1)
    idx = idx.clamp(0, 999)
    ans = torch.gather(v, 1, idx.reshape(-1,1).type(torch.int64))
    return ans
