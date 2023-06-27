from DGP import *
from models import *
from Clayton import Clayton
from Frank import Frank
from utils import *
from models import *

from evaluation import *
import matplotlib.pyplot as plt
from Convex_copula import Convex_Arch
import argparse
import pickle
from tqdm import tqdm
from utils import loss_function

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def dependent_train_loop(model1, model2,train_data, val_data, copula,dgp1, dgp2,test_data, n_itr, optimizer1='Adam', optimizer2='Adam', lr1=1e-3, lr2=1e-2, sub_itr=5, verbose=False):
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
        model_optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters())+list(copula.parameters()), lr=lr1, weight_decay=0.0)
    

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, mode='min', factor=0.9, patience=1000, threshold=0.0001, verbose=True)
    for itr in tqdm(range(n_itr)):
        
        model_optimizer.zero_grad()
        loss = loss_function(model1, model2, train_data, copula)
        loss.backward()
        #print(loss)
        #print(copula.theta.grad)
        #print(copula.theta.grad)
        copula.clayton.theta.grad = copula.clayton.theta.grad * 1000
        copula.clayton.theta.grad = copula.clayton.theta.grad.clamp(-0.1,0.1)
        copula.frank.theta.grad = copula.frank.theta.grad * 1000
        copula.frank.theta.grad = copula.frank.theta.grad.clamp(-0.1,0.1)
        #torch.nn.utils.clip_grad_value_(copula.theta, 1)
        #torch.nn.utils.clip_grad_value_(model1.parameters(), 1)
        #torch.nn.utils.clip_grad_value_(model2.parameters(), 1)
        #if torch.isnan(copula.theta.grad):
        #    print(copula.theta)
        #    assert 0
        """copula.theta.grad = torch.nan_to_num(copula.theta.grad, 0.0)
        model1.mu.grad = torch.nan_to_num(model1.mu.grad, 0.0)
        model2.mu.grad = torch.nan_to_num(model2.mu.grad, 0.0)
        model1.sigma.grad = torch.nan_to_num(model1.sigma.grad, 0.0)
        model2.sigma.grad = torch.nan_to_num(model2.sigma.grad, 0.0)
        model1.coeff.grad = torch.nan_to_num(model1.coeff.grad, 0.0)
        model2.coeff.grad = torch.nan_to_num(model2.coeff.grad, 0.0)"""

        """if torch.isnan(copula.theta.grad):
            copula.theta.grad = torch.ones_like(copula.theta.grad) * 0.2
        if torch.isnan(model1.coeff.grad):
            model1.coeff.grad = 0.2 * torch.ones_like(model1.coeff.grad)
        if torch.isnan(model2.coeff.grad):
            model2.coeff.grad = 0.2 * torch.ones_like(model2.coeff.grad)
        if torch.isnan(model1.mu.grad):
            model1.mu.grad = 0.2 * torch.ones_like(model1.mu.grad)
        if torch.isnan(model2.mu.grad):
            model2.mu.grad = 0.2 * torch.ones_like(model2.mu.grad)
        if torch.isnan(model1.sigma.grad):
            model1.sigma.grad = 0.2 * torch.ones_like(model1.sigma.grad)
        if torch.isnan(model2.sigma.grad):
            model2.sigma.grad = 0.2 * torch.ones_like(model2.sigma.grad)"""
        
        

        
        #copula_grad_log.append(copula.theta.grad.detach().clone())
        """mu_grad_log[0].append(model1.mu.grad.detach().clone())
        mu_grad_log[1].append(model2.mu.grad.detach().clone())
        sigma_grad_log[0].append(model1.sigma.grad.detach().clone())
        sigma_grad_log[1].append(model2.sigma.grad.detach().clone())

        coeff_grad_log[0].append(model1.coeff.grad.detach().clone().reshape(-1,1))
        coeff_grad_log[1].append(model2.coeff.grad.detach().clone().reshape(-1,1))"""
        model_optimizer.step()
        
        #train_loss_log.append(loss.detach().clone())
        #copula_log[itr] = copula.theta.detach().clone()
        #if itr % 1000 == 0:
        #    print(min_val_loss)

    ##########################
        with torch.no_grad():
            val_loss = loss_function(model1, model2, val_data, copula)
            #scheduler.step(val_loss)
            #print(val_loss)
            val_loss_log.append(val_loss.detach().clone())
            if not torch.isnan(val_loss) and val_loss + 5e-5 < min_val_loss:
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
                best_theta_c = copula.clayton.theta.detach().clone()
                best_theta_f = copula.frank.theta.detach().clone()
                best_w = copula.beta.detach().clone()
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
    copula.clayton.theta = best_theta_c
    copula.frank.theta = best_theta_f
    copula.beta = best_w
    """best_perf1 = surv_diff(dgp1, model1, test_data['X'],200)
    best_perf2 = surv_diff(dgp2, model2, test_data['X'],200)
    print('gamma:', 1/torch.exp(best_sig1), 1/torch.exp(best_sig2))
    print('alpha:', torch.exp(best_mu1), torch.exp(best_mu2))
    print('coeff:', -best_c1/torch.exp(best_sig1), -best_c2/torch.exp(best_sig2))

    #print(best_mu1, best_mu2, best_sig1, best_sig2)
    print(min_val_loss, best_theta_c, best_theta_f, best_w, itr, best_perf1, best_perf2,\
        loss_function(model1, model2, val_data, copula), loss_function(model1, model2, test_data, copula))"""
    return model1, model2, copula
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nf', type=int, required=True)
    parser.add_argument('--theta1', type=float, required=True)
    parser.add_argument('--theta2', type=float, required=True)
    parser.add_argument('--tau', type=int, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    nf = args.nf
    n_train = 20000
    n_val = 10000
    n_test = 10000
    x_dict = synthetic_x(n_train, n_val, n_test, nf, DEVICE)
    dgp1 = Weibull_linear(nf, 14, 4,DEVICE)
    dgp2 = Weibull_linear(nf, 16, 3,DEVICE)
    dgp1.coeff = torch.rand((nf,),device=DEVICE)
    dgp2.coeff = torch.rand((nf,), device=DEVICE)
        
    copula_dgp = Convex_Arch(args.theta1, args.theta2, device=DEVICE)
        
    train_dict, val_dict, test_dict = \
            generate_data(x_dict, dgp1, dgp2,DEVICE, copula_dgp)
    
    min_val = 1000
    min_test = 1000
    
    km_h1, km_p1 = KM(train_dict['T'], 1.0-train_dict['E'])
    km_h2, km_p2 = KM(train_dict['T'], train_dict['E'])
    results_dict = {'dgp_loss':[], 'dep_loss':[], 'indep_loss':[],\
                        'dep1_l1':[], 'dep2_l1':[], 'indep1_l1':[],\
                        'indep2_l1':[], 'ibs_r1':[], 'ibs_r2':[], \
                        'c_r1':[], 'c_r2':[],\
                        'theta_cl':[], 'theta_fr':[], 'p':[]}
    #print(loss_function(dgp1, dgp2, val_dict, copula_dgp))
    #print(loss_function(dgp1, dgp2, test_dict, copula_dgp))
        
    print(torch.mean(train_dict['E']))
        
        
        #assert 0
    for i in range(10):
        copula = Convex_Arch(4, 6, device=DEVICE)
        dep_model1 = Weibull_log_linear(nf, 2,2, DEVICE)
        dep_model2 = Weibull_log_linear(nf, 2,2, DEVICE)
        while loss_function(dep_model1, dep_model2, train_dict, copula)==0:
            dep_model1 = Weibull_log_linear(nf, 2,2, DEVICE)
            dep_model2 = Weibull_log_linear(nf, 2,2, DEVICE)
            
        dep_model1, dep_model2, copula = dependent_train_loop(dep_model1, dep_model2, train_dict, val_dict, copula, dgp1, dgp2, test_dict, 30000)

        indep_model1 = Weibull_log_linear(nf, 2,1, DEVICE)
        indep_model2 = Weibull_log_linear(nf, 1,1, DEVICE)
        while loss_function(indep_model1, indep_model2, train_dict, copula)==0:
            indep_model1 = Weibull_log_linear(nf, 2,1, DEVICE)
            indep_model2 = Weibull_log_linear(nf, 1,1, DEVICE)

        indep_model1, indep_model2 = independent_train_loop_linear(indep_model1, indep_model2, train_dict, val_dict, dgp1, dgp2, test_dict, 30000)
        #likelihood
        results_dict['p'].append((1/(1+torch.exp(-copula.beta))).cpu().detach().clone().item())
        results_dict['theta_cl'].append(copula.clayton.theta.cpu().detach().clone().item())
        results_dict['theta_fr'].append(copula.frank.theta.cpu().detach().clone().item())
        dgp_loss = loss_function(dgp1, dgp2, test_dict, copula_dgp).cpu().detach().clone().numpy().item()
        dep_loss = loss_function(dep_model1, dep_model2, test_dict, copula).cpu().detach().clone().numpy().item()
        indep_loss = loss_function(indep_model1, indep_model2, test_dict, None).cpu().detach().clone().numpy().item()
        results_dict['dgp_loss'].append(dgp_loss)
        results_dict['dep_loss'].append(dep_loss)
        results_dict['indep_loss'].append(indep_loss)
        
        #IBS
        IBS_r1 = evaluate_IBS(dep_model1, indep_model1, dgp1, test_dict,km_h1, km_p1, False)
        IBS_r2 = evaluate_IBS(dep_model2, indep_model2, dgp2, test_dict,km_h2, km_p2, True)
        results_dict['ibs_r1'].append(IBS_r1)
        results_dict['ibs_r2'].append(IBS_r2)
        #c_index
        c_index_r1 = evaluate_c_index(dep_model1, indep_model1, dgp1, test_dict, False)
        c_index_r2 = evaluate_c_index(dep_model2, indep_model2, dgp2, test_dict, True)
        results_dict['c_r1'].append(c_index_r1)
        results_dict['c_r2'].append(c_index_r2)
        #l1
        dep_model1_l1 = surv_diff(dgp1, dep_model1, test_dict['X'], 200).cpu().detach().clone().numpy().item()
        dep_model2_l1 = surv_diff(dgp2, dep_model2, test_dict['X'], 200).cpu().detach().clone().numpy().item()
        indep_model1_l1 = surv_diff(dgp1, indep_model1, test_dict['X'], 200).cpu().detach().clone().numpy().item()
        indep_model2_l1 = surv_diff(dgp2, indep_model2, test_dict['X'], 200).cpu().detach().clone().numpy().item()
        results_dict['dep1_l1'].append(dep_model1_l1)
        results_dict['dep2_l1'].append(dep_model2_l1)
        results_dict['indep1_l1'].append(indep_model1_l1)
        results_dict['indep2_l1'].append(indep_model2_l1)
    print(results_dict)
    file_name = "convex_linear_" + str(args.tau)+'.pickle'
    with open(file_name, 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            