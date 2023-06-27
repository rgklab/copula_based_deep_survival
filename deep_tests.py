print(1)
from DGP import *
from models import *
from Clayton import Clayton
from Frank import Frank
from utils import *
from evaluation import *
import argparse
import pickle
from tqdm import tqdm
from net_toronto import risk1, risk2, deep_optimization_loop, deep_optimization_loop_indep, safe_log
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    print(DEVICE)
    parser = argparse.ArgumentParser()
    parser.add_argument('--nf', type=int, required=True)
    parser.add_argument('--copula', type=str, choices=['cl', 'fr'], required=True)
    parser.add_argument('--theta', type=float, required=True)
    parser.add_argument('--tau', type=int, required=True)
    args = parser.parse_args()
    print(args)
    torch.manual_seed(0)
    np.random.seed(0)
    nf = args.nf
    coeff1 = torch.rand((nf,), device=DEVICE)
    coeff2 = torch.rand((nf,), device=DEVICE)
    n_train = 20000
    n_val = 10000
    n_test = 10000
    
    
    alpha1 = 17
    gamma1 = 4
    alpha2 = 16
    gamma2 = 3
    dgp1 = Weibull_nonlinear(nf, alpha1, gamma1, risk1, DEVICE)
    dgp2 = Weibull_nonlinear(nf, alpha2, gamma2, risk2, DEVICE)
    
    if args.copula == 'cl':
        copula_dgp = Clayton(torch.tensor([args.theta], device=DEVICE).type(torch.float32), device=DEVICE)
    else:
        copula_dgp = Frank(torch.tensor([args.theta], device=DEVICE).type(torch.float32), device=DEVICE)
    x_dict = synthetic_x(n_train, n_val, n_test, nf, DEVICE)
    train_dict, val_dict, test_dict = \
                generate_data(x_dict, dgp1, dgp2,DEVICE, copula_dgp)
    run_name = args.copula + str(args.tau)
    print(run_name)
    
    
    km_h1, km_p1 = KM(train_dict['T'], 1.0-train_dict['E'])
    km_h2, km_p2 = KM(train_dict['T'], train_dict['E'])
    print(torch.mean(train_dict['E']))
    results_dict = {'dgp_loss':[], 'dep_loss':[], 'indep_loss':[],\
                        'dep1_l1':[], 'dep2_l1':[], 'indep1_l1':[],\
                        'indep2_l1':[], 'ibs_r1':[], 'ibs_r2':[], \
                        'c_r1':[], 'c_r2':[], 'theta':[]}
    for i in tqdm(range(10)):
        pass
    for i in range(10):
        print(i)
        
        if args.copula == 'cl':
            copula = Clayton(torch.tensor([5.0], device=DEVICE).type(torch.float32), device=DEVICE)
        else:
            copula = Frank(torch.tensor([5.0], device=DEVICE).type(torch.float32), device=DEVICE)
        print('copula')
        dep_model1 = Weibull_log(nf, 1,1,[10,4,4,4,2,1], device=DEVICE)
        dep_model2 = Weibull_log(nf, 1,1,[10,4,4,4,2,1], device=DEVICE)
        print('dep')
        
        while loss_function(dep_model1, dep_model2, train_dict, copula)==0:
                dep_model1 = Weibull_log(nf, 1,1,[10,4,4,4,2,1], device=DEVICE)
                dep_model2 = Weibull_log(nf, 1,1,[10,4,4,4,2,1], device=DEVICE)
        print('deep strt')
        
        dep_model1, dep_model2, copula = deep_optimization_loop(dep_model1, dep_model2, train_dict, val_dict, copula, 50000, run_name)
        print('deep ended')
        
        indep_model1 = Weibull_log(nf, 2,2,[10,4,4,4,2,1], device=DEVICE)
        indep_model2 = Weibull_log(nf, 2,2,[10,4,4,4,2,1], device=DEVICE)
        while loss_function(indep_model1, indep_model2, train_dict, None)==0:
                indep_model1 = Weibull_log(nf, 1,1,[10,4,4,4,2,1], device=DEVICE)
                indep_model2 = Weibull_log(nf, 1,1,[10,4,4,4,2,1], device=DEVICE)
        print('indep started')
        
        indep_model1, indep_model2 = deep_optimization_loop_indep(indep_model1, indep_model2, train_dict, val_dict, 50000, run_name)
        #likelihood
        print('indep end')
        
        results_dict['theta'].append(copula.theta.cpu().detach().clone().item())
        dgp_loss = loss_function(dgp1, dgp2, test_dict, copula_dgp).cpu().detach().clone().numpy().item()
        dep_loss = loss_function(dep_model1, dep_model2, test_dict, copula).cpu().detach().clone().numpy().item()
        indep_loss = loss_function(indep_model1, indep_model2, test_dict, None).cpu().detach().clone().numpy().item()
        results_dict['dgp_loss'].append(dgp_loss)
        results_dict['dep_loss'].append(dep_loss)
        results_dict['indep_loss'].append(indep_loss)
        
        #IBS
        with torch.no_grad():
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
    file_name = "deep_"+args.copula+"_" + str(args.tau)+'.pickle'
    with open(file_name, 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)