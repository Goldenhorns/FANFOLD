#%% import packet
import argparse
import data_loader
import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.distributions import MultivariateNormal
from sklearn.metrics import auc, roc_curve
from loss import *
from my_model import *
from utils import *
from datetime import datetime
from matplotlib import rcParams
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
config = {
            "font.family":'Times New Roman',
            "font.size": 12,
            "mathtext.fontset":'stix',
          }
rcParams.update(config)

#%% Parameter information
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-DS', help='Dataset', default='COX2') 
    # AIDS BZR COX2 DHFR NCI1 
    # ENZYMES PROTEINS COLLAB
    # IMDB-BINARY REDDIT-BINARY DD
    # Tox21_HSE Tox21_MMP Tox21_p53 Tox21_PPAR-gamma
    parser.add_argument('-DS_ood', help='Dataset', default='')
    parser.add_argument('-DS_pair', default=None)
    parser.add_argument('-rw_dim', type=int, default=16)
    parser.add_argument('-dg_dim', type=int, default=16)
    parser.add_argument('-gnn_name', type=str, default='GIN') # GAT GCN GIN GraphUNet GraphSAGE
    parser.add_argument('-batch_size', type=int, default=3)
    parser.add_argument('-batch_size_test', type=int, default=9999)
    parser.add_argument('-lr', type=float, default= 0.0001)
    parser.add_argument('-lr1', type=float, default=0.0001)
    parser.add_argument('-lr2', type=float, default=0.0001)
    parser.add_argument('-num_timesteps', type=int, default=2)
    parser.add_argument('-hidden_dim', type=int, default=7)
    parser.add_argument('-hidden_dim1', type=int, default=20)
    parser.add_argument('-hidden_dim2', type=int, default=6)
    parser.add_argument('-num_epoch', type=int, default=2)
    parser.add_argument('-num_epoch1', type=int, default=4)
    parser.add_argument('-num_epoch2', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.)
    parser.add_argument('-dropout1', type=float, default=0.)
    parser.add_argument('-dropout2', type=float, default=0.)
    parser.add_argument('-num_layer', type=int, default=2)
    parser.add_argument('-alpha', type=float, default=0.8)
    parser.add_argument('-beta', type=float, default=0.5)
    parser.add_argument('-loss_type', type=bool, default=False)
    parser.add_argument('-use_a', type=bool, default=True)
    parser.add_argument('-readout', type=str, default='max')
    parser.add_argument('-seed',type=int, default=7)
    parser.add_argument('-num_trial', type=int, default=5)
    return parser.parse_known_args() 
#%% Experimental information
args, unknown = arg_parse()
args_dic = vars(args); Pkey=[]; Pvalue=[]
for key, value in args_dic.items():
    Pkey.append(key)
    Pvalue.append(value)
setup_seed(args.seed)
cnt_wait=0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
if args.DS.startswith('Tox21'):
    dataloader, dataloader_test, meta=data_loader.get_ad_dataset_Tox21(args)
else:
    splits = data_loader.get_ad_split_TU(args, fold=args.num_trial)
    dataloader, dataloader_test, meta=data_loader.get_ad_dataset_TU(args,split=splits[1])
zero_edge_index=torch.tensor([[0],[0]]).to(device)
if not args.loss_type:
    output_dim=meta['num_feat']+args.dg_dim + args.rw_dim
    args.output_dim=output_dim
print('==========Experimental information==========')
print('Device:', device)
print('Dataset: {}'.format(args.DS_pair if args.DS_pair is not None else args.DS))
print('Num_features: {}'.format(meta['num_feat']))
print('Num_structural_encodings: {}'.format(args.dg_dim + args.rw_dim))
print('Hidden_dim: {}'.format(args.hidden_dim))
print('Seed: {}'.format(args.seed))
print('num_epoch: {}'.format(args.num_epoch))
print('hidden_dim: {}'.format(args.hidden_dim))
%% Train TeaEncoder
aucs=[]
for trial in tqdm(range(args.num_trial), colour='red', dynamic_ncols=True, desc='[Total      trial]'):
    TeaEnmodel=make_gnn_model(args, hidden_dim=args.hidden_dim, out_dim=output_dim).to(device); 
    TeaEnoptimizer = torch.optim.Adam(TeaEnmodel.parameters(), lr=args.lr)
    best = 1e9; best_t = 0
    with tqdm(total=args.num_epoch,colour='blue',dynamic_ncols=True, leave=False) as pbar:
        pbar.set_description('[Train TeaEncoder]')
        for epoch in range(1, args.num_epoch + 1):
            TeaEnmodel.train(); loss_tea = 0
            for data in dataloader:
                data = data.to(device); TeaEnoptimizer.zero_grad()
                x = TeaEnmodel(x=torch.cat((data.x, data.x_s), dim=1), edge_index=data.edge_index)
                if args.loss_type:
                    loss_tea, true_adj, pred_adj= \
                    caculate_loss(output_X=x, 
                                  edge_index=data.edge_index,
                                  device=device)
                else:
                    loss_tea, true_adj, pred_adj= \
                    caculate_loss_p(x=torch.cat((data.x, data.x_s), dim=1), 
                                    output_X=x,
                                    edge_index=data.edge_index,
                                    device=device,
                                    alpha=args.alpha)
                loss_tea.backward(); TeaEnoptimizer.step()

            loss_tea = loss_tea.detach().cpu().numpy()
            mean_loss = (loss_tea) / len(x)

            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(TeaEnmodel.state_dict(), r'./modelpkl/best_TeaEnmodel.pkl')
            else:
                cnt_wait += 1
            pbar.set_postfix(loss=mean_loss)
            pbar.update()
#%% NormalFlow
    TeaEnmodel.load_state_dict(torch.load(r'./modelpkl/best_TeaEnmodel.pkl')); TeaEnmodel.eval()
    NfModel=NF(args,
               use_batch_norm=False,
               weight_sharing=False,
               name="GRevNet").to(device)
    Nfoptimizer = torch.optim.Adam(NfModel.parameters(), lr=args.lr1)
    best = 1e9; best_t = 0
    mvn = MultivariateNormal(torch.zeros(args.output_dim).to(device), 
                             scale_tril=torch.diag(torch.ones(args.output_dim)).to(device))
    with tqdm(total=args.num_epoch1,colour='blue',dynamic_ncols=True, leave=False) as pbar:
        pbar.set_description('[Train  NfEncoder]')
        for epoch in range(1, args.num_epoch1 + 1):
            for data in dataloader:
                Nfoptimizer.zero_grad()
                data=data.to(device)
                with torch.no_grad():
                    x=TeaEnmodel(x=torch.cat((data.x, data.x_s), dim=1), edge_index=data.edge_index)
                    x=torch.sigmoid(x)
                if args.use_a:
                    nfoutput, log_det_jacobian=NfModel(x=x, edge_index=data.edge_index)
                else:
                    nfoutput, log_det_jacobian=NfModel(x=x, edge_index=zero_edge_index)
                num_nodes=len(nfoutput)
                nfoutput=torch.sigmoid(nfoutput)
                log_prob_zs = torch.sum(mvn.log_prob(nfoutput))
                log_prob_xs = log_prob_zs + log_det_jacobian
                nf_loss = -1 * log_prob_xs
                nf_loss.backward()
                Nfoptimizer.step()
                if nf_loss < best:
                    best = nf_loss
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(NfModel.state_dict(), r'./modelpkl/best_NfEnmodel.pkl')
                else:
                    cnt_wait += 1
            pbar.set_postfix(loss=nf_loss.detach().cpu().numpy()/num_nodes)
            pbar.update()

#%% Train StuEncoder
    StuEnmodel = make_gnn_model(args, hidden_dim=args.hidden_dim2, out_dim=output_dim).to(device)
    StuEnoptimizer = torch.optim.Adam(StuEnmodel.parameters(), lr=args.lr2)
    TeaEnmodel.load_state_dict(torch.load(r'./modelpkl/best_TeaEnmodel.pkl')); TeaEnmodel.eval()
    NfModel.load_state_dict(torch.load(r'./modelpkl/best_NfEnmodel.pkl')); NfModel.eval()
    best = 1e9; best_t = 0
    with tqdm(total=args.num_epoch2,colour='yellow',dynamic_ncols=True, leave=False) as pbar:
            pbar.set_description('[Train StuEncoder]')
            for epoch in range(1, args.num_epoch2 + 1):
                for data in dataloader:
                    data = data.to(device)
                    with torch.no_grad():
                        x = TeaEnmodel(x=torch.cat((data.x, data.x_s), dim=1), edge_index=data.edge_index)
                        if args.use_a:
                            nfoutput, _=NfModel(x=x, edge_index=data.edge_index)
                        else:
                            nfoutput, _=NfModel(x=x, edge_index=zero_edge_index)
                    StuEnoptimizer.zero_grad()
                    stuoutput=StuEnmodel(x=torch.cat((data.x, data.x_s), dim=1), edge_index=data.edge_index)
                    nfoutput=torch.sigmoid(nfoutput)
                    stuoutput=torch.sigmoid(stuoutput)
                    stu_em_node=stuoutput
                    tea_em_node=nfoutput
                    ptr=data.ptr
                    stu_em_graph=split_tensor_by_ptr(stuoutput, ptr, readout=args.readout)
                    tea_em_graph=split_tensor_by_ptr(nfoutput, ptr, readout=args.readout)
                    num_graph=len(stu_em_graph)
                    loss_node = torch.mean(F.mse_loss(stu_em_node, tea_em_node, reduction='mean'))
                    loss_graph=0
                    loss_graph = F.mse_loss(stu_em_graph, tea_em_graph, reduction='mean')
                    loss=args.beta*loss_node+(1-args.beta)*loss_graph
                    loss.backward()
                    StuEnoptimizer.step()
                if  loss < best:
                    best = nf_loss
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(StuEnmodel.state_dict(), r'./modelpkl/best_StuEnmodel.pkl')
                else:
                    cnt_wait += 1
                pbar.set_postfix(loss=loss.detach().cpu().numpy()/num_graph)
                pbar.update()

#%% Test model
    TeaEnmodel.load_state_dict(torch.load(r'./modelpkl/best_TeaEnmodel.pkl'))
    NfModel.load_state_dict(torch.load(r'./modelpkl/best_NfEnmodel.pkl'))
    StuEnmodel.load_state_dict(torch.load(r'./modelpkl/best_StuEnmodel.pkl'))
    TeaEnmodel.eval()
    NfModel.eval()
    StuEnmodel.eval()
    for data in dataloader_test:
        data.to(device)
        with torch.no_grad():
            x = TeaEnmodel(x=torch.cat((data.x, data.x_s), dim=1), edge_index=data.edge_index)
            if args.use_a:
                nfoutput, _ = NfModel(x=x, edge_index=data.edge_index)
            else:
                nfoutput, _ = NfModel(x=x, edge_index=zero_edge_index)
            stuoutput=StuEnmodel(x=torch.cat((data.x, data.x_s), dim=1), edge_index=data.edge_index)
        stu_em_node=torch.sigmoid(stuoutput)
        tea_em_node=torch.sigmoid(nfoutput)
        ptr=data.ptr
        stu_em_graph=split_tensor_by_ptr(stuoutput, ptr, readout=args.readout)
        tea_em_graph=split_tensor_by_ptr(nfoutput, ptr, readout=args.readout)
        #score_node = F.mse_loss(stu_em_node, tea_em_node, reduction='mean')
        score_graph = squared_diff = (stu_em_graph - tea_em_graph).pow(2).mean(dim=1)
        score_graph=torch.sigmoid(score_graph)
        label=data.y.cpu().numpy()
        NF_em_graph=tea_em_graph
        tea2_em_graph=split_tensor_by_ptr(x, ptr, readout=args.readout)
        fpr_ab, tpr_ab, thr_ = roc_curve(y_true=label, y_score=score_graph, drop_intermediate=False)
        AUC = auc(fpr_ab, tpr_ab)
        aucs.append(AUC)
#%% Save result
auc_map=read_auc()
auc_best=auc_map[args.DS]
avg_auc = np.mean(aucs)
std_auc = np.std(aucs)
print('{:.4f}±{:.4f}'.format(avg_auc*100, std_auc*100))

