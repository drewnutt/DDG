import re
import molgrid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch import autograd
import wandb
import argparse
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--ligtr', required=True, help='location of training ligand cache file input')
parser.add_argument('--rectr', required=True,help='location of training receptor cache file input')
parser.add_argument('--trainfile', required=True, help='location of training information, this must have a group indicator')
parser.add_argument('--ligte', required=True, help='location of testing ligand cache file input')
parser.add_argument('--recte', required=True, help='location of testing receptor cache file input')
parser.add_argument('--testfile', required=True, help='location of testing information, this must have a group indicator')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--dropout', '-d',default=0, type=float,help='dropout of layers')
parser.add_argument('--non_lin',choices=['relu','leakyrelu'],default='relu',help='non-linearity to use in the network')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of optimizer')
parser.add_argument('--solver', default="adam", choices=('adam','sgd','lars'), type=str, help="solver to use")
parser.add_argument('--epoch',default=200,type=int,help='number of epochs to train for (default %(default)d)')
parser.add_argument('--tags',default=[],nargs='*',help='tags to use for wandb run')
parser.add_argument('--batch_norm',default=0,choices=[0,1],type=int,help='use batch normalization during the training process')
parser.add_argument('--weight_decay',default=0,type=float,help='weight decay to use with the optimizer')
parser.add_argument('--clip',default=0,type=float,help='keep gradients within [clip]')
parser.add_argument('--binary_rep',default=False,action='store_true',help='use a binary representation of the atoms')
parser.add_argument('--use_model','-m',default='paper',choices=['paper', 'def2018', 'extend_def2018', 'multtask_def2018','ext_mult_def2018', 'multtask_latent_def2018', 'multtask_latent_dense'], help='Network architecture to use')
parser.add_argument('--use_weights','-w',help='pretrained weights to use for the model')
parser.add_argument('--freeze_arms',choices=[0,1],default=0,type=int,help='freeze the weights of the CNN arms of the network (applies after using pretrained weights)')
parser.add_argument('--hidden_size',default=1024,type=int,help='size of fully connected layer before subtraction in latent space')
parser.add_argument('--batch_size',default=16,type=int,help='batch size (default: %(default)d)')
parser.add_argument('--absolute_dg_loss', '-L',action='store_true',default=False,help='use a loss function (and model architecture) that utilizes the absolute binding affinity')
parser.add_argument('--self_supervised_test', '-ST',action='store_true',default=False,help='Use the self supervised loss on the test files (no labels used)')
parser.add_argument('--rotation_loss_weight','-R',default=1.0,type=float,help='weight to use in adding the rotation loss to the other losses (default: %(default)d)')
parser.add_argument('--consistency_loss_weight','-C',default=1.0,type=float,help='weight to use in adding the consistency term to the other losses (default: %(default)d')
parser.add_argument('--absolute_loss_weight','-A',default=1.0,type=float,help='weight to use in adding the absolute loss terms to the other losses (default: %(default)d')
parser.add_argument('--ddg_loss_weight','-D',default=1.0,type=float,help='weight to use in adding the DDG loss terms to the other losses (default: %(default)d')
parser.add_argument('--train_type',default='no_SS', choices=['no_SS','SS_simult_before','SS_simult_after'],help='what type of training loop to use')
parser.add_argument('--latent_loss',default='mse', choices=['mse','corr'],help='what type of loss to apply to the latent representations')
parser.add_argument('--crosscorr_lambda', type=float, help='lambda value to use in the Cross Correlation Loss')
parser.add_argument('--proj_size',type=int,default=4096,help='size to project the latent representation to, this is the dimension that the CrossCorrLoss will be applied to (default: %(default)d')
parser.add_argument('--proj_layers',type=int,default=3,help='how many layers in the projection network, if 0 then there is no projection network(default: %(default)d')
parser.add_argument('--rot_warmup','-RW',default=30,type=int,help='how many epochs to warmup from 0 to your desired weight for rotation loss')
parser.add_argument('--stratify_rec','-S',default=False,action='store_true',help='toggle the example provider stratifying by the receptor')
parser.add_argument('--iter_scheme','-I',choices=['small','large'],default='small',help='what sort of epoch iteration scheme to use')
args = parser.parse_args()

print(args.absolute_dg_loss, args.use_model)
assert (args.absolute_dg_loss and args.use_model in ['multtask_def2018', 'ext_mult_def2018', 'multtask_latent_def2018', 'multtask_latent_dense']) or (not args.absolute_dg_loss and args.use_model in ['paper','def2018','extend_def2018']), 'Cannot have multitask loss with a non-multitask model'

if args.use_model == 'paper':
    from paper_model import Net
elif args.use_model == 'def2018':
    from default2018_model import Net
elif args.use_model == 'extend_def2018':
    from extended_default2018_model import Net
elif args.use_model == 'multtask_def2018':
    from multtask_def2018_model import Net
elif args.use_model == 'multtask_latent_def2018':
    from multtask_latent_def2018_model import Net
elif args.use_model == 'multtask_latent_dense':
    from multtask_latent_dense_model import Dense as Net
elif args.use_model == 'ext_mult_def2018':
    from extended_multtask_def2018_model import Net

def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)

class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

class CrossCorrLoss(nn.Module):    
    def __init__(self, rep_size, lambd, device='cpu'):    
        super(CrossCorrLoss,self).__init__()    
        self.bn = nn.BatchNorm1d(rep_size, affine=False).to(device)    
        self.device = device
        self.lambd = lambd    
        
    def forward(self, z1, z2):    
        z1 = z1.to(self.device)
        z2 = z2.to(self.device)
        c = self.bn(z1).T @ self.bn(z2)    
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()    
            
        n, m = c.shape    
        assert n == m    
        off_diagonals = c.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()    
        off_diag = off_diagonals.pow_(2).sum()    
           
        loss = on_diag + self.lambd * off_diag    
        return loss

class Projector(nn.Module):
    def __init__(self, rep_size, final_dim,layers=3):
        super(Projector, self).__init__()
        self.modules = []

        first_layer = nn.Linear(rep_size,final_dim)
        self.modules.append(first_layer)
        self.add_module('first_proj',first_layer)
        for idx in range(layers-1):
            next_layer = nn.Linear(final_dim,final_dim)
            self.modules.append(next_layer)
            self.add_module(f'proj_{idx}',next_layer)

    def forward(self, x):
        x = self.modules[0](x)
        for layer in self.modules[1:]:
           x = F.relu(x) 
           x = layer(x)
        return x
        

def train(model, traine, test_data, optimizer, latent_rep, epoch, proj=None):
    model.train()
    full_loss, lig_loss, rot_loss, DDG_loss = 0, 0, 0, 0

    output_dist, actual = [], []
    lig_pred, lig_labels = [], []
    for idx, batch in enumerate(traine):
        it = num_exs/args.batch_size * epoch + idx
        gmaker.forward(batch, input_tensor_1, random_translation=2.0, random_rotation=True) 
        gmaker.forward(batch, input_tensor_2, random_translation=2.0, random_rotation=True) 
        batch.extract_label(1, float_labels)
        labels = torch.unsqueeze(float_labels, 1).float().to('cuda')
        optimizer.zero_grad()
        batch.extract_label(2, lig1_label)
        batch.extract_label(3, lig2_label)
        lig1_labels = torch.unsqueeze(lig1_label, 1).float().to('cuda')
        lig2_labels = torch.unsqueeze(lig2_label, 1).float().to('cuda')
        if latent_rep:
            output, lig1, lig2, lig1_rep1, lig2_rep1 = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
            output2, lig1_2, lig2_2, lig1_rep2, lig2_rep2 = model(input_tensor_2[:, :28, :, :, :], input_tensor_2[:, 28:, :, :, :])
            # ddg_lig1, dg1_lig1, dg2_lig1, lig1_rep1, lig1_rep2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            # ddg_lig2, dg1_lig2, dg2_lig2, lig2_rep1, lig2_rep2  = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            if proj:
                lig1_rep1 = proj(lig1_rep1)
                lig1_rep2 = proj(lig1_rep2)
                lig2_rep1 = proj(lig2_rep1)
                lig2_rep2 = proj(lig2_rep2)
            rotation_loss = dgrotloss1(lig1_rep1, lig1_rep2)
            rotation_loss += dgrotloss2(lig2_rep1, lig2_rep2)
        else:
            output, lig1, lig2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
            ddg_lig1, dg1_lig1, dg2_lig1 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            ddg_lig2, dg1_lig2, dg2_lig2 = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            rotation_loss = dgrotloss1(dg1_lig1, dg2_lig1)
            rotation_loss += dgrotloss2(dg1_lig2, dg2_lig2)
            rotation_loss += ddgrotloss1(ddg_lig1, torch.zeros(ddg_lig1.size(), device='cuda:0')) 
            rotation_loss += ddgrotloss2(ddg_lig2, torch.zeros(ddg_lig1.size(), device='cuda:0')) 
        loss_lig1 = criterion_lig1(lig1, lig1_labels)
        loss_lig2 = criterion_lig2(lig2, lig2_labels)
        ddg_loss = criterion(output, labels)
        loss = args.absolute_loss_weight * (loss_lig1 + loss_lig2) + args.ddg_loss_weight * ddg_loss + args.rotation_loss_weight[it] * rotation_loss + args.consistency_loss_weight * nn.functional.mse_loss((lig1-lig2), output)
        lig_pred += lig1.flatten().tolist() + lig2.flatten().tolist()
        lig_labels += lig1_labels.flatten().tolist() + lig2_labels.flatten().tolist()
        lig_loss += loss_lig1 + loss_lig2
        rot_loss += rotation_loss
        DDG_loss += ddg_loss
        full_loss += loss
        loss.backward()
        if args.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        optimizer.step()
        output_dist += output.flatten().tolist()
        actual += labels.flatten().tolist()

    total_samples = (idx + 1) * len(batch)
    try:
        r, _=pearsonr(np.array(actual),np.array(output_dist))
    except ValueError as e:
        print('{}:{}'.format(epoch,e))
        r=np.nan
    try:
        rligs,_=pearsonr(np.array(lig_pred),np.array(lig_labels))
        temp = r
        r = (temp,rligs)
    except ValueError as e:
        print(f'{epoch}:{e}')
        tmp = r
        r = (tmp,np.nan)
    rmse = np.sqrt(((np.array(output_dist)-np.array(actual)) ** 2).mean())
    avg_loss = full_loss/(total_samples)
    avg_lig_loss = lig_loss / (2*total_samples)
    avg_rot_loss = rot_loss / (total_samples)
    avg_DDG_loss = DDG_loss / (total_samples)
    tmp = avg_loss
    avg_loss = (tmp,avg_lig_loss,avg_DDG_loss,avg_rot_loss,None)
    rmse_ligs = np.sqrt(((np.array(lig_pred)-np.array(lig_labels)) ** 2).mean())
    tmp = rmse
    rmse = (rmse, rmse_ligs)
    both_calc_distr = (output_dist,lig_pred)
    both_labels = (actual,lig_labels)
    return avg_loss, both_calc_distr, r, rmse, both_labels


def train_ss_after(model, train_data,test_data, optimizer, latent_rep):
    model.train()
    full_loss, lig_loss, rot_loss, DDG_loss, ss_loss = 0, 0, 0, 0, 0

    output_dist, actual = [], []
    lig_pred, lig_labels = [], []
    for idx, train_batch in enumerate(train_data):
        #now do a batch of actual training with labels
        gmaker.forward(train_batch, input_tensor_1, random_translation=2.0, random_rotation=True) 
        gmaker.forward(train_batch, input_tensor_2, random_translation=2.0, random_rotation=True) 
        train_batch.extract_label(1, float_labels)
        labels = torch.unsqueeze(float_labels, 1).float().to('cuda')
        optimizer.zero_grad()
        train_batch.extract_label(2, lig1_label)
        train_batch.extract_label(3, lig2_label)
        lig1_labels = torch.unsqueeze(lig1_label, 1).float().to('cuda')
        lig2_labels = torch.unsqueeze(lig2_label, 1).float().to('cuda')
        if latent_rep:
            train_output, lig1, lig2, lig1_rep, lig2_rep = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
            ddg_lig1, dg1_lig1, dg2_lig1, lig1_selfrep1, lig1_selfrep2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            ddg_lig2, dg1_lig2, dg2_lig2, lig2_selfrep1, lig2_selfrep2  = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            rotation_loss = dgrotloss1(lig1_selfrep1, lig1_selfrep2)
            rotation_loss += dgrotloss2(lig2_selfrep1, lig2_selfrep2)
        else:
            train_output, lig1, lig2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
            ddg_lig1, dg1_lig1, dg2_lig1 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            ddg_lig2, dg1_lig2, dg2_lig2 = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            rotation_loss = dgrotloss1(dg1_lig1, dg2_lig1)
            rotation_loss += dgrotloss2(dg1_lig2, dg2_lig2)
        loss_lig1 = criterion_lig1(lig1, lig1_labels)
        loss_lig2 = criterion_lig2(lig2, lig2_labels)
        ddg_loss = criterion(train_output, labels)
        rotation_loss += ddgrotloss1(ddg_lig1, torch.zeros(ddg_lig1.size(), device='cuda')) 
        rotation_loss += ddgrotloss2(ddg_lig2, torch.zeros(ddg_lig1.size(), device='cuda')) 
        loss = args.absolute_loss_weight * (loss_lig1 + loss_lig2) + args.ddg_loss_weight * ddg_loss + args.rotation_loss_weight * rotation_loss + args.consistency_loss_weight * nn.functional.mse_loss((lig1-lig2), train_output)
        lig_pred += lig1.flatten().tolist() + lig2.flatten().tolist()
        lig_labels += lig1_labels.flatten().tolist() + lig2_labels.flatten().tolist()
        lig_loss += loss_lig1 + loss_lig2
        rot_loss += rotation_loss
        DDG_loss += ddg_loss
        full_loss += loss
        loss.backward()
        if args.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        optimizer.step()

        # Now do a batch of the self-supervised training on the unlabelled test data
        optimizer.zero_grad()
        test_batch = test_data.next_batch()
        gmaker.forward(test_batch, input_tensor_1, random_translation=2.0, random_rotation=True) 
        gmaker.forward(test_batch, input_tensor_2, random_translation=2.0, random_rotation=True) 
        if latent_rep:
            output, lig1, lig2, lig1_rep, lig2_rep = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
            ddg_lig1, dg1_lig1, dg2_lig1, lig1_selfrep1, lig1_selfrep2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            ddg_lig2, dg1_lig2, dg2_lig2, lig2_selfrep1, lig2_selfrep2  = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            self_super_loss = dgrotloss1(lig1_selfrep1, lig1_selfrep2)
            self_super_loss += dgrotloss2(lig2_selfrep1, lig2_selfrep2)
        else:
            output, lig1, lig2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
            ddg_lig1, dg1_lig1, dg2_lig1 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            ddg_lig2, dg1_lig2, dg2_lig2 = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            self_super_loss = dgrotloss1(dg1_lig1, dg2_lig1)
            self_super_loss += dgrotloss2(dg1_lig2, dg2_lig2)
        self_super_loss += ddgrotloss1(ddg_lig1, torch.zeros(ddg_lig1.size(), device='cuda')) 
        self_super_loss += ddgrotloss2(ddg_lig2, torch.zeros(ddg_lig1.size(), device='cuda')) 
        ss_loss += self_super_loss 
        self_super_loss.backward()
        if args.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        optimizer.step()

        output_dist += train_output.flatten().tolist()
        actual += labels.flatten().tolist()

    total_samples = (idx + 1) * len(train_batch)
    try:
        r, _=pearsonr(np.array(actual),np.array(output_dist))
    except ValueError as e:
        print('{}:{}'.format(epoch,e))
        r=np.nan
    try:
        rligs,_=pearsonr(np.array(lig_pred),np.array(lig_labels))
        temp = r
        r = (temp,rligs)
    except ValueError as e:
        print(f'{epoch}:{e}')
        tmp = r
        r = (tmp,np.nan)
    rmse = np.sqrt(((np.array(output_dist)-np.array(actual)) ** 2).mean())
    avg_loss = full_loss/(total_samples)
    avg_lig_loss = lig_loss / (2*total_samples)
    avg_rot_loss = rot_loss / (total_samples)
    avg_DDG_loss = DDG_loss / (total_samples)
    avg_ss_loss = ss_loss / total_samples
    tmp = avg_loss
    avg_loss = (tmp, avg_lig_loss, avg_DDG_loss, avg_rot_loss, avg_ss_loss)
    rmse_ligs = np.sqrt(((np.array(lig_pred)-np.array(lig_labels)) ** 2).mean())
    tmp = rmse
    rmse = (rmse, rmse_ligs)
    both_calc_distr = (output_dist, lig_pred)
    both_labels = (actual, lig_labels)
    return avg_loss, both_calc_distr, r, rmse, both_labels

def train_ss_before(model, train_data,test_data, optimizer, latent_rep):
    model.train()
    full_loss, lig_loss, rot_loss, DDG_loss, ss_loss = 0, 0, 0, 0, 0

    output_dist, actual = [], []
    lig_pred, lig_labels = [], []
    for idx, train_batch in enumerate(train_data):
        # Now do a batch of the self-supervised training on the unlabelled test data
        optimizer.zero_grad()
        test_batch = test_data.next_batch()
        gmaker.forward(test_batch, input_tensor_1, random_translation=2.0, random_rotation=True) 
        gmaker.forward(test_batch, input_tensor_2, random_translation=2.0, random_rotation=True) 
        if latent_rep:
            output, lig1, lig2, lig1_rep, lig2_rep = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
            ddg_lig1, dg1_lig1, dg2_lig1, lig1_selfrep1, lig1_selfrep2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            ddg_lig2, dg1_lig2, dg2_lig2, lig2_selfrep1, lig2_selfrep2  = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            self_super_loss = dgrotloss1(lig1_selfrep1, lig1_selfrep2)
            self_super_loss += dgrotloss2(lig2_selfrep1, lig2_selfrep2)
        else:
            output, lig1, lig2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
            ddg_lig1, dg1_lig1, dg2_lig1 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            ddg_lig2, dg1_lig2, dg2_lig2 = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            self_super_loss = dgrotloss1(dg1_lig1, dg2_lig1)
            self_super_loss += dgrotloss2(dg1_lig2, dg2_lig2)
        self_super_loss += ddgrotloss1(ddg_lig1, torch.zeros(ddg_lig1.size(), device='cuda')) 
        self_super_loss += ddgrotloss2(ddg_lig2, torch.zeros(ddg_lig1.size(), device='cuda')) 
        ss_loss += self_super_loss 
        self_super_loss.backward()
        if args.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        optimizer.step()

        #now do a batch of actual training with labels
        gmaker.forward(train_batch, input_tensor_1, random_translation=2.0, random_rotation=True) 
        gmaker.forward(train_batch, input_tensor_2, random_translation=2.0, random_rotation=True) 
        train_batch.extract_label(1, float_labels)
        labels = torch.unsqueeze(float_labels, 1).float().to('cuda')
        optimizer.zero_grad()
        train_batch.extract_label(2, lig1_label)
        train_batch.extract_label(3, lig2_label)
        lig1_labels = torch.unsqueeze(lig1_label, 1).float().to('cuda')
        lig2_labels = torch.unsqueeze(lig2_label, 1).float().to('cuda')
        if latent_rep:
            output, lig1, lig2, lig1_rep, lig2_rep = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
            ddg_lig1, dg1_lig1, dg2_lig1, lig1_selfrep1, lig1_selfrep2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            ddg_lig2, dg1_lig2, dg2_lig2, lig2_selfrep1, lig2_selfrep2  = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            rotation_loss = dgrotloss1(lig1_selfrep1, lig1_selfrep2)
            rotation_loss += dgrotloss2(lig2_selfrep1, lig2_selfrep2)
        else:
            output, lig1, lig2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
            ddg_lig1, dg1_lig1, dg2_lig1 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            ddg_lig2, dg1_lig2, dg2_lig2 = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            rotation_loss = dgrotloss1(dg1_lig1, dg2_lig1)
            rotation_loss += dgrotloss2(dg1_lig2, dg2_lig2)
        loss_lig1 = criterion_lig1(lig1, lig1_labels)
        loss_lig2 = criterion_lig2(lig2, lig2_labels)
        ddg_loss = criterion(output, labels)
        rotation_loss += ddgrotloss1(ddg_lig1, torch.zeros(ddg_lig1.size(), device='cuda')) 
        rotation_loss += ddgrotloss2(ddg_lig2, torch.zeros(ddg_lig1.size(), device='cuda')) 
        loss = args.absolute_loss_weight * (loss_lig1 + loss_lig2) + args.ddg_loss_weight * ddg_loss + args.rotation_loss_weight * rotation_loss + args.consistency_loss_weight * nn.functional.mse_loss((lig1-lig2), output)
        lig_pred += lig1.flatten().tolist() + lig2.flatten().tolist()
        lig_labels += lig1_labels.flatten().tolist() + lig2_labels.flatten().tolist()
        lig_loss += loss_lig1 + loss_lig2
        rot_loss += rotation_loss
        DDG_loss += ddg_loss
        full_loss += loss
        loss.backward()
        if args.clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        optimizer.step()



        output_dist += output.flatten().tolist()
        actual += labels.flatten().tolist()

    total_samples = (idx + 1) * len(train_batch)
    try:
        r, _=pearsonr(np.array(actual),np.array(output_dist))
    except ValueError as e:
        print('{}:{}'.format(epoch,e))
        r=np.nan
    try:
        rligs,_=pearsonr(np.array(lig_pred),np.array(lig_labels))
        temp = r
        r = (temp,rligs)
    except ValueError as e:
        print(f'{epoch}:{e}')
        tmp = r
        r = (tmp,np.nan)
    rmse = np.sqrt(((np.array(output_dist)-np.array(actual)) ** 2).mean())
    avg_loss = full_loss/(total_samples)
    avg_lig_loss = lig_loss / (2*total_samples)
    avg_rot_loss = rot_loss / (total_samples)
    avg_DDG_loss = DDG_loss / (total_samples)
    avg_ss_loss = ss_loss / total_samples
    tmp = avg_loss
    avg_loss = (tmp, avg_lig_loss, avg_DDG_loss, avg_rot_loss, avg_ss_loss)
    rmse_ligs = np.sqrt(((np.array(lig_pred)-np.array(lig_labels)) ** 2).mean())
    tmp = rmse
    rmse = (rmse, rmse_ligs)
    both_calc_distr = (output_dist, lig_pred)
    both_labels = (actual, lig_labels)
    return avg_loss, both_calc_distr, r, rmse, both_labels

def train_rotation(model, ss_data, optimizer, latent_rep):
    model.train()
    full_loss = 0

    for idx, batch in enumerate(ss_data):
        gmaker.forward(batch, input_tensor_1, random_translation=2.0, random_rotation=True) 
        gmaker.forward(batch, input_tensor_2, random_translation=2.0, random_rotation=True) 
        optimizer.zero_grad()
        if latent_rep:
            ddg_lig1, dg1_lig1, dg2_lig1, lig1_selfrep1, lig1_selfrep2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            ddg_lig2, dg1_lig2, dg2_lig2, lig2_selfrep1, lig2_selfrep2 = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            rotation_loss = dgrotloss1(lig1_selfrep1, lig1_selfrep2)
            rotation_loss += dgrotloss2(lig2_selfrep1, lig2_selfrep2)
        else:
            ddg_lig1, dg1_lig1, dg2_lig1 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
            ddg_lig2, dg1_lig2, dg2_lig2 = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
            rotation_loss = dgrotloss1(dg1_lig1, dg2_lig1)
            rotation_loss += dgrotloss2(dg1_lig2, dg2_lig2)
        rotation_loss += ddgrotloss1(ddg_lig1, torch.zeros(ddg_lig1.size(), device='cuda'))
        rotation_loss += ddgrotloss2(ddg_lig2, torch.zeros(ddg_lig1.size(), device='cuda'))
        loss = args.rotation_loss_weight * rotation_loss
        full_loss += loss
        loss.backward()
        optimizer.step()

    total_samples = (idx + 1) * len(batch)
    avg_loss = full_loss/(total_samples)
    return avg_loss

def test(model, test_data, latent_rep, epoch, proj=None):
    model.eval()
    test_loss, lig_loss, rot_loss, DDG_loss = 0, 0, 0, 0

    output_dist, actual = [], []
    lig_pred, lig_labels = [], []
    with torch.no_grad():
        for idx, batch in enumerate(test_data):        
            it = num_exs/args.batch_size * epoch + idx
            gmaker.forward(batch, input_tensor_1, random_translation=2.0, random_rotation=True) 
            gmaker.forward(batch, input_tensor_2, random_translation=2.0, random_rotation=True) 
            batch.extract_label(1, float_labels)
            labels = torch.unsqueeze(float_labels, 1).float().to('cuda')
            optimizer.zero_grad()
            batch.extract_label(2, lig1_label)
            batch.extract_label(3, lig2_label)
            lig1_labels = torch.unsqueeze(lig1_label, 1).float().to('cuda')
            lig2_labels = torch.unsqueeze(lig2_label, 1).float().to('cuda')
            if latent_rep:
                output, lig1, lig2, lig1_rep1, lig2_rep1 = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
                output2, lig1_2, lig2_2, lig1_rep2, lig2_rep2 = model(input_tensor_2[:, :28, :, :, :], input_tensor_2[:, 28:, :, :, :])
                # ddg_lig1, dg1_lig1, dg2_lig1, lig1_selfrep1, lig1_selfrep2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
                # ddg_lig2, dg1_lig2, dg2_lig2, lig2_selfrep1, lig2_selfrep2  = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
                if proj:
                    lig1_rep1 = proj(lig1_rep1)
                    lig1_rep2 = proj(lig1_rep2)
                    lig2_rep1 = proj(lig2_rep1)
                    lig2_rep2 = proj(lig2_rep2)
                rotation_loss = dgrotloss1(lig1_rep1, lig1_rep2)
                rotation_loss += dgrotloss2(lig2_rep1, lig2_rep2)
            else:
                output, lig1, lig2 = model(input_tensor_1[:, :28, :, :, :], input_tensor_1[:, 28:, :, :, :])
                ddg_lig1, dg1_lig1, dg2_lig1 = model(input_tensor_1[:, :28, :, :, :], input_tensor_2[:, :28, :, :, :]) #Same rec-lig pair input to both arms, just rotated/translated differently
                ddg_lig2, dg1_lig2, dg2_lig2 = model(input_tensor_1[:, 28:, :, :, :], input_tensor_2[:, 28:, :, :, :]) #Repeated for the second ligand
                rotation_loss = dgrotloss1(dg1_lig1, dg2_lig1)
                rotation_loss += dgrotloss2(dg1_lig2, dg2_lig2)
                zero_tensor = torch.zeros(ddg_lig1.size()).to("cuda:0")
                rotation_loss += ddgrotloss1(ddg_lig1, zero_tensor) 
                rotation_loss += ddgrotloss2(ddg_lig2, zero_tensor) 
            loss_lig1 = criterion_lig1(lig1, lig1_labels)
            loss_lig2 = criterion_lig2(lig2, lig2_labels)
            ddg_loss = criterion(output, labels)
            loss = args.absolute_loss_weight * (loss_lig1 + loss_lig2) + args.ddg_loss_weight * ddg_loss + args.rotation_loss_weight[it] * rotation_loss + args.consistency_loss_weight * nn.functional.mse_loss((lig1-lig2),output)
            lig_pred += lig1.flatten().tolist() + lig2.flatten().tolist()
            lig_labels += lig1_labels.flatten().tolist() + lig2_labels.flatten().tolist()
            lig_loss += loss_lig1 + loss_lig2
            rot_loss += rotation_loss
            DDG_loss += ddg_loss
            test_loss += loss
            output_dist += output.flatten().tolist()
            actual += labels.flatten().tolist()

    total_samples = (idx + 1) * len(batch) 

    try:
        r,_=pearsonr(np.array(actual),np.array(output_dist))
    except ValueError as e:
        print('{}:{}'.format(epoch,e))
        r=np.nan
    try:
        rligs, _=pearsonr(np.array(lig_pred), np.array(lig_labels))
        temp = r
        r = (temp,rligs)
    except ValueError as e:
        print(f'{epoch}:{e}')
        tmp = r
        r = (tmp,np.nan)
    rmse = np.sqrt(((np.array(output_dist)-np.array(actual)) ** 2).mean())
    avg_loss = float(test_loss)/(total_samples)
    avg_lig_loss = float(lig_loss) / (2*total_samples)
    avg_rot_loss = float(rot_loss) / (total_samples)
    avg_DDG_loss = float(DDG_loss) / (total_samples)
    tmp = avg_loss
    avg_loss = (tmp, avg_lig_loss, avg_DDG_loss, avg_rot_loss)
    rmse_ligs = np.sqrt(((np.array(lig_pred)-np.array(lig_labels)) ** 2).mean())
    tmp = rmse
    rmse = (rmse, rmse_ligs)
    both_calc_distr = (output_dist, lig_pred)
    both_labels = (actual, lig_labels)
    return avg_loss, both_calc_distr, r, rmse, both_labels


# Make helper function to make meaningful tags
def make_tags(args):
    addnl_tags = []
    addnl_tags.append(args.use_model)
    if 'full_bdb' in args.ligtr:
        addnl_tags.append('full_BDB')
    addnl_tags.append(args.train_type)
    addnl_tags.append(f'{args.latent_loss.title()}Loss')
    return addnl_tags


tgs = make_tags(args) + args.tags
wandb.init(entity='andmcnutt', project='DDG_model_Regression',config=args, tags=tgs)

if args.use_model == 'multtask_latent_def2018' or args.use_model == 'multtask_latent_dense':
    latent_rep = True
else:
    latent_rep = False
#Parameters that are not important for hyperparameter sweep
batch_size = args.batch_size
epochs = args.epoch

# print('ligtr={}, rectr={}'.format(args.ligtr,args.rectr))

iter_scheme = molgrid.IterationScheme.SmallEpoch
if args.iter_scheme == 'large':
    iter_scheme = molgrid.IterationScheme.LargeEpoch
traine = molgrid.ExampleProvider(ligmolcache=args.ligtr, recmolcache=args.rectr, stratify_receptor=args.stratify_rec, shuffle=True, duplicate_first=True, default_batch_size=batch_size, iteration_scheme=iter_scheme)
traine.populate(args.trainfile)
teste = molgrid.ExampleProvider(ligmolcache=args.ligte, recmolcache=args.recte, shuffle=True, duplicate_first=True, default_batch_size=batch_size, iteration_scheme=molgrid.IterationScheme.LargeEpoch)
teste.populate(args.testfile)
ss_test = molgrid.ExampleProvider(ligmolcache=args.ligte, recmolcache=args.recte, shuffle=True, duplicate_first=True, default_batch_size=batch_size, iteration_scheme=molgrid.IterationScheme.LargeEpoch)
ss_test.populate(args.testfile)

gmaker = molgrid.GridMaker(binary=args.binary_rep)
dims = gmaker.grid_dimensions(14*4)  # only one rec+onelig per example
tensor_shape = (batch_size,)+dims

actual_dims = (dims[0]//2, *dims[1:])
model = Net(actual_dims,args)
if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
else:
        print('GPUS: {}'.format(torch.cuda.device_count()), flush=True)
model.to('cuda')
print('done moving model')
model.apply(weights_init)
print('applied weights')

# Setup the training regimen'no_SS','SS_simult_before','SS_simult_after'
train_func = None
if args.train_type == 'no_SS':
    train_func = train
elif args.train_type == 'SS_simult_before':
    train_func = train_ss_before
elif args.train_type == 'SS_simult_after':
    train_func = train_ss_after
else:
    print('train_func was never set, something wrong with args.train_type')
print('setup train')

if args.use_weights is not None:  # using the weights from an external source, only some of the network layers need to be the same
    print('about to use weights')
    pretrained_state_dict = torch.load(args.use_weights)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
if args.freeze_arms:
    for name,param in model.named_parameters():
        if 'conv' in name:
            print(name)
            param.requires_grad = False
    
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
criterion_lig1 = nn.MSELoss()
criterion_lig2 = nn.MSELoss()
# Not sure if these are replaceable with the Barlow Twins loss
ddgrotloss1 = nn.MSELoss()
ddgrotloss2 = nn.MSELoss()
projector = None
if args.latent_loss == 'mse':
    dgrotloss1 = nn.MSELoss()
    dgrotloss2 = nn.MSELoss()
elif latent_rep: ## only other option is 'covar' for the Barlow Twins approach, but requires latent rep
    _,_,_,rep1,rep2 = model(torch.rand(tensor_shape, device='cuda')[:, :28, :, :, :], torch.rand(tensor_shape,device='cuda')[:, 28:, :, :, :])
    init_size = rep1.shape[-1]
    assert init_size == rep2.shape[-1]
    proj_size = args.proj_size 
    if args.proj_layers:
        projector = Projector(init_size,proj_size,args.proj_layers)
        projector.to('cuda')
        projector.apply(weights_init)
        print(init_size,proj_size)
    else:
        proj_size = init_size
        print("not using any projector")
    crosscorr_lambda = 1.0/proj_size
    if args.crosscorr_lambda:
        crosscorr_lambda = args.crosscorr_lambda
    else:
       wandb.config.update({"crosscorr_lambda": crosscorr_lambda},allow_val_change=True) 
    dgrotloss1 = CrossCorrLoss(proj_size,crosscorr_lambda,device='cuda')
    dgrotloss2 = CrossCorrLoss(proj_size,crosscorr_lambda,device='cuda')
params = [param for param in model.parameters()]
if projector:
    print("Adding projector to optimizer parameters")
    params += [param for param in projector.parameters()]
optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
if args.solver == "sgd":
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.solver == 'lars':
    optimizer = LARS(params, lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)

num_exs = traine.small_epoch_size()
if args.iter_scheme == 'large':
    num_exs = traine.large_epoch_size()
if args.rot_warmup:
    num_iters = num_exs/args.batch_size * args.rot_warmup
    print(num_iters)
    warmup_schedule = np.linspace(0,args.rotation_loss_weight,int(num_iters) )
    final_iters = num_exs/args.batch_size * (args.epoch-args.rot_warmup)
    args.rotation_loss_weight = np.concatenate((warmup_schedule,np.full((final_iters,),args.rotation_loss_weight)))
else:
    all_iters = num_exs/args.batch_size * (args.epoch)
    args.rotation_loss_weight = np.full((all_iters,),args.rotation_loss_weight)



scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, threshold=0.001, patience=20, verbose=True)

input_tensor_1 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
input_tensor_2 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
float_labels = torch.zeros(batch_size, dtype=torch.float32)
lig1_label = torch.zeros(batch_size, dtype=torch.float32)
lig2_label = torch.zeros(batch_size, dtype=torch.float32)


wandb.watch(model, log='all')
# max_rot_weight = args.rotation_loss_weight
print('training now')
## I want to see how the model is doing on the test before even training, mostly for the pretrained models
tt_loss, out_d, tt_r, tt_rmse, tt_act = test(model, teste, latent_rep,1,proj=projector)
print(f'Before Training at all:\n\tTest Loss: {tt_loss}\n\tTest R:{tt_r}\n\tTest RMSE:{tt_rmse}')
for epoch in range(1, epochs+1):
    # if args.self_supervised_test:
    #     ss_loss = train_rotation(model, teste, optimizer, latent_rep)
    # tr_loss, out_dist, tr_r, tr_rmse, tr_act = train(model, traine, optimizer, latent_rep)
    # args.rotation_loss_weight = max_rot_weight * epoch/epochs
    tr_loss, out_dist, tr_r, tr_rmse, tr_act = train_func(model, traine, ss_test, optimizer, latent_rep, epoch, proj=projector)
    tt_loss, out_d, tt_r, tt_rmse, tt_act = test(model, teste, latent_rep, epoch, proj=projector)
    if args.absolute_dg_loss:
        scheduler.step(tr_loss[0])
    else:
        scheduler.step(tr_loss)
    
    wandb.log({"Output Distribution Train": wandb.Histogram(np.array(out_dist[0]))}, commit=False)
    wandb.log({"Output Distribution Test": wandb.Histogram(np.array(out_d[0]))}, commit=False)
    if epoch % 10 == 0: # only log the graphs every 10 epochs, make things a bit faster
        fig = plt.figure(1)
        fig.clf()
        plt.scatter(tr_act[0], out_dist[0])
        plt.xlabel('Actual DDG')
        plt.ylabel('Predicted DDG')
        wandb.log({"Actual vs. Predicted DDG (Train)": fig}, commit=False)
        test_fig = plt.figure(2)
        test_fig.clf()
        plt.scatter(tt_act[0], out_d[0])  # the first one is the ddg
        plt.xlabel('Actual DDG')
        plt.ylabel('Predicted DDG')
        wandb.log({"Actual vs. Predicted DDG (Test)": test_fig}, commit=False)
        train_absaff_fig = plt.figure(3)
        train_absaff_fig.clf()
        plt.scatter(tr_act[1],out_dist[1])
        plt.xlabel('Actual affinity')
        plt.ylabel('Predicted affinity')
        wandb.log({"Actual vs. Predicted Affinity (Train)": train_absaff_fig}, commit=False)
        test_absaff_fig = plt.figure(4)
        test_absaff_fig.clf()
        plt.scatter(tt_act[1],out_d[1])
        plt.xlabel('Actual affinity')
        plt.ylabel('Predicted affinity')
        wandb.log({"Actual vs. Predicted Affinity (Test)": test_absaff_fig}, commit=False)

    print(f'Test/Train AbsAff R:{tt_r[1]:.4f}\t{tr_r[1]:.4f}')
    wandb.log({
       "Avg Train Loss Total": tr_loss[0],
       "Avg Test Loss Total": tt_loss[0],
       "Train R": tr_r[0],
       "Test R": tt_r[0],
       "Train RMSE": tr_rmse[0],
       "Test RMSE": tt_rmse[0],
       "Avg Train Loss AbsAff": tr_loss[1],
       "Avg Test Loss AbsAff": tt_loss[1],
       "Avg Train Loss DDG": tr_loss[2],
       "Avg Test Loss DDG": tt_loss[2],
       "Avg Train Loss Rotation": tr_loss[3],
       "Avg Self-Supervised Train Loss Rotation": tr_loss[4],
       "Avg Test Loss Rotation": tt_loss[3],
       "Train R AbsAff": float(tr_r[1]),
       "Test R AbsAff": float(tt_r[1]),
       "Train RMSE AbsAff": tr_rmse[1],
       "Test RMSE AbsAff": tt_rmse[1]})
    if not epoch % 50:
            torch.save(model.state_dict(), "model.h5")
            wandb.save('model.h5')
torch.save(model.state_dict(), "model.h5")
wandb.save('model.h5')
# print("Final Train Distribution: Mean={:.4f}, Var={:.4f}".format(np.mean(out_dist),np.var(out_dist)))
# print("Final Test Distribution: Mean={:.4f}, Var={:.4f}".format(np.mean(out_d),np.var(out_d)))
