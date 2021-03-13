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
parser.add_argument('--trainfile', required=True, help='location of training information')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of optimizer')
parser.add_argument('--solver', default="adam", choices=('adam','sgd'), type=str, help="solver to use")
parser.add_argument('--epoch',default=500,type=int,help='number of epochs to train for (default %(default)d)')
parser.add_argument('--tags',default=[],nargs='*',help='tags to use for wandb run')
parser.add_argument('--weight_decay',default=0,type=float,help='weight decay to use with the optimizer')
parser.add_argument('--temperature','-t',default=0.5,type=float,help='temperature parameter of the NT-Xent Loss')
parser.add_argument('--binary_rep',default=False,action='store_true',help='use a binary representation of the atoms')
args = parser.parse_args()

from embedding_encoder import Net

class ContrastiveLoss(nn.Module):      
    def __init__(self, batch_size, temperature=0.5):      
        super().__init__()      
        self.batch_size = int(batch_size)      
        self.temperature = torch.tensor(temperature)
        self.negatives_mask = ((~torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):      
        """      
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs      
        z_i, z_j as per SimCLR paper      
        """      
        representations = torch.cat([F.normalize(emb_i, dim=1),F.normalize(emb_j, dim=1)], dim=0)      
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      

        sim_ij = torch.diag(similarity_matrix, self.batch_size)      
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)      

        loss_partial = -torch.log(torch.exp(torch.cat([sim_ij, sim_ji], dim=0) / self.temperature) / torch.sum(self.negatives_mask * torch.exp(similarity_matrix / self.temperature), dim=1))      
        return torch.sum(loss_partial) / (2 * self.batch_size)

def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)

def train(model, train_data, optimizer):
    model.train()
    full_loss = 0

    for idx, batch in enumerate(train_data):
        gmaker.forward(batch, input_tensor_1, random_translation=2.0, random_rotation=True) 
        gmaker.forward(batch, input_tensor_2, random_translation=2.0, random_rotation=True) 
        optimizer.zero_grad()
        proj_1 = model(input_tensor_1)
        proj_2 = model(input_tensor_2)
        loss = criterion(proj_1.to('cpu'), proj_2.to('cpu'))
        loss.backward()
        optimizer.step()

    total_samples = (idx + 1) * len(batch)
    avg_loss = full_loss / total_samples
    return avg_loss

# Make helper function to make meaningful tags
def make_tags(args):
    addnl_tags = []
    addnl_tags.append(args.use_model)
    if 'full_bdb' in args.ligtr:
        addnl_tags.append('full_BDB')
    addnl_tags.append(args.train_type)
    return addnl_tags


tgs = ['LearningReps'] + args.tags
# wandb.init(entity='andmcnutt', project='DDG_model_Regression',config=args, tags=tgs)

#Parameters that are not important for hyperparameter sweep
batch_size = 2
epochs = args.epoch

# print('ligtr={}, rectr={}'.format(args.ligtr,args.rectr))



traine = molgrid.ExampleProvider(ligmolcache=args.ligtr, recmolcache=args.rectr, shuffle=True, default_batch_size=batch_size, iteration_scheme=molgrid.IterationScheme.SmallEpoch)
traine.populate(args.trainfile)
# To compute the "average" pearson R per receptor, count the number of pairs for each rec then iterate over that number later during test time
# test_exs_per_rec=dict()
# with open(args.testfile) as test_types:
#     count = 0
#     rec = ''
#     for lineuse a loss function (and model architecture) that utilizes the absolute binding affinity in test_types:
#         line_args = line.split(' ')
#         newrec = re.findall(r'([A-Z0-9]{4})/',line_args[4])[0]
#         if newrec != rec:
#             if count > 0:
#                 test_exs_per_rec[rec] = count
#                 count = 1
#             rec = newrec
#         else:
#             count += 1

gmaker = molgrid.GridMaker(binary=args.binary_rep)
dims = gmaker.grid_dimensions(14*2)  # only one rec+onelig per example
tensor_shape = (batch_size,)+dims

actual_dims = (dims[0], *dims[1:])
model = Net(actual_dims)
model.to('cuda:0')
model.apply(weights_init)

# if args.use_weights is not None:  # using the weights from an external source, only some of the network layers need to be the same
#     print('about to use weights')
#     pretrained_state_dict = torch.load(args.use_weights)
#     model_dict = model.state_dict()
#     pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)
# if args.freeze_arms:
#     for name,param in model.named_parameters():
#         if 'conv' in name:
#             print(name)
#             param.requires_grad = False
    

optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.solver == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = ContrastiveLoss(batch_size, args.temperature)


input_tensor_1 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
input_tensor_2 = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')

# wandb.watch(model, log='all')
print('training now')
for epoch in range(1, epochs+1):
    tr_loss = train(model, traine, optimizer)

    # wandb.log({
    #     "Avg Train Loss AbsAff": tr_loss})
    if not epoch % 50:
            torch.save(encoder.state_dict(), "model.h5")
            # wandb.save('model.h5')
torch.save(model.state_dict(), "model.h5")
# wandb.save('model.h5')
