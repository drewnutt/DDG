import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
        def __init__(self, dims,config):
                super(Net, self).__init__()
                self.modules = []
                self.residuals = []
                nchannels = dims[0]
                ksize = 3

                #select activation function
                self.func = F.relu

                self.conv1 = nn.Conv3d(nchannels, 32, kernel_size=ksize)
                self.conv2 = nn.Conv3d(32,32,kernel_size=ksize)
                self.max1 = nn.MaxPool3d(3)
                
                self.conv3 = nn.Conv3d(32,3,kernel_size=ksize)
                self.lin1 = nn.Linear(5184,1,bias=False)
                self.dg = nn.Linear(5184,1)
                self.dp1 = nn.Dropout(p=config.dropout)


        def forward_one(self, x): #should approximate the affinity of the receptor/ligand pair
                x= self.func(self.conv1(x))
                x= self.func(self.conv2(x))
                x= self.max1(x)

                x= self.func(self.conv3(x))

                x= x.view(x.shape[0], -1)       
                return x

        def forward(self,x1,x2):
                lig1 = self.forward_one(x1)
                lig2 = self.forward_one(x2)
                lig1_aff = self.dg(lig1)
                lig2_aff = self.dg(lig2)
                diff = self.dp1(lig1 - lig2)
                return self.lin1(diff), lig1_aff, lig2_aff, lig1, lig2
