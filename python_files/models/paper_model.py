import torch.nn as nn
import torch.nn.functional as F

class View(nn.Module):
        def __init__(self,shape):
                super(View, self).__init__()
                self.shape = shape
        def forward(self, input):
                return input.view(*self.shape)

class Net(nn.Module):
        def __init__(self, dims,config):
                super(Net, self).__init__()
                self.modules = []
                self.residuals = []
                nchannels = dims[0]
                ksize = 3

                #select activation function
                self.func = F.relu
                if config.non_lin != 'relu':
                    self.func = F.leaky_relu
                if config.batch_norm:
                    self.bn1=nn.BatchNorm3d(32)
                    self.bn2=nn.BatchNorm3d(32)
                    self.bn3=nn.BatchNorm3d(3)
                else:
                    self.bn1 = None
                

                self.conv1 = nn.Conv3d(nchannels, 32, kernel_size=ksize)
                self.conv2 = nn.Conv3d(32,32,kernel_size=ksize)
                self.max1 = nn.MaxPool3d(3)
                
                self.conv3 = nn.Conv3d(32,3,kernel_size=ksize)
                self.lin1 = nn.Linear(5184,1,bias=False)
                self.dp1 = nn.Dropout(p=config.dropout)


        def forward_one(self, x): #should approximate the affinity of the receptor/ligand pair
                if self.bn1 is not None:
                    x= self.bn1(self.conv1(x))
                    x= self.bn2(self.conv2(self.func(x)))
                    x=self.max1(self.func(x))

                    x= self.func(self.bn3(self.conv3(x)))
                else:
                    x= self.conv1(x)
                    x= self.conv2(self.func(x))
                    x=self.max1(self.func(x))

                    x= self.func(self.conv3(x))

                x= x.view(x.shape[0], -1)       
                return x

        def forward(self,x1,x2):
                lig1 = self.forward_one(x1)
                lig2 = self.forward_one(x2)
                diff = self.dp1(lig1 - lig2)
                return self.lin1(diff)
