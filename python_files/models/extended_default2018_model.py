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
                nchannels = dims[0]

                self.func = F.relu
                self.dropout = nn.Dropout(p=config.dropout)

                avgpool1 = nn.AvgPool3d(2,stride=2)
                self.add_module('avgpool_0', avgpool1)
                self.modules.append(avgpool1)
                conv1 = nn.Conv3d(nchannels,out_channels=32,padding=1,kernel_size=3,stride=1) 
                self.add_module('unit1_conv',conv1)
                self.modules.append(conv1)
                conv2 = nn.Conv3d(32,out_channels=32,padding=0,kernel_size=1,stride=1) 
                self.add_module('unit2_conv',conv2)
                self.modules.append(conv2)
                avgpool2 = nn.AvgPool3d(2,stride=2)
                self.add_module('avgpool_1', avgpool2)
                self.modules.append(avgpool2)
                conv3 = nn.Conv3d(32,out_channels=64,padding=1,kernel_size=3,stride=1) 
                self.add_module('unit3_conv',conv3)
                self.modules.append(conv3)
                conv4 = nn.Conv3d(64,out_channels=64,padding=0,kernel_size=1,stride=1) 
                self.add_module('unit4_conv',conv4)
                self.modules.append(conv4)
                avgpool3 = nn.AvgPool3d(2,stride=2)
                self.add_module('avgpool_2', avgpool3)
                self.modules.append(avgpool3)
                conv5 = nn.Conv3d(64,out_channels=128,padding=1,kernel_size=3,stride=1) 
                self.add_module('unit5_conv',conv5)
                self.modules.append(conv5)
                div = 2*2*2
                last_size = int(dims[1]//div * dims[2]//div * dims[3]//div * 128)
                print(last_size)
                flattener = View((-1,last_size))
                self.add_module('flatten',flattener)
                self.modules.append(flattener)
                latent = nn.Linear(last_size,config.hidden_size)
                self.add_module('make_latent',latent)
                self.modules.append(latent)
                self.fc = nn.Linear(config.hidden_size,1)
                self.add_module('last_fc',self.fc)

        def forward_one(self, x): #should approximate the affinity of the receptor/ligand pair
                for layer in self.modules:
                        x= layer(x)
                        if isinstance(layer,nn.Conv3d):
                                x=self.func(x)
                return x

        def forward(self,x1,x2):
                lig1 = self.forward_one(x1)
                lig2 = self.forward_one(x2)
                diff = self.dropout(lig1 - lig2)
                return self.fc(diff)
