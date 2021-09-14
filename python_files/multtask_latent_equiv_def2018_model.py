from se3cnn.image.gated_block import GatedBlock
from se3cnn.image.filter import low_pass_filter
import torch.nn as nn
import torch.nn.functional as F

class LowPass(nn.Module):
    def __init__(self,scale,stride):
        super(LowPass,self).__init__()
        self.scale = scale
        self.stride = stride
        
    def forward(self,inp):
        return low_pass_filter(inp,self.scale,self.stride)

class View(nn.Module):
        def __init__(self, shape):
                super(View, self).__init__()
                self.shape = shape
        def forward(self, input):
                return input.view(*self.shape)

class Net(nn.Module):
    def __init__(self, dims, config):
        super(Net, self).__init__()
        self.modules = []
        nchannels = dims[0]

        self.dropout = nn.Dropout(p=config.dropout)

        lowpass1 = LowPass(2,stride=2)
        self.add_module('lowpass_filter_0', lowpass1)
        self.modules.append(lowpass1)
        conv1 = GatedBlock((nchannels,),(32,32),size=3,padding=1,stride=1,activation=(None,F.sigmoid))
        self.add_module('unit1_conv_equiv',conv1)
        self.modules.append(conv1)
        conv2 = GatedBlock((32,32),(32,32),padding=0,size=1,stride=1,activation=(F.relu,F.sigmoid))
        self.modules.append(self.conv3)
        self.modules.append(self.conv4)
        self.lowpass3 = LowPass(2,2)
        self.modules.append(self.lowpass3)
        conv2 = GatedBlock((32,32),(32,32),padding=0,size=1,stride=1,activation=(F.relu,F.sigmoid))
        self.add_module('unit2_conv_equiv',conv2)
        self.modules.append(conv2)
        lowpass2 = LowPass(2,stride=2)
        self.add_module('lowpass_filter_1', lowpass2)
        self.modules.append(lowpass2)
        conv3 = GatedBlock((32,32),(64,64),padding=1,size=3,stride=1,activation=(F.relu,F.sigmoid))
        self.add_module('unit3_conv_equiv',conv3)
        self.modules.append(conv3)
        conv4 = GatedBlock((64,64),(64,64),padding=0,size=1,stride=1,activation=(F.relu,F.sigmoid))
        self.add_module('unit4_conv_equiv',conv4)
        self.modules.append(conv4)
        lowpass3 = LowPass(2,stride=2)
        self.add_module('lowpass_filter_2', lowpass3)
        self.modules.append(lowpass3)
        conv5 = GatedBlock((64,64),(128,128),padding=1,size=3,stride=1,activation=(F.relu,F.sigmoid))
        self.add_module('unit5_conv',conv5)
        self.modules.append(conv5)
        div = 2*2*2
        last_size = int(dims[1]//div * dims[2]//div * dims[3]//div * 512)
        print(last_size)
        flattener = View((-1,last_size))
        self.add_module('flatten',flattener)
        self.modules.append(flattener)
        if config.hidden_size > 0:
            fc = nn.Linear(last_size,config.hidden_size)
            self.add_module('reduce_dim',fc)
            self.modules.append(fc)
            self.ddg = nn.Linear(config.hidden_size,1)
            self.add_module('DDG',self.ddg)
            self.dg = nn.Linear(config.hidden_size,1)
            self.add_module('affinity',self.dg)
        else:
            self.ddg = nn.Linear(last_size,1)
            self.add_module('DDG',self.ddg)
            self.dg = nn.Linear(last_size,1)
            self.add_module('affinity',self.dg)

    def forward_one(self, x): #should approximate the affinity of the receptor/ligand pair
        for layer in self.modules:
            x= layer(x)
        return x

    def forward(self,x1,x2):
        lig1 = self.forward_one(x1)
        lig2 = self.forward_one(x2)
        lig1_aff = self.dg(lig1)
        lig2_aff = self.dg(lig2)
        diff = self.dropout(lig1 - lig2)
        return self.ddg(diff), lig1_aff, lig2_aff,lig1,lig2
