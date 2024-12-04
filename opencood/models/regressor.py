import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_, kaiming_normal_

class PointNetfeat(nn.Module):
    def __init__(self, pts_dim, x=1):
        super(PointNetfeat, self).__init__()
        self.output_channel = 512 * x
        self.conv1 = torch.nn.Conv1d(pts_dim, 64 * x, 1)
        self.conv2 = torch.nn.Conv1d(64 * x, 128 * x, 1)
        self.conv3 = torch.nn.Conv1d(128 * x, self.output_channel, 1)
        # self.bn1 = nn.BatchNorm1d(64 * x)
        # self.bn2 = nn.BatchNorm1d(128 * x)
        # self.bn3 = nn.BatchNorm1d(self.output_channel)


    def forward(self, x):
        
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = self.bn3(self.conv3(x))
        # x = torch.max(x, 0, keepdim=True)[0]
        # x = x.view(-1, self.output_channel)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 0, keepdim=True)[0]
        x = x.view(-1, self.output_channel)
        
        return x


class PointNet(nn.Module):
    def __init__(self, pts_dim, x):
        super(PointNet, self).__init__()
        self.feat = PointNetfeat(pts_dim, x)
        self.fc1 = nn.Linear(512 * x, 256 * x)
        self.fc2 = nn.Linear(256 * x, 2048)

        # self.pre_bn = nn.BatchNorm1d(pts_dim)
        # self.bn1 = nn.BatchNorm1d(256 * x)
        # self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        # NOTE: should there put a BN?
        self.fc_s1 = nn.Linear(2048,1024)
        self.fc_s2 = nn.Linear(1024, 1024)
        self.fc_s3 = nn.Linear(1024, 256)
        self.fc_s4 = nn.Linear(256, 3)
        

        self.fc_ce1= nn.Linear(2048,1024)
        self.fc_ce2 = nn.Linear(1024, 1024)
        self.fc_ce3 = nn.Linear(1024, 256)
        self.fc_ce4 = nn.Linear(256, 3)

        self.fc_hr1 = nn.Linear(2048,1024)
        self.fc_hr2 = nn.Linear(1024, 1024)
        self.fc_hr3 = nn.Linear(1024, 256)
        self.fc_hr4 = nn.Linear(256, 2)
        

        # self.fc_hr1 = nn.Linear(512,512)
        # self.fc_hr2 = nn.Linear(512,256)
        # self.fc_hr3 = nn.Linear(256,2)

    def forward(self, x):
        
        
        x=x.unsqueeze(0)
        x=x.permute(1,2,0)
        centerpoint=torch.mean(x,dim=0,keepdim=True)

        x=x-centerpoint
    

        x = self.feat(x)
        

        x = F.relu(self.fc1(x))
        feat = F.relu(self.fc2(x))
        
        
        x1 = F.relu(self.fc_ce1(feat))
        x1 = F.relu(self.fc_ce2(x1))
        x1 = F.relu(self.fc_ce3(x1))
        centers = self.fc_ce4(x1)
        centers = centers+centerpoint[:,:3,0]
        
        x2 = F.relu(self.fc_s1(feat))
        x2 = F.relu(self.fc_s2(x2))
        x2 = F.relu(self.fc_s3(x2))
        sizes = self.fc_s4(x2)


        x3 = F.relu(self.fc_hr1(feat))
        x3 = F.relu(self.fc_hr2(x3))
        x3 = F.relu(self.fc_hr3(x3))
        headings = self.fc_hr4(x3)

        result=torch.cat((centers,sizes,headings),dim=1)
        

        return result

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

class Sampler(nn.Module):
    
    def __init__(self):

        super(Sampler, self).__init__()
        # input(500,4),output(100,4)
        self.MLP1=nn.Sequential(
            
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=256,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=[1,4])  
            
        )
         
        self.MLP2=nn.Sequential(
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,50)
        )

        self.relu= nn.ReLU()

    def forward(self, input):
        
        
        origin=input.clone().squeeze(1).permute(0,2,1)
        center=torch.mean(input,dim=2,keepdim=True)
        input=input-center
        x = self.relu(self.MLP1(input))
        global_feature = torch.max(x,2,keepdim=True)[0]
        # cat global feature behind each point feature
        global_feature = global_feature.repeat(1,1,origin.shape[2],1)
        x = torch.cat((x,global_feature),1)
        # change dimension
        x = x.permute(0,3,2,1)
        x = self.MLP2(x).squeeze(1)
        x = nn.functional.gumbel_softmax(x,hard=True,dim=1)
        
        output = torch.bmm(origin,x).permute(0,2,1).unsqueeze(1)
        

        return output