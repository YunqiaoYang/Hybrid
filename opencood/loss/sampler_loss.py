import torch
import torch.nn as nn

class SampleLoss(nn.Module):

    def __init__(self):
        super(SampleLoss, self).__init__()
    
    def forward(self, sampled_lidar_list, raw_lidar_list):
        
        assert len(sampled_lidar_list)==len(raw_lidar_list)
        sampled_loss=0
        for sampled_lidar,raw_lidar in zip(sampled_lidar_list,raw_lidar_list):
            
            raw_lidar=raw_lidar.to(sampled_lidar.device)
            dist_mat=torch.cdist(sampled_lidar[:,0:3].contiguous(),raw_lidar[:,0:3].contiguous())
            min_index_raw=torch.argmin(dist_mat,dim=0)
            min_index_sample=torch.argmin(dist_mat,dim=1)
            
            LB = torch.mean(dist_mat[min_index_raw,range(len(min_index_raw))])

            LF = torch.mean(dist_mat[range(len(min_index_sample)),min_index_sample])
            
            LM = torch.max(dist_mat[range(len(min_index_sample)),min_index_sample])


            box_sampled_loss =5*LB + 1*LF + LM
            sampled_loss+=box_sampled_loss
        sampled_loss=sampled_loss/len(sampled_lidar_list)
        return sampled_loss

class IOUloss(nn.Module):


    def __init__(self):
        super(IOUloss, self).__init__()
    
    def forward(self,pred_corners,gt_corners):
        assert len(pred_corners)==len(gt_corners)
        loss=0
        for pred_corner,gt_corner in zip(pred_corners,gt_corners):
            pass
            #calc IOU
            
        return loss

class SampleBoxLoss(nn.Module):

    def __init__(self):
        super(SampleBoxLoss, self).__init__()

    def forward(self,sampled_lidar_list,surface_points):
        assert len(sampled_lidar_list)==surface_points.shape[0]
        
        corner_pos=surface_points
        sampled_loss=0
        for i in range(len(sampled_lidar_list)):
            sampled_lidar=sampled_lidar_list[i].to(corner_pos.device)
            dist_mat=torch.cdist(sampled_lidar[:,0:3].contiguous(),corner_pos[i,:,:].contiguous())
            min_index_raw=torch.argmin(dist_mat,dim=0)
            min_index_sample=torch.argmin(dist_mat,dim=1)
            
            LB = torch.mean(dist_mat[min_index_raw,range(len(min_index_raw))])

            LF = torch.mean(dist_mat[range(len(min_index_sample)),min_index_sample])
            
            LM = torch.max(dist_mat[range(len(min_index_sample)),min_index_sample])


            box_sampled_loss =5*LB + 1*LF + LM
            sampled_loss+=box_sampled_loss
        sampled_loss=sampled_loss/len(sampled_lidar_list)
        return sampled_loss
        

