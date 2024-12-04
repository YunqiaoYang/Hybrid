# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
from torch import nn

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.regressor import PointNet,PointNetfeat,Sampler
from opencood.loss.sampler_loss import *
from opencood.utils import eval_utils

import torch.nn.functional as F
import numpy as np
import glob
from icecream import ic
from tqdm import tqdm
import torch_cluster.fps as fps
import time
from opencood.visualization import vis_utils, my_vis, simple_vis
#import open3d as o3d
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum


class Regressor(nn.Module):

    def __init__(self, pts_dim=4, x=4):
        super(Regressor, self).__init__()
        self.feat = PointNetfeat(pts_dim, x)
        self.fc1 = nn.Linear(512 * x, 256 * x)
        self.fc2 = nn.Linear(256 * x, 256)

        self.pre_bn = nn.BatchNorm1d(pts_dim)
        self.relu = nn.ReLU()
        # NOTE: should there put a BN?
        self.fc_s1 = nn.Linear(256, 256)
        self.fc_s2 = nn.Linear(256, 8, bias=False)


    def forward(self, x):
        
        x=x.unsqueeze(dim=0)
        x=x.permute(0,2,1)
        x = self.feat(self.pre_bn(x))
        print(x.shape)
        x = F.relu(self.fc1(x))
        feat = F.relu(self.fc2(x))
        x = F.relu(self.fc_s1(feat))
        corner_pos = self.fc_s2(x).reshape(4,2)
        



        return corner_pos

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',required=False,
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="early",
                        help='passed to inference.')
    parser.add_argument('--regressor_dir', '-r', default="")
    parser.add_argument('--sampler_dir', '-S', default="")
    parser.add_argument('--note', '-n', default='')
    parser.add_argument('--stage', '-s', default="train_regressor",required=True)
    parser.add_argument('--unc_only', '-U', default=False,action='store_true')
    parser.add_argument('--use_angle', '-a', default=True,action='store_true')
    opt = parser.parse_args()
    return opt


def main():
    
    opt = train_parser()
    
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    hypes.update({'sample_method': None,
                  'sampling_rate': 1.0 ,
                  'model_dir': opt.model_dir  })
    
    print('Dataset Building')

    hypes['fusion']['core_method']="sampler"

    opencood_train_dataset = build_dataset(hypes, visualize=True, train=True)
    opencood_validate_dataset = build_dataset(hypes,visualize=True,train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=1,
                              num_workers=8,
                              collate_fn=opencood_train_dataset.collate_batch_test,
                              shuffle=True,
                              pin_memory=False,
                              drop_last=False,
                              prefetch_factor=4)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=1,
                            num_workers=8,
                            collate_fn=opencood_validate_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False,
                            prefetch_factor=4)

    print('Creating Model')
    saved_path = opt.model_dir+'/'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'/'
    
    if opt.regressor_dir!="":
        saved_path=opt.regressor_dir
    if opt.stage=="train_sampler":
        saved_path+="sampler/"
    
    if opt.sampler_dir!="":
        saved_path=opt.sampler_dir
    
    if opt.note != '':
        saved_path += opt.note + '/'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
        train_utils.backup_script(saved_path)
    
    if opt.stage == "train_regressor" :
        print("training regressor")
        if opt.use_angle:
            detect_model = PointNet(pts_dim=3,x=16)
            detect_model.init_weights()
        else:
            detect_model = Regressor()
        if opt.regressor_dir != '':
            file_list = glob.glob(os.path.join(opt.regressor_dir, 'best_regressor_at_*.pth'))
            if file_list:
                assert len(file_list) == 1
                print("using best regression model at epoch %s" % \
                        file_list[0].split("/")[-1].rstrip(".pth").lstrip("best_regressor_at_"))
                detect_model.load_state_dict(torch.load(file_list[0] , map_location='cpu'), strict=False)
        # # define the loss
        criterion = nn.SmoothL1Loss(reduction='sum')
        # optimizer setup
        optimizer = torch.optim.Adam(detect_model.parameters(),
                                    lr=1e-5,
                                    weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=2,
                                                gamma=0.95)
        # backup the scripts

    

    
    if opt.stage == "train_sampler":
        print("training sampler")
        calc_sim_loss= SampleLoss()
        
        if opt.use_angle:
            detect_model = PointNet(pts_dim=3,x=16)
        else:
            detect_model = Regressor()
        file_list = glob.glob(os.path.join(opt.regressor_dir, 'best_regressor_at_*.pth'))
        if file_list:
            assert len(file_list) == 1
            print("using best regression model at epoch %s" % \
                    file_list[0].split("/")[-1].rstrip(".pth").lstrip("best_regressor_at_"))
            detect_model.load_state_dict(torch.load(file_list[0] , map_location='cpu'), strict=False)
        detect_model.eval()
        
        sampler=Sampler()
        if opt.sampler_dir!="":
            file_list = glob.glob(os.path.join(opt.sampler_dir, 'best_sampler.pth'))
            if file_list:
                assert len(file_list) == 1
                print("using best sampler model at epoch %s" % \
                        file_list[0].split("/")[-1].rstrip(".pth").lstrip("best_sampler_at_*.pth"))
                sampler.load_state_dict(torch.load(file_list[0] , map_location='cpu'), strict=False)

        criterion = nn.SmoothL1Loss(reduction='sum')
        # optimizer setup
        optimizer = torch.optim.Adam(sampler.parameters(),
                                    lr=1e-6,
                                    weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=2,
                                                gamma=0.9)
    if opt.stage == "test_sampler" :
        print("testing sampler")
        sampler.load_state_dict(torch.load(os.path.join(opt.sampler_dir,'best_sampler.pth'), map_location='cpu'), strict=False)
        sampler.eval()
    if opt.stage == "test_regressor" :
        print("testing regressor")
        if opt.use_angle:
            detect_model = PointNet(pts_dim=4,x=4)
        else:
            detect_model = Regressor()
        file_list = glob.glob(os.path.join(opt.regressor_dir, 'best_regressor_at_*.pth'))
        if file_list:
            assert len(file_list) == 1
            print("using best regression model at epoch %s" % \
                    file_list[0].split("/")[-1].rstrip(".pth").lstrip("best_regressor_at_"))
            detect_model.load_state_dict(torch.load(file_list[0] , map_location='cpu'), strict=False)
        detect_model.eval()   
    if opt.stage == "finetune":
        print("finetune")
        detect_model=train_utils.create_model(hypes)
        _, detect_model = train_utils.load_saved_model(saved_path, detect_model)
    
    #return 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #model = Sampler()
    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    init_epoch = 0
       
    
    # scheduler setup
    

    # we assume gpu is necessary
    if torch.cuda.is_available():
        detect_model.to(device)
        if opt.stage=="train_sampler":
            sampler.to(device)
        if opt.stage=="test_sampler":
            sampler.to(device)
        
    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    print('Training start')
    best_eval_loss = 1e5
    lowest_epoch=-1

    

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if opt.stage =="test_regressor" or opt.stage =="test_sampler":
            break
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        running_loss = 0.0
        running_angle_loss = 0.0
        running_center_loss = 0.0
        running_size_loss=0.0
        running_box = 0
        running_sampler_loss=0.0
        for i, batch_data in tqdm(enumerate(train_loader)):
            
            
            if batch_data is None:
                continue
            
            if opt.stage == "train_regressor":
                detect_model.train()
                # the model will be evaluation mode during validation
                lidar_list=batch_data['ego']['processed_lidar']
                
                if len(lidar_list)==0:
                    continue
                lidar_list=train_utils.to_device(lidar_list,device)
                corner_pos=batch_data['ego']['bbox']
                corner_pos=corner_pos.to(device)
                #raw_lidar=torch.vstack(lidar_list).to(device)#.detach().cpu().numpy()
                detect_model.zero_grad()
                optimizer.zero_grad()
                regression,supervised_mask=batch_detect(lidar_list,detect_model)
                

            elif opt.stage == "train_sampler":
                sampler.train()
                # the model will be evaluation mode during validation
                lidar_list=batch_data['ego']['processed_lidar']
                if len(lidar_list)==0:
                    continue
                lidar_list=train_utils.to_device(lidar_list,device)
                corner_pos=batch_data['ego']['bbox']
                surface_points=batch_data['ego']['surface_points']
                
                corner_pos=corner_pos.to(device)
                
                raw_lidar=torch.vstack(lidar_list).to(device)
                
                sampler.zero_grad()
                optimizer.zero_grad()
                
                sampled_lidar_list=sample_process(lidar_list,sampler,device)



                #sample_loss = calc_sim_loss(sampled_lidar_list,surface_points)
                sample_loss = 0
                if len(sampled_lidar_list)==0:
                    continue

                regression,supervised_mask=batch_detect(sampled_lidar_list,detect_model)
                
                if regression!=None:
                    # running_sampler_loss+=sample_loss.item()
                    pass
            
            if regression!=None:
                    
                    corner_pos=corner_pos[supervised_mask]
                    
                    if opt.use_angle:

                        label=bbox2labels(corner_pos)
                        center_label=label[:,:,:3]
                        size_label=label[:,:,3:6]
                        angle_label=label[:,:,6:]
                        center_regression=regression[:,:,:3]
                        size_regression=regression[:,:,3:6]
                        angle_regression=regression[:,:,6:]

                        # print("angle_label",angle_label[0])
                        # print("angle_regression",angle_regression[0])

                        # print("center_label",center_label[0])
                        # print("center_regression",center_regression[0])
                        
                        center_loss=criterion(center_regression,center_label)
                        size_loss=criterion(size_regression,size_label)
                        angle_loss=criterion(angle_regression,angle_label)
                        loss=center_loss+size_loss+15*angle_loss
                        
                        running_loss+=loss.item()
                        running_box+=regression.shape[0]
                        running_angle_loss+=angle_loss.item()
                        running_center_loss+=center_loss.item()
                        running_size_loss+=size_loss.item()
                        

                        if i%500==499:
                            print("epoch:",epoch,"|batch:",i,"|loss:",running_loss/running_box,"|angle_loss:",running_angle_loss/running_box,"|center_loss:",running_center_loss/running_box,"|size_loss:",running_size_loss/running_box,"|sampling loss",running_sampler_loss/running_box)
                            running_loss=0.0
                            running_box=0.0
                            running_angle_loss=0.0
                            running_center_loss=0.0
                            running_size_loss=0.0
                            running_sampler_loss=0.0
                    
                    else:#use corner_pos directly
                        dist_mat = torch.cdist(corner_pos, regression)
                        min_index = torch.argmin(dist_mat, dim=2)
                        regression= torch.gather(regression, 1, min_index.unsqueeze(-1).expand(-1, -1, 2))
                        loss=criterion(regression,corner_pos)
                        if i%200==199:
                            print("e:",epoch,"|b:",i,"|loss:",running_loss/running_box,"|sampling loss",running_sampler_loss/running_box)
                            running_loss=0.0
                            running_box=0.0
                            running_sampler_loss=0.0
                        running_loss+=loss.item()
                        running_box+=regression.shape[0]
                    if opt.stage=="train_sampler":
                        loss+=0*sample_loss
                    loss=loss/regression.shape[0]
                    loss.backward()
                    optimizer.step()
                    # for name,param in detect_model.named_parameters():
                    #     if param.grad is None:
                    #         print(name)


            #####
            # if i%800==0:
            #     regression=regression.detach().cpu()
            #     regression=labels2bbox(regression)
            #     print(regression.shape)
            #     print(corner_pos.shape)
            #     print(batch_data['ego']['origin_lidar'].shape)
            #     infer_result={
            #             "pred_box_tensor":regression.detach(),
            #             "gt_box_tensor":corner_pos.detach()
            #         }
            #     vis_save_path_root = os.path.join(saved_path, f'vis_{epoch}')
            #     if not os.path.exists(vis_save_path_root):
            #             os.makedirs(vis_save_path_root)
            #     vis_save_path = os.path.join(vis_save_path_root, 'train_bev_%05d.png' % i)
            #     simple_vis.visualize(infer_result,
            #                             batch_data['ego']['origin_lidar'][0],
            #                             hypes['postprocess']['gt_range'],
            #                             vis_save_path,
            #                             method='bev',
            #                             left_hand=True)
            ######        


            
        if opt.stage=="train_sampler" or opt.stage=="test_regressor":
            detect_model.eval()
            sampler.eval()
        elif opt.stage=="train_regressor":
            detect_model.eval()

        eval_loss=0.0
        eval_box=0.0
        for i, batch_data in tqdm(enumerate(val_loader)):
            if batch_data is None:
                continue
            lidar_list=batch_data['ego']['processed_lidar']
            if len(lidar_list)==0:
                continue
            lidar_list=train_utils.to_device(lidar_list,device)
            corner_pos=batch_data['ego']['bbox'].to(device)
            surface_points=batch_data['ego']['surface_points']
            #corner_pos=torch.from_numpy(corner_pos).to(device)
            
            raw_lidar=batch_data['ego']['origin_lidar'][0].to(device)
            origin_lidar_list=lidar_list
            if opt.stage=="train_sampler":
                lidar_list=sample_process(lidar_list,sampler,device)
                if len(sampled_lidar_list)==0:
                    continue
            
            regression,mask=batch_detect(lidar_list,detect_model)
            if regression!=None:
                if corner_pos.shape[1]!=8:
                    corner_pos=corner_pos.unsqueeze(0)
                corner_pos=corner_pos[mask]
                if opt.use_angle:
                    
                    label=bbox2labels(corner_pos)
                    center_label=label[:,:,:3]
                    size_label=label[:,:,3:6]
                    angle_label=label[:,:,6:]
                    center_regression=regression[:,:,:3]
                    size_regression=regression[:,:,3:6]
                    angle_regression=regression[:,:,6:]

                    # print("angle_label",angle_label[0])
                    # print("angle_regression",angle_regression[0])

                    # print("center_label",center_label[0])
                    # print("center_regression",center_regression[0])
                    
                    center_loss=criterion(center_regression,center_label)
                    size_loss=criterion(size_regression,size_label)
                    angle_loss=criterion(angle_regression,angle_label)
                    loss=center_loss+size_loss+10*angle_loss

                    regression=labels2bbox(regression)

                else:
                    dist_mat = torch.cdist(corner_pos, regression)
                    min_index = torch.argmin(dist_mat, dim=2)
                    regression= torch.gather(regression, 1, min_index.unsqueeze(-1).expand(-1, -1, 2))
                    loss=criterion(regression,corner_pos)
                
                eval_loss+=loss.item()
                eval_box+=len(mask)
            
            if (i % 40 == 0) and (regression is not None):
                vis_save_path_root = os.path.join(saved_path, f'vis_{epoch}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)
                infer_result={
                    "pred_box_tensor":regression.detach(),
                    "gt_box_tensor":corner_pos.detach()
                }
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                sample_save_path = os.path.join(vis_save_path_root, 'sample_%d.png' % i)
                ax = plt.subplot(projection = '3d')  # 创建一个三维的绘图工程
                ax.set_title('3d_image_show')  # 设置本图名称
                sample_sample=lidar_list[0].detach().cpu().numpy()
                x = sample_sample[:,0]  # 读取X轴的坐标
                y = sample_sample[:,1]  # 读取Y轴的坐标
                z = sample_sample[:,2]  # 读取Z轴的坐标
        
                ax.scatter(x, y, z, c = 'r')   # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
                
                origin_sample=origin_lidar_list[0].detach().cpu().numpy()
                x = origin_sample[:,0]  # 读取X轴的坐标
                y = origin_sample[:,1]  # 读取Y轴的坐标
                z = origin_sample[:,2]  # 读取Z轴的坐标
                ax.scatter(x, y, z, c = 'b')   # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
                ax.set_xlabel('X')  # 设置x坐标轴
                ax.set_ylabel('Y')  # 设置y坐标轴
                ax.set_zlabel('Z')  # 设置z坐标轴
                plt.savefig(sample_save_path)
                plt.clf()


                

                simple_vis.visualize(infer_result,
                                    raw_lidar,
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=True)


                
        print("epoch:",epoch,"|eval_loss:",eval_loss/eval_box)

        if opt.stage=="train_regressor":
            torch.save(detect_model.state_dict(),os.path.join(saved_path,'regressor_%05d.pth'%epoch))

        if eval_loss/eval_box<best_eval_loss:
            best_eval_loss=eval_loss/eval_box
            if opt.stage=="train_sampler":
                #torch.save(sampler.state_dict(),os.path.join(saved_path,'best_sampler.pth'))
                torch.save(sampler.state_dict(),os.path.join(saved_path,'best_sampler_at_%05d.pth'%(epoch+1)))
                if lowest_epoch!=-1 and os.path.exists(os.path.join(saved_path,'best_sampler_at_%05d.pth' % (lowest_epoch))):
                    os.remove(os.path.join(saved_path,'best_sampler_at_%05d.pth'%(lowest_epoch)))
                lowest_epoch=epoch+1
            elif opt.stage=="train_regressor":
                torch.save(detect_model.state_dict(),os.path.join(saved_path,'best_regressor_at_%05d.pth'%(epoch+1)))
                if lowest_epoch!=-1 and os.path.exists(os.path.join(saved_path,'best_regressor_at_%05d.pth' % (lowest_epoch))):
                    os.remove(os.path.join(saved_path,'best_regressor_at_%05d.pth'%(lowest_epoch)))
                lowest_epoch=epoch+1
        
        scheduler.step(epoch)
        opencood_train_dataset.reinitialize()
    if opt.stage=="train_sampler":         
        print('Training Finished, checkpoints saved to ', saved_path)
    elif opt.stage=="train_regressor":
        print('Training Finished, checkpoints saved to ', saved_path)
    else:
        pass

    print("-----start inferencing-----")
    
    opencood_validate_dataset.reinitialize()
    eval_loss=0
    eval_box=0
    detect_model.eval()
    final_result=[]
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    sampler.cuda()
    for i, batch_data in tqdm(enumerate(val_loader)):
        if batch_data is None:
            continue
        with torch.no_grad():
            lidar_list=batch_data['ego']['processed_lidar']
            if len(lidar_list)==0:
                continue
            lidar_list=train_utils.to_device(lidar_list,device)
            corner_pos=batch_data['ego']['bbox']
            corner_pos=torch.from_numpy(corner_pos).to(device)
            raw_lidar=torch.vstack(lidar_list).to(device)
            raw_lidar=batch_data['ego']['origin_lidar']
            origin_lidar_list=lidar_list
            if opt.stage=="train_sampler" or "test_sampler":
                lidar_list=sample_process(lidar_list,sampler,device)
                if len(sampled_lidar_list)==0:
                    continue
            
            regression,mask=batch_detect(lidar_list,detect_model)
            if regression!=None:
                if corner_pos.shape[1]!=8:
                    corner_pos=corner_pos.unsqueeze(0)
                corner_pos=corner_pos[mask]
                if opt.use_angle:

                    regression=labels2bbox(regression)

                else:
                    dist_mat = torch.cdist(corner_pos, regression)
                    min_index = torch.argmin(dist_mat, dim=2)
                    regression= torch.gather(regression, 1, min_index.unsqueeze(-1).expand(-1, -1, 2))
                # print(regression.shape)
                # print(corner_pos.shape)
                pred_score=torch.ones(regression.shape[0])
                eval_utils.caluclate_tp_fp(regression,
                                    pred_score,
                                    corner_pos,
                                    result_stat,
                                    0.3)
                eval_utils.caluclate_tp_fp(regression,
                                    pred_score,
                                    corner_pos,
                                    result_stat,
                                    0.5)
                eval_utils.caluclate_tp_fp(regression,
                                    pred_score,
                                    corner_pos,
                                    result_stat,
                                    0.7)
                #iou_tensor=np.array(iou_list)
                
            if (i % 40 == 0) and (regression is not None):
                vis_save_path_root = os.path.join(saved_path, f'final_vis')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)
                infer_result={
                    "pred_box_tensor":regression.detach(),
                    "gt_box_tensor":corner_pos.detach()
                }
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                

                simple_vis.visualize(infer_result,
                                    raw_lidar,
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=False)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                            saved_path,0) 
    #print("ap30:",ap30,"ap50:",ap50,"ap70:",ap70)      

                
                


def lidar2data(projected_lidar_stack,dataset):
    projected_lidar_stack = mask_points_by_range(projected_lidar_stack,\
        dataset.params['preprocess']['cav_lidar_range'])
    lidar_dict=dataset.pre_processor.preprocess(projected_lidar_stack)
    lidar_torch_dict=dataset.pre_processor.collate_batch([lidar_dict])
    temp_dict={}
    temp_dict.update({'processed_lidar':lidar_torch_dict})
    return temp_dict

def batch_detect(lidar_list,detect_model):

    results=[]
    supervised_mask=[]
    for p in detect_model.parameters():
        device=p.device
        break
    
    for i,lidar in enumerate(lidar_list):
        lidar=lidar.to(device)
        if lidar.shape[0] < 500 and False:
            continue

        supervised_mask.append(i)
        corner_pos=detect_model(lidar[:,:3])
        result=corner_pos.unsqueeze(0)
        results.append(result)
    if len(results)>0:
        results=torch.cat(results,dim=0)
    else:
        results=None
    
    return results,supervised_mask

def sample_process(input_list,model,device):
    sampled_data=[]
    boxes=[]
    for box in input_list:
        
        if box.shape[0]<1000:
            box=replicate_to_num(box,1000).unsqueeze(0)
        else:
            sampling_idx = np.sort(np.random.choice(range(box.shape[0]),1000))
            box = box[sampling_idx].unsqueeze(0)
        
        boxes.append(box)

    
            

    if len(boxes)!=0:
        box_stack=torch.cat(boxes,dim=0).unsqueeze(1).to(device)
        box_stack=model(box_stack)
        for i in range(len(boxes)):
            sampled_data.append(box_stack[i].squeeze(0))

   
      
    
    return sampled_data

def up_sampling(box, num):
    num_point=box.shape[0]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(box[:,:3].numpy())
    radius = 1
    max_nn = 10
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))  # 法线估计
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=2)
        cloud=mesh.sample_points_uniformly(number_of_points=num)
    output=np.asarray(cloud.points)
    output=torch.from_numpy(output)
    return output

def replicate_to_num(points,num):
    num_point=points.shape[0]
    replicate_n=num-num_point
    sampling_idx = np.sort(np.random.choice(range(num_point), replicate_n))
    replicate =points[sampling_idx]
    points=torch.cat([points,replicate])
    return points

def bbox2labels(corner_pos):
    sizex=0.5*(torch.sqrt((corner_pos[:,1,0]-corner_pos[:,0,0])**2+(corner_pos[:,1,1]-corner_pos[:,0,1])**2)+torch.sqrt((corner_pos[:,2,0]-corner_pos[:,3,0])**2+(corner_pos[:,2,1]-corner_pos[:,3,1])**2)).unsqueeze(1)
    sizey=0.5*(torch.sqrt((corner_pos[:,2,0]-corner_pos[:,1,0])**2+(corner_pos[:,2,1]-corner_pos[:,1,1])**2)+torch.sqrt((corner_pos[:,3,0]-corner_pos[:,0,0])**2+(corner_pos[:,3,1]-corner_pos[:,0,1])**2)).unsqueeze(1)
    sizez=(corner_pos[:,4,2]-corner_pos[:,0,2]).unsqueeze(1)
    centerx=0.25*(corner_pos[:,0,0]+corner_pos[:,1,0]+corner_pos[:,2,0]+corner_pos[:,3,0]).unsqueeze(1)
    centery=0.25*(corner_pos[:,0,1]+corner_pos[:,1,1]+corner_pos[:,2,1]+corner_pos[:,3,1]).unsqueeze(1)
    centerz=0.25*(corner_pos[:,0,2]+corner_pos[:,4,2]+corner_pos[:,2,2]+corner_pos[:,6,2]).unsqueeze(1)
    cosdir=(corner_pos[:,1,0]-corner_pos[:,0,0]).unsqueeze(1)/sizex
    sindir=(corner_pos[:,1,1]-corner_pos[:,0,1]).unsqueeze(1)/sizex
    #angle=torch.atan2(corner_pos[:,3,1]-corner_pos[:,0,1],corner_pos[:,3,0]-corner_pos[:,0,0]).unsqueeze(1)
    label=torch.cat([centerx,centery,centerz,sizex,sizey,sizez,cosdir,sindir],dim=1).unsqueeze(1)
    return label

def sortbyx(rectangles):
    indices = torch.argsort(rectangles[:, :, 0], dim=1)
    rectangles = torch.gather(rectangles, 1, indices.unsqueeze(-1).repeat(1, 1, 2))
    return rectangles

def labels2bbox(labels):

    centers=labels[:,:,:3]
    sizes=labels[:,:,3:6]
    angle=labels[:,:,6:]

    bbox=torch.zeros(labels.shape[0],8,3)

    bbox[:,0,:]=-0.5*sizes[:,0,:]

    bbox[:,1,0]=+0.5*sizes[:,0,0]
    bbox[:,1,1]=-0.5*sizes[:,0,1]
    bbox[:,1,2]=-0.5*sizes[:,0,2]

    bbox[:,2,:2]=+0.5*sizes[:,0,:2]
    bbox[:,2,2]=-0.5*sizes[:,0,2]

    bbox[:,3,0]=-0.5*sizes[:,0,0]
    bbox[:,3,1]=+0.5*sizes[:,0,1]
    bbox[:,3,2]=-0.5*sizes[:,0,2]

    bbox[:,4,:2]=-0.5*sizes[:,0,:2]
    bbox[:,4,2]=0.5*sizes[:,0,2]

    bbox[:,5,0]=+0.5*sizes[:,0,0]
    bbox[:,5,1]=-0.5*sizes[:,0,1]
    bbox[:,5,2]=+0.5*sizes[:,0,2]

    bbox[:,6,0]=-0.5*sizes[:,0,0]
    bbox[:,6,1:]=+0.5*sizes[:,0,1:]

    bbox[:,7,:]=+0.5*sizes[:,0,:]

    for i in range(bbox.shape[0]):

        theta=torch.atan(angle[i,0,1]/angle[i,0,0])
        transform_matrix=torch.tensor(
            [
            [torch.cos(theta),torch.sin(theta),0],
            [-torch.sin(theta),torch.cos(theta),0],
            [0,0,1]
            ]
        )
        bbox[i]=torch.matmul(bbox[i],transform_matrix)

    bbox+=centers.detach().cpu()
    #print(bbox.shape)
    return bbox



#def calc_iou()

if __name__ == '__main__':
    main()
