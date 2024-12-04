# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import yaml
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.utils.box_utils import create_bbx, project_box3d, nms_rotated
from opencood.visualization import vis_utils, my_vis, simple_vis, simple_vis_ref
from tqdm import tqdm
import math
torch.multiprocessing.set_sharing_strategy('file_system')
from opencood.loss.point_pillar_uncertainty_loss_outunc import PointPillarUncertaintyLoss

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='early',
                        help='no, no_w_uncertainty, late, early,hybrid or intermediate')
    parser.add_argument('--is_hybrid', action='store_true',default=False)
    parser.add_argument('--save_vis_interval', type=int, default=20,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="140.8,40",
                        help="detection range is [-140.8,+140.8m, -40m, +40m]")
    parser.add_argument('--modal', type=int, default=0,
                        help='used in heterogeneous setting, 0 lidaronly, 1 camonly, 2 ego_lidar_other_cam, 3 ego_cam_other_lidar')
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")

    parser.add_argument('--note', default="", type=str, help="any other thing?")

    parser.add_argument('--sample_method', type=str, default="none",
                        help="the method to downsample the point cloud")
    parser.add_argument('--store_boxes', default=False,action='store_true',
                        help= "store detection boxes and gt boxes")
    parser.add_argument('--save_unc', default=False,action='store_true')
    parser.add_argument('--save_diffxy', default=False,action='store_true')
    parser.add_argument('--sampling_rate', type=float, default=1.0)
    parser.add_argument('--sampler_path', type=str, default=None)
    parser.add_argument('--box_ratio',type=float,default=1.0)
    parser.add_argument('--background_ratio',type=float,default=1.0)
    parser.add_argument('--expansion_ratio',type=float,default=20.0)
    parser.add_argument('--w2cthreshold',type=float,default=0.01)
    parser.add_argument('--vis_score',type=str,default='confidence',
                        help='confidence or uncertainty')
    parser.add_argument('--pose_err',type=float,default=0)
    parser.add_argument('--pose_index',type=float,default=1)
    parser.add_argument('--thre',type=float,default=1)

    

                    
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single','hybrid','intermediate_with_comm'] 
    assert opt.sample_method in ['confidence_map','gt','unc','none']
    hypes = yaml_utils.load_yaml(None, opt)
    if opt.fusion_method == 'intermediate':
        hypes['model']['args']['fusion_args']['communication']['thre']=opt.w2cthreshold
    hypes.update({'sample_method': opt.sample_method})
    hypes.update({'sampling_rate': opt.sampling_rate})
    hypes.update({'store_boxes': opt.store_boxes})
    hypes.update({'model_dir':opt.model_dir})
    hypes.update({'sampler_path':opt.sampler_path})
    hypes.update({'is_hybrid':opt.is_hybrid})
    hypes.update({'expansion_ratio':opt.expansion_ratio})
    hypes.update({'box_ratio':opt.box_ratio})
    hypes.update({'background_ratio':opt.background_ratio})
    hypes.update({'late_choose_method':False})

    hypes.update({'pose_err':opt.pose_err})
    hypes.update({'pose_index':opt.pose_index})
    hypes['model']['args']['fusion_args']['communication']['thre']=opt.thre
    # downsample
    #hypes['preprocess']['down_sample']=0.5
    
    if 'heter' in hypes:
        if opt.modal == 0:
            hypes['heter']['lidar_ratio'] = 1
            hypes['heter']['ego_modality'] = 'lidar'
            opt.note += '_lidaronly' 

        if opt.modal == 1:
            hypes['heter']['lidar_ratio'] = 0
            hypes['heter']['ego_modality'] = 'camera'
            opt.note += '_camonly' 
            
        if opt.modal == 2:
            hypes['heter']['lidar_ratio'] = 0
            hypes['heter']['ego_modality'] = 'lidar'
            opt.note += 'ego_lidar_other_cam'

        if opt.modal == 3:
            hypes['heter']['lidar_ratio'] = 1
            hypes['heter']['ego_modality'] = 'camera'
            opt.note += '_ego_cam_other_lidar'

        x_min, x_max = -140.8, 140.8
        y_min, y_max = -40, 40
        opt.note += f"_{x_max}_{y_max}"
        hypes['fusion']['args']['grid_conf']['xbound'] = [x_min, x_max, hypes['fusion']['args']['grid_conf']['xbound'][2]]
        hypes['fusion']['args']['grid_conf']['ybound'] = [y_min, y_max, hypes['fusion']['args']['grid_conf']['ybound'][2]]
        hypes['model']['args']['grid_conf'] = hypes['fusion']['args']['grid_conf']

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]
        
        hypes['preprocess']['cav_lidar_range'] =  new_cav_range
        hypes['postprocess']['anchor_args']['cav_lidar_range'] = new_cav_range
        hypes['postprocess']['gt_range'] = new_cav_range
        hypes['model']['args']['lidar_args']['lidar_range'] = new_cav_range
        if 'camera_mask_args' in hypes['model']['args']:
            hypes['model']['args']['camera_mask_args']['cav_lidar_range'] = new_cav_range

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

        
    
    hypes['validate_dir'] = hypes['test_dir']
    # if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
    #     assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    ego='ego'
    
    

    # if opt.fusion_method == 'early':
    #     #hypes['model']['core_method'] = 'center_point'
    #     hypes['fusion']['core_method'] = 'early'
    # elif opt.fusion_method == 'hybrid':
    #     hypes['model']['core_method'] = 'center_point_hybrid'
    #     hypes['fusion']['core_method'] = 'hybrid'
    #     #ego='hybrid'
    
        

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    
    opt.pose_index=0.4
    
    # hypes['pose_err']=0.3
    # add noise to pose.
    noise_setting = OrderedDict()
    if(hypes['pose_err']!=0) :
        pos_std = hypes['pose_err']
        rot_std = hypes['pose_err']
        pos_mean = 0
        rot_mean = 0
        if(opt.fusion_method == 'late') :
            noise_setting['pred_box_noise']= True
            noise_setting['add_noise'] = False
        else:
            noise_setting['pred_box_noise']= False
            noise_setting['add_noise'] = True
    else:
        pos_std = 0
        rot_std = 0
        pos_mean = 0
        rot_mean = 0
        noise_setting['pred_box_noise']=False
        noise_setting['add_noise'] = False
        
    # setting noise
    np.random.seed(303)

    
    print(hypes['fusion']['core_method'])
    hypes['fusion']['core_method'] = opt.fusion_method
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # opencood_dataset_subset = Subset(opencood_dataset, range(640,2100))
    # data_loader = DataLoader(opencood_dataset_subset,
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=12,
                            collate_fn=opencood_dataset.collate_batch_test,
                            prefetch_factor=8,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    if(hypes['sampling_rate']!=1):
        opt.note=opt.note+f"_{hypes['sampling_rate']}"
    if(opt.thre!=1):
        opt.note=opt.note+f"_{opt.thre}"
    
    if(opt.pose_index!=1):
        if 'box_align' in hypes.keys():
            infer_info = opt.fusion_method + opt.note + f"po_err{hypes['pose_err']}_coalign"+f"po-index{opt.pose_index}new"
        else:
            infer_info = opt.fusion_method + opt.note + f"po_err{hypes['pose_err']}_new(commfix)"+f"po-index{opt.pose_index}"
    else:
        if 'box_align' in hypes.keys():
            infer_info = opt.fusion_method + opt.note + f"po_err{hypes['pose_err']}_coalign"
        else:
            infer_info = opt.fusion_method + opt.note + f"po_err{hypes['pose_err']}_newlidarbox(0.4vis)"

    comms=[]
    late_boxes={}
    total_comm_rates = []
    late_boxes={}
    unc_x_dict={}
    unc_y_dict={}
    diff_x_dict={}
    diff_y_dict={}
    for i, batch_data in tqdm(enumerate(data_loader)):
        # if i>300:
        #     break
        #print(f"{infer_info}_{i}")
        # if i>10:
        #     break
        if batch_data is None:
            print("none data")
            late_boxes.update({i:None})
            unc_x_dict.update({i:None})
            unc_y_dict.update({i:None})
            diff_x_dict.update({i:None})
            diff_y_dict.update({i:None})
            
            continue
        with torch.no_grad():
            batch_data=train_utils.to_device(batch_data,device)
            if 'points_num' in batch_data[ego].keys():
                points_num= batch_data[ego]['points_num']
            else:
                points_num = 0
            
            final_result=[]
            final_score=[]
            final_unc=[]
            if opt.sample_method == 'unc' and opt.is_hybrid:
                non_ego_boxes=batch_data['ego']['non_ego_boxes']
                non_ego_score=batch_data['ego']['non_ego_score']
                non_ego_unc=batch_data['ego']['non_ego_unc']
                
                if non_ego_boxes!=None:
                    
                    final_result.append(non_ego_boxes)
                    final_score.append(non_ego_score*0.8)
                    non_ego_unc=non_ego_unc[0]+non_ego_unc[1]
                    final_unc.append(non_ego_unc)
                    points_num=points_num+non_ego_boxes.shape[0]*8


            batch_data = train_utils.to_device(batch_data, device)

            if opt.fusion_method == 'late':
                #print("late fusion")
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                    model,
                                                    opencood_dataset)
            elif opt.fusion_method == 'hybrid':
                #print("hybrid fusion")
                infer_result = inference_utils.inference_hybrid_fusion(batch_data,
                                                        model,
                                                        opencood_dataset,opt.sampling_rate)
                
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                comm_rate=infer_result['output_dict']['ego']['comm_rate']
            elif opt.fusion_method == 'intermediate_with_comm':
                infer_result = \
                    inference_utils.inference_intermediate_fusion_withcomm(batch_data,
                                                                  model,
                                                                  opencood_dataset)
                total_comm_rates.append(infer_result['comm_rates'])
                
                

                                                            
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')


            
            if opt.store_boxes and opt.fusion_method=='late':
                
                
                pred_box_tensor = infer_result['ego']['pred_box_tensor']
                gt_box_tensor = infer_result['ego']['gt_box_tensor']
                pred_score = infer_result['ego']['pred_score']

            else:
                pred_box_tensor = infer_result['pred_box_tensor']
                gt_box_tensor = infer_result['gt_box_tensor']
                pred_score = infer_result['pred_score']

            if opt.store_boxes:
                infer_result=train_utils.to_device(infer_result,'cpu')
                late_boxes.update({i:infer_result})

            if opt.sample_method=='unc' and pred_box_tensor!=None:
                
                unc_x=infer_result['unc_x']
                unc_y=infer_result['unc_y']
                
                if(opt.save_unc):
                    unc_x_dict.update({i:unc_x.cpu().numpy().tolist()})
                    unc_y_dict.update({i:unc_y.cpu().numpy().tolist()})
                
                ego_unc=unc_x**2+unc_y**2
                final_result.append(pred_box_tensor)
                final_score.append(pred_score)
                final_unc.append(ego_unc)
                
                pred_box_tensor=torch.cat(final_result)
                
                pred_score=torch.cat(final_score,0)
                final_unc=torch.cat(final_unc,0)
                
                unc_score=-final_unc
                keep_index=nms_rotated(pred_box_tensor,
                                                    pred_score,
                                                    0.15
                                                    )
                pred_box_tensor=pred_box_tensor[keep_index]
                pred_score=pred_score[keep_index]
                unc_score=unc_score[keep_index]
                infer_result['pred_box_tensor']=pred_box_tensor
                infer_result['pred_score']=pred_score
                infer_result['unc_score']=unc_score

            # print(points_num)
            if 'points_num' in infer_result.keys():
                points_num= infer_result['points_num']
            # print(points_num)


            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.7)
            
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if opt.save_diffxy:
                diff_x_dict.update({i:infer_result['diff_x'].cpu().numpy().tolist()})
                diff_y_dict.update({i:infer_result['diff_y'].cpu().numpy().tolist()})
            
            
            # if opt.save_diffxy:
                
            #     ouput_dict = model(batch_data['ego'])
            #     loss_func=PointPillarUncertaintyLoss(hypes['loss']['args'])
                
                
            #     diff_xy=loss_func(ouput_dict, batch_data['ego']['label_dict'])
            #     print(batch_data['ego']['label_dict']['targets'].shape)
            #     print(diff_xy.shape)
            #     diff_x_dict.update({i:diff_xy[0,:,0].cpu().numpy().tolist()})
            #     diff_y_dict.update({i:diff_xy[0,:,1].cpu().numpy().tolist()})
                
            
            if not opt.no_score and not opt.store_boxes:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, lidar_agent_record = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "lidar_agent_record": lidar_agent_record})
            
            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand)
                if hypes['fusion']['dataset'] == 'opv2v':
                    vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                    simple_vis.visualize(infer_result,
                                    batch_data[ego][
                                        'origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand)
                else:
                    if(batch_data[ego]['only_ego']):
                        vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                        simple_vis.visualize(infer_result,
                                        batch_data[ego][
                                            'origin_lidar'][0],
                                        hypes['postprocess']['gt_range'],
                                        vis_save_path,
                                        method='bev',
                                        left_hand=left_hand)
                    
                    else:
                        vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                        simple_vis_ref.visualize(infer_result,
                                        batch_data[ego][
                                            'ego_lidar'][0],batch_data[ego][
                                            'collaboration_lidar'][0],(142, 207, 201),(255, 190, 122),
                                        hypes['postprocess']['gt_range'],
                                        vis_save_path,
                                        method='bev',
                                        left_hand=left_hand)
            
            
            if points_num!=0 and points_num!=None:
                temp_comm=points_num
            else:
                temp_comm=0

            if opt.fusion_method == 'intermediate':
                if comm_rate!=0:
                    temp_comm=comm_rate
                else:
                    temp_comm=0
                

            comms.append(temp_comm)
        torch.cuda.empty_cache()
    ver='_all'
    if(opt.save_unc):
        
        print('unc saved')
        with open(os.path.join(opt.model_dir,f'unc_x{ver}.yaml'), "w", encoding="utf-8") as file:
            yaml.dump(unc_x_dict, file, allow_unicode=True) 
        with open(os.path.join(opt.model_dir,f'unc_y{ver}.yaml'), "w", encoding="utf-8") as file:
            yaml.dump(unc_y_dict, file, allow_unicode=True) 
    
    average_num=sum(comms)/len(comms)
    if average_num==0:
        comm=0
    else:
        if hypes['fusion']['dataset'] == 'dair_v2x' and opt.fusion_method == 'intermediate':
            comm=math.log2(average_num*100*252*64)
        else:
            comm=math.log2(average_num*16)

    if opt.store_boxes:
        print('saving late boxes')     
        np.save(os.path.join(opt.model_dir, 'val_box_non_ego.npy'),late_boxes)
    else:
        if(opt.fusion_method=='intermediate_with_comm'):
            if len(total_comm_rates) > 0:
                comm_rates = (sum(total_comm_rates)/len(total_comm_rates)).item()
            else:
                comm_rates = 0
            _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                    opt.model_dir, comm_rates,infer_info)
            print('comm',comm_rates)
        else:
            _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                    opt.model_dir, comm,infer_info)
                print('comm',comm)

if __name__ == '__main__':

    main()
