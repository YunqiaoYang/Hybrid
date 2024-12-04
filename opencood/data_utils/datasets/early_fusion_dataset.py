# early fusion dataset
import torch
import numpy as np
from opencood.utils.pcd_utils import downsample_lidar_minimum
import math
from collections import OrderedDict
import os
from opencood.models.regressor import Sampler
from opencood.utils import box_utils
from opencood.utils.common_utils import merge_features_to_dict
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.heter_utils import AgentSelector
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.tools import train_utils, inference_utils
from opencood.utils.fast_sampling import fast_sampling
import random
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pose_utils import add_noise_box
from opencood.utils.common_utils import read_json


def getEarlyFusionDataset(cls):
    class EarlyFusionDataset(cls):
        """
        This dataset is used for early fusion, where each CAV transmit the raw
        point cloud to the ego vehicle.
        """
        

        def __init__(self, params, visualize, train=True):
            super(EarlyFusionDataset, self).__init__(params, visualize, train)
            self.supervise_single = True if ('supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) \
                                        else False
            #assert self.supervise_single is False
            self.proj_first = False if 'proj_first' not in params['fusion']['args']\
                                         else params['fusion']['args']['proj_first']
            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)
            self.heterogeneous = False
            self.sample_method = params['sample_method']
            self.sampling_rate = params['sampling_rate']
            self.is_hybrid = params['is_hybrid']
            self.store_boxes = params.get("store_boxes", False)
            self.use_random=params['late_choose_method']#True for random
            
            print('sampling rate: {}'.format(self.sampling_rate))
            print('sample method: {}'.format(self.sample_method))
            
            if self.sample_method == 'unc':
                model_dir=params['model_dir']
                self.detect_result=np.load(os.path.join(model_dir, 'val_box_non_ego.npy'),allow_pickle=True).item()
                
                self.sampling_rate=params['sampling_rate']
                self.expansion_ratio=params['expansion_ratio']
                self.box_ratio=params['box_ratio']
                self.background_ratio=params['background_ratio']
                print("box ratio: {}".format(self.box_ratio))
                if params['fusion']['dataset'] == 'opv2v':
                    self.dataset='opv2v'
                    confidences=[]
                    for _, value in self.detect_result.items():
                        
                        for _,cav in value.items():
                            if cav!=None:
                                confidence=cav['pred_score']
                                
                                if confidence!=None:
                                    
                                    confidences.append(confidence)
                    confidences=np.concatenate(confidences,axis=0)
                    indices = np.argsort(confidences)[::-1]
                    self.confidences_desc=confidences[indices]
                    # print(self.confidences)
                    self.confidence_threshold=self.confidences_desc[int((len(self.confidences_desc))*self.box_ratio)]
                    self.score_threshold=params['postprocess']['target_args']['score_threshold']

                    print('confidences thr: {}'.format(self.confidence_threshold))
                    # print('uncs thr: {}'.format(self.unc_threshold))
                else:
                    self.dataset='dairv2x'
                    scores=[]
                    for _, value in self.detect_result.items():
                        if value['pred_score'] is not None:
                            scores.append(value['pred_score'])
                    scores=np.concatenate(scores,axis=0)
                    scores=np.sort(scores)
                    
                    self.score_threshold=scores[int((len(scores)-1)*(1-self.box_ratio))]
                    self.score_threshold=max(self.score_threshold,0.3)
                    print('conf thr: {}'.format(self.score_threshold))
                    if 'sampler_path' in params and params['sampler_path'] != None:
                        self.sampler=Sampler()
                        print(params['sampler_path'])
                        self.sampler.load_state_dict(torch.load(params['sampler_path']))
                        self.use_sampler=True
                        print('sampler loaded')
                    else:
                        self.use_sampler=False
            
            if 'heter' in params:
                self.heterogeneous = True
                self.selector = AgentSelector(params['heter'], self.max_cav)
                
            self.box_align = False
            if "box_align" in params:
                self.box_align = True
                self.stage1_result_path = params['box_align']['train_result'] if train else params['box_align']['val_result']
                self.stage1_result = read_json(self.stage1_result_path)
                self.box_align_args = params['box_align']['args']


        def point_in_polygon(self, points, polygon):
            """
            Determine if a set of points are inside a set of polygons.

            Args:
            - points: Tensor of shape (N, 2) representing the N points
            - polygon: Tensor of shape (M, N_i, 2) representing the M polygons, where N_i is the number of polygon in the i-th polygon

            Returns:
            - in_polygon: Tensor of shape (N,) indicating if each point is inside the polygon.
            - in_which_box: Tensor of shape (N,) indicating which box the point inside
            """
            
            # Repeat the first vertex of each polygon to the end so we can easily calculate the winding number.
            # print("point",points.shape)
            # print("polygon",polygon.shape)
            polygon = torch.cat([polygon, polygon[:, :1, :]], dim=1)
            polygon = polygon.unsqueeze(0).repeat(points.shape[0], 1, 1, 1)
            
            # Repeat the points along the polygon dimension to match the shape of the polyggon.
            points = points.unsqueeze(1).unsqueeze(2).repeat(1, polygon.shape[1], 1, 1)
            # Compute the winding number for each point by checking if the ray from the point to the right crosses each edge of the polygon in the positive direction.
            
            
            i1=(polygon[:, :, :-1, 0]-polygon[:,:,1:,0])*(points[:,:,:,0]-polygon[:,:,1:,0])>0
            i2=(polygon[:, :, :-1, 0]-polygon[:,:,1:,0])*(points[:,:,:,0]-polygon[:,:,:-1,0])<0
            i3=(points[:,:,:,1]-polygon[:,:,1:,1])<(polygon[:,:,:-1,1]-polygon[:,:,1:,1])*(points[:,:,:,0]-polygon[:,:,1:,0])/(polygon[:,:,:-1,0]-polygon[:,:,1:,0])
            ray_intersections=i1 & i2 & i3
            ray_intersections=ray_intersections.sum(2)
            in_poly=(ray_intersections[:,:]%2)==1
            # print(in_poly[0,:])
            
            in_which_box=torch.argmax(in_poly.float(),dim=1)
            # meaningful only when this point is selected 
            # print(in_which_box)     
            # Sum the winding number along the edge dimension and check if the winding number is non-zero, which indicates the point is inside the polygon.
            in_polygon = in_poly.sum(1) > 0
            out_polygon = ~in_polygon
            
            
            return in_polygon,out_polygon,in_which_box

        def fps(self,points, num_samples):
            if num_samples >= points.shape[0]:
                return points
            else:
                dist = torch.sum((points[:, None, :] - points[None, :, :])**2, dim=-1)
                print(dist.shape)
                selected = torch.zeros(num_samples, dtype=torch.long) # Index of selected points
                selected[0] = torch.randint(points.shape[0], size=(1,)) # Select first point randomly
                for i in range(1, num_samples):
                    dist_to_selected = dist[selected[:i], :].min(dim=0)[0]
                    next_selected = torch.argmax(dist_to_selected, dim=0)
                    selected[i] = next_selected
                points=points[selected]
                return points

        
        def box_filtering(self,inputlidar,box):
            
            #print(inputlidar.shape)
            
            inputlidar=torch.tensor(inputlidar)
            box=torch.tensor(box[:,:4,:2])
            mask,_=self.point_in_polygon(inputlidar[:,0:2],box)
            #print(mask)
            inputlidar=inputlidar[mask]
            points_num=inputlidar.shape[0]
            
            
            #2.1-downsapling-random
            sampling_idx = np.sort(np.random.choice(range(points_num), int(points_num*self.sampling_rate)))
            # inputlidar=inputlidar[sampling_idx]
            
            #2.2-downsampling-fps
            #inputlidar = self.fps(inputlidar, self.sampling_num)
            
            
            
            output=inputlidar
            points_num=output.shape[0]
            
            return output,points_num

        

        def sample_process(self,inputlidar,box):
            return None

        def unc_filtering(self,inputlidar,tensors,scores,use_sampler=False):
            
            # print(inputlidar.shape)
            # print(tensors.shape)
            

            inputlidar=torch.tensor(inputlidar)
            
            box=tensors[:,:4,:2].clone().detach()
            
            # average=torch.mean(box,keepdim=True,dim=1)
            

            # diff=box-average

            # print(box)
            
            # diff[:,:,0]=diff[:,:,0]*(1+self.expansion_ratio*scores[:1]).T
            # diff[:,:,1]=diff[:,:,1]*(1+self.expansion_ratio*scores[1:]).T

            # box=average+diff
            if self.dataset == 'opv2v':
                if(scores.size(0)==2):
                    sigma_x=torch.sqrt(scores[0,:])
                    sigma_y=torch.sqrt(scores[1,:])
                    box[:,0,0]+=self.expansion_ratio*sigma_x[:,]
                    box[:,1,0]+=self.expansion_ratio*sigma_x[:,]
                    box[:,2,0]-=self.expansion_ratio*sigma_x[:,]
                    box[:,3,0]-=self.expansion_ratio*sigma_x[:,]
                    
                    box[:,1,1]+=self.expansion_ratio*sigma_y[:,]
                    box[:,2,1]+=self.expansion_ratio*sigma_y[:,]
                    box[:,0,1]-=self.expansion_ratio*sigma_y[:,]
                    box[:,3,1]-=self.expansion_ratio*sigma_y[:,]

            
            else:
                diff=box-average
                scores=torch.sqrt(scores)

                x_sign=torch.sign(diff[:,:,0])
                rescale_x=scores[:1].T*x_sign
                y_sign=torch.sign(diff[:,:,1])
                rescale_y=scores[1:].T*y_sign

                diff[:,:,0]=diff[:,:,0]+rescale_x
                diff[:,:,1]=diff[:,:,1]+rescale_y

                box=average+diff
            if box.shape[0]==0:
                return torch.zeros(0,4),0
            mask,background,in_which_box=self.point_in_polygon(inputlidar[:,0:2],box)

            backgroundlidar=inputlidar

            inputlidar=inputlidar[mask]
            box_num=in_which_box[mask]

            points_num=inputlidar.shape[0]
           
            xy_scores=(scores[0]+scores[1]).T
            prob=xy_scores[box_num]
            #normalization
            prob=(prob/sum(prob)).cpu().numpy()

            #2.1-downsapling-random
            
            
            if use_sampler:
                
                
                unique_boxes=torch.unique(box_num)
                num_of_boxes=unique_boxes.shape[0]
                temp=[torch.empty(0,4) for _ in range(num_of_boxes)]
                for i in range(num_of_boxes):
                    box_mask=(box_num==unique_boxes[i]).unsqueeze(1).expand_as(inputlidar)
                    temp[i]=inputlidar[box_mask].view(-1, inputlidar.shape[1])

                for i,box_lidar in enumerate(temp):
                    
                    if box_lidar.shape[0]<100:
                        temp[i]=box_lidar
                    else:
                        
                        gen_point=self.sampler(box_lidar.unsqueeze(0).unsqueeze(0))
                        temp[i]=gen_point.squeeze(0).squeeze(0)

                    
                inputlidar=torch.cat(temp,dim=0).detach()
            
                sampling_idx=np.sort(random.sample(range(inputlidar.shape[0]),k=max(int((inputlidar.shape[0])*self.sampling_rate)-1,0)))
                inputlidar=inputlidar[sampling_idx]
            else:   
                if(self.use_random):
                    sampling_idx = np.sort(random.sample(range(inputlidar.shape[0]), int(inputlidar.shape[0]*self.sampling_rate)))
                    inputlidar=inputlidar[sampling_idx]
                else:
                #print(inputlidar.shape)
                #2.3-downsampling-uncertainty
                #print(self.sampling_rate)
                    if self.dataset == 'opv2v':
                        sampling_idx=np.sort(random.sample(range   (points_num), k=max(int((points_num)*self. sampling_rate)-1,0)))
                        inputlidar=inputlidar[sampling_idx]
                    else:
                        points_num=inputlidar.shape[0]
                        assert points_num==prob.shape[0]
                        if prob.shape[0]>0:
                            sampling_idx=np.sort(random.choices(range(points_num), k=max(int((points_num)*self.sampling_rate)-1,0),weights=prob))
                            inputlidar=inputlidar[sampling_idx]
                    # sampling_idx=np.sort(random.sample(range(points_num), k=max(int((points_num)*self.sampling_rate)-1,0)))
                    # inputlidar=inputlidar[sampling_idx]
                # #2.2-downsampling-graph
                # sampling_idx=fast_sampling(inputlidar[:,:3],int(points_num*self.sampling_rate),1)
                # inputlidar=inputlidar[sampling_idx]
            
            # random sampling background pc

            sampling_idx=np.sort(random.sample(range(backgroundlidar.shape[0]),k=max(int((backgroundlidar.shape[0])*self.background_ratio)-1,0)))
            backgroundlidar=backgroundlidar[sampling_idx]
            output=torch.cat([inputlidar,backgroundlidar],dim=0)

            points_num=output.shape[0]
            # print(points_num)

            return output,points_num

        # def confidencemap_filtering(self,confidence_map,inputlidar,threshold=1):
            
        #     #temp_confidence_map=confidence_map[1,0,:,:].reshape(-1)
        #     # threshold=np.percentile(temp_confidence_map,1-threshold)
        #     confidence_map=confidence_map.sigmoid()
        #     confidence_map=self.kernel(confidence_map)
        #     #confidence_map=torch.tensor(self.test_confidence_map)
        #     #temp_confidence_map=confidence_map.reshape(-1).cpu().numpy()
        #     #threshold_c=np.percentile(temp_confidence_map,100-100*threshold)
        #     threshold_c=threshold
        #     # fig=plt.matshow(confidence_map.cpu(), cmap=plt.cm.Blues)
        #     # plt.colorbar(fig)
        #     # plt.savefig('/GPFS/data/juntongpeng/confidence_map.png')
        #     #print(max(confidence_map.reshape(-1)),min(confidence_map.reshape(-1)))
        #     confidence_map=confidence_map[1,0,:,:]>threshold_c
        #     confidence_map=confidence_map.cpu().numpy()
            
            
        #     position=np.floor(inputlidar[:,0:2]*1.25).astype(np.int)
            
            
        #     temp_index=np.where(abs(position[:,0])>=126)[0]   
        #     position=np.delete(position,temp_index,0)
        #     inputlidar=np.delete(inputlidar,temp_index,0)
        #     temp_index=np.where(abs(position[:,1])>=50)[0]
        #     position=np.delete(position,temp_index,0)
        #     inputlidar=np.delete(inputlidar,temp_index,0)
            
        #     position[:,0]=position[:,0]+125
        #     position[:,1]=position[:,1]+49
        #     #print(min(position[:,0]),max(position[:,0]),min(position[:,1]),max(position[:,1]))
            
           
        #     temp_index=confidence_map[position[:,1],position[:,0]]
            
         
        #     inputlidar=inputlidar[temp_index,:]
        #     points_num=inputlidar.shape[0]


        #     return inputlidar,points_num
        

        
        
        def __getitem__(self, idx):
            #print('here early dataset')
            base_data_dict = self.retrieve_base_data(idx)
            #print(self.params['noise_setting'])
            base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])
            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            ego_id = -1
            ego_lidar_pose = []

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    break
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            projected_lidar_stack = []
            object_stack = []
            object_id_stack = []
            direct_tensors=[]
            direct_scores=[]
            direct_uncs=[]
            ego_lidar_stack=[]
            collaboration_lidar_stack=[]

            only_ego=False
            points_num=0
            # loop over all CAVs to process information
            too_far = []
            lidar_pose_list = []
            lidar_pose_clean_list = []
            cav_id_list = []
            points_num=0
            
            cav_dict=self.detect_result[idx].keys()
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                if cav_id == ego_id and self.store_boxes:
                    continue 
                #two line for coalign
                distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                            ego_lidar_pose[0]) ** 2 + (
                                    selected_cav_base['params'][
                                        'lidar_pose'][1] - ego_lidar_pose[
                                        1]) ** 2)

                # if distance is too far, we will just skip this agent
                if distance > self.params['comm_range']:
                    too_far.append(cav_id)
                    continue

                lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
                cav_id_list.append(cav_id) 
            
            for cav_id in too_far:
                base_data_dict.pop(cav_id)
                
            ########## Updated by Yifan Lu 2022.1.26 ############
            # box align to correct pose.
            # stage1_content contains all agent. Even out of comm range.
            if self.box_align and str(idx) in self.stage1_result.keys():
                from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np
                stage1_content = self.stage1_result[str(idx)]
                if stage1_content is not None:
                    all_agent_id_list = stage1_content['cav_id_list'] # include those out of range
                    all_agent_corners_list = stage1_content['pred_corner3d_np_list']
                    all_agent_uncertainty_list = stage1_content['uncertainty_np_list']

                    cur_agent_id_list = cav_id_list
                    cur_agent_pose = [base_data_dict[cav_id]['params']['lidar_pose'] for cav_id in cav_id_list]
                    cur_agnet_pose = np.array(cur_agent_pose)
                    cur_agent_in_all_agent = [all_agent_id_list.index(cur_agent) for cur_agent in cur_agent_id_list] # indexing current agent in `all_agent_id_list`

                    pred_corners_list = [np.array(all_agent_corners_list[cur_in_all_ind], dtype=np.float64) 
                                            for cur_in_all_ind in cur_agent_in_all_agent]
                    uncertainty_list = [np.array(all_agent_uncertainty_list[cur_in_all_ind], dtype=np.float64) 
                                            for cur_in_all_ind in cur_agent_in_all_agent]

                    if sum([len(pred_corners) for pred_corners in pred_corners_list]) != 0:
                        refined_pose = box_alignment_relative_sample_np(pred_corners_list,
                                                                        cur_agnet_pose, 
                                                                        uncertainty_list=uncertainty_list, 
                                                                        **self.box_align_args)
                        cur_agnet_pose[:,[0,1,4]] = refined_pose 

                        for i, cav_id in enumerate(cav_id_list):
                            lidar_pose_list[i] = cur_agnet_pose[i].tolist()
                            base_data_dict[cav_id]['params']['lidar_pose'] = cur_agnet_pose[i].tolist()
            
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                if cav_id == ego_id and self.store_boxes and False:
                    continue
                  
                
                if self.sample_method == 'confidence_map':
                    if cav_id!=ego_id:
                        infra_lidar=selected_cav_base['lidar_np']
                        filtered_lidar,points_num=self.confidencemap_filtering(self.confidence_map_dict[idx],infra_lidar,self.threshold)
                        selected_cav_base['lidar_np']=filtered_lidar

                selected_cav_processed = self.get_item_single_car(
                    selected_cav_base,
                    ego_lidar_pose)
                # all these lidar and object coordinates are projected to ego
                # already.
                #print(self.sample_method)
                if self.sample_method == 'unc':
                    if cav_id!=ego_id:
                        infra_lidar=selected_cav_processed['projected_lidar']
                        sample_dict=self.detect_result[idx]
                        #print(sample_dict.keys())
                        #sample_dictsample_dict['pred_box_tensor']  =box_add_noise(sample_dictsample_dict ['pred_box_tensor'],self.params['noise_setting'])
                        if(self.params ['noise_setting']['add_noise']) :
                            # print('lidar_noise_added')
                            sample_dict['pred_box_tensor']=add_noise_box(sample_dict['pred_box_tensor'],self.params ['noise_setting'])
                        # print(sample_dict['pred_box_tensor'])
                        tensors=sample_dict['pred_box_tensor']

                        unc_x=sample_dict['unc_x']
                        unc_y=sample_dict['unc_y']

                        pred_score=sample_dict['pred_score']
                        #
                        if tensors!=None:
                            uncs=torch.stack([unc_x,unc_y])
                            if self.is_hybrid:
                                
                    
                                unc = uncs[0]**2+uncs[1]**2

                                deliver_mask= unc>= 0
                                #deliver_mask= unc>=0
                                if(self.use_random):
                                    # random_mask
                                    
                                    random_mask=np.sort(random.sample(range(tensors.shape[0]),k=max(int((tensors.shape[0])*self.box_ratio),0)))
                                    
                                    
                                    direct_mask=(pred_score>=self.score_threshold)
                                    
                                    # print(direct_mask)
                                    # print('random')
                                else:
                                    # print('unc')
                                    if self.dataset == 'opv2v':
                                        direct_mask = (pred_score>self.confidence_threshold) & (pred_score>=self.score_threshold)#apply box_ratio
                                    else:
                                        direct_mask =  (pred_score>=self.score_threshold)#apply box_ratio
                                
                                
                                
                                filter_tensor = tensors[deliver_mask]
                                filter_unc = uncs[:,deliver_mask]
                                
                                if filter_tensor.shape[0]!=0:
                                    filtered_lidar,temppoints_num=self.unc_filtering(infra_lidar,filter_tensor,filter_unc,self.use_sampler)
                                else:
                                    filtered_lidar=infra_lidar[0:0]
                                    temppoints_num=0
                                # print(temppoints_num)
                                direct_tensor = tensors[direct_mask]
                                direct_score  = pred_score[direct_mask]
                                direct_unc    = uncs[:,direct_mask]
                                
                                if(self.use_random):
                                    direct_tensor = tensors[random_mask]
                                    direct_score  = pred_score[random_mask]
                                    direct_unc    = uncs[:,random_mask]
                                    
                            else:
                                filter_tensor = tensors

                                direct_tensor = tensors
                                direct_score  = pred_score

                                
                                direct_unc    = uncs

                                filtered_lidar,temppoints_num=self.unc_filtering(infra_lidar,filter_tensor,uncs,self.use_sampler)
                                
                            direct_tensors.append(direct_tensor)
                            direct_scores.append(direct_score)
                            direct_uncs.append(direct_unc)
                            
                        
                        else:
                            filtered_lidar=infra_lidar[0:0]
                            temppoints_num=0
                            
                        # print(filtered_lidar.shape)
                        selected_cav_processed['projected_lidar']=filtered_lidar
                        points_num+=temppoints_num
                        
                else:
                    stat=[]
                    stat=torch.tensor(stat)
                    if cav_id!=ego_id:
                        lidar=selected_cav_processed['projected_lidar']
                        sampling_idx=np.sort(random.sample(range(lidar.shape[0]),k=max(int((lidar.shape[0])*self.sampling_rate),0)))
                        if len(sampling_idx)!=0:
                            lidar=lidar[sampling_idx]
                        else:
                            lidar=lidar[0:0]
                        selected_cav_processed['projected_lidar']=lidar
                        points_num+=selected_cav_processed['projected_lidar'].shape[0]
                # print(selected_cav_processed['projected_lidar'].shape)
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
                if(cav_id==ego_id):
                    ego_lidar_stack.append(selected_cav_processed['projected_lidar'])
                else:
                    collaboration_lidar_stack.append(selected_cav_processed['projected_lidar'])
            # print(len(projected_lidar_stack))
            
            if len(projected_lidar_stack) == 0:
                # no car is within the communication range
                # print('no car within')
                return None
            # exclude all repetitive objects
            unique_indices = \
                [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack)
            object_stack = object_stack[unique_indices]

            # make sure bounding boxes across all frames have the same number
            object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1

            # convert list to numpy array, (N, 4)
            projected_lidar_stack = np.vstack(projected_lidar_stack)
            raw_pls=projected_lidar_stack.copy()
            ego_lidar_stack = np.vstack(ego_lidar_stack)
            ego_pls=ego_lidar_stack.copy()
            if(len(collaboration_lidar_stack)!=0):
                collaboration_lidar_stack = np.vstack(collaboration_lidar_stack)
                collaboration_pls=collaboration_lidar_stack.copy()
            else:
                collaboration_pls=None
                only_ego=True
            # data augmentation
            # projected_lidar_stack, object_bbx_center, mask = \
            #     self.augment(projected_lidar_stack, object_bbx_center, mask)

            # we do lidar filtering in the stacked lidar
            projected_lidar_stack = mask_points_by_range(projected_lidar_stack,
                                                        self.params['preprocess'][
                                                            'cav_lidar_range'])
            # augmentation may remove some of the bbx out of range
            object_bbx_center_valid = object_bbx_center[mask == 1]
            object_bbx_center_valid, range_mask = \
                box_utils.mask_boxes_outside_range_numpy(object_bbx_center_valid,
                                                        self.params['preprocess'][
                                                            'cav_lidar_range'],
                                                        self.params['postprocess'][
                                                            'order'],
                                                        return_mask=True
                                                        )
            mask[object_bbx_center_valid.shape[0]:] = 0
            object_bbx_center[:object_bbx_center_valid.shape[0]] = \
                object_bbx_center_valid
            object_bbx_center[object_bbx_center_valid.shape[0]:] = 0
            unique_indices = list(np.array(unique_indices)[range_mask])

            # pre-process the lidar to voxel/bev/downsampled lidar
            lidar_dict = self.pre_processor.preprocess(projected_lidar_stack)
            # generate the anchor boxes
            anchor_box = self.post_processor.generate_anchor_box()

            # generate targets label
            label_dict = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=anchor_box,
                    mask=mask)

            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': anchor_box,
                'processed_lidar': lidar_dict,
                'label_dict': label_dict})

            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar':
                                                    raw_pls})
                processed_data_dict['ego'].update({'ego_lidar':
                                                    ego_pls})
                processed_data_dict['ego'].update({'collaboration_lidar':
                                                    collaboration_pls})
                processed_data_dict['ego'].update({'only_ego':
                                                    only_ego})
            
            if len(direct_tensors)!=0:
                direct_tensors=torch.cat(direct_tensors,dim=0)
                direct_scores=torch.cat(direct_scores,dim=0)
                direct_uncs=torch.cat(direct_uncs,dim=1)
            else:
                direct_tensors=None
                direct_scores=None
                direct_uncs=None
            
            if self.sample_method == 'unc':
                
                processed_data_dict['ego'].update({'non_ego_boxes':direct_tensors})
                processed_data_dict['ego'].update({'non_ego_score':direct_scores})
                processed_data_dict['ego'].update({'non_ego_unc':direct_uncs})

            
            processed_data_dict['ego'].update({'points_num':points_num})

            return processed_data_dict


        def get_item_single_car(self, selected_cav_base, ego_pose):
            """
            Project the lidar and bbx to ego space first, and then do clipping.

            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
            ego_pose : list
                The ego vehicle lidar pose under world coordinate.

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}

            # calculate the transformation matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['params']['lidar_pose'],
                        ego_pose)

            # retrieve objects under ego coordinates
            object_bbx_center, object_bbx_mask, object_ids = \
                self.generate_object_center([selected_cav_base],
                                                        ego_pose)

            # filter lidar
            lidar_np = selected_cav_base['lidar_np']
            lidar_np = shuffle_points(lidar_np)
            # remove points that hit itself
            lidar_np = mask_ego_points(lidar_np)
            
            # down_sample
            if 'down_sample' in self.params['preprocess'].keys():
                select_idx = np.random.choice(lidar_np.shape[0],int(lidar_np.shape[0]*self.params['preprocess']['down_sample']),replace=False)
                lidar_np = lidar_np[select_idx,:]
            
            # project the lidar to ego space
            lidar_np[:, :3] = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                        transformation_matrix)

            selected_cav_processed.update(
                {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
                'object_ids': object_ids,
                'projected_lidar': lidar_np})

            return selected_cav_processed

        def collate_batch_test(self, batch):
            """
            Customized collate function for pytorch dataloader during testing
            for late fusion dataset.

            Parameters
            ----------
            batch : dict

            Returns
            -------
            batch : dict
                Reformatted batch.
            """
            # currently, we only support batch size of 1 during testing
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            if batch[0] is None:
                return None
            batch = batch[0] # only ego

            output_dict = {}

            for cav_id, cav_content in batch.items():
                output_dict.update({cav_id: {}})
                # shape: (1, max_num, 7)
                object_bbx_center = \
                    torch.from_numpy(np.array([cav_content['object_bbx_center']]))
                object_bbx_mask = \
                    torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
                object_ids = cav_content['object_ids']

                # the anchor box is the same for all bounding boxes usually, thus
                # we don't need the batch dimension.
                if cav_content['anchor_box'] is not None:
                    output_dict[cav_id].update({'anchor_box':
                        torch.from_numpy(np.array(
                            cav_content[
                                'anchor_box']))})
                if self.visualize:
                    origin_lidar = [cav_content['origin_lidar']]
                    ego_lidar = [cav_content['ego_lidar']]
                    if(cav_content['only_ego']==False):
                        collaboration_lidar = [cav_content['collaboration_lidar']]

                # processed lidar dictionary
                processed_lidar_torch_dict = \
                    self.pre_processor.collate_batch(
                        [cav_content['processed_lidar']])
                # label dictionary
                label_torch_dict = \
                    self.post_processor.collate_batch([cav_content['label_dict']])

                # save the transformation matrix (4, 4) to ego vehicle
                transformation_matrix_torch = \
                    torch.from_numpy(np.identity(4)).float()
                transformation_matrix_clean_torch = \
                    torch.from_numpy(np.identity(4)).float()

                output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                            'object_bbx_mask': object_bbx_mask,
                                            'processed_lidar': processed_lidar_torch_dict,
                                            'label_dict': label_torch_dict,
                                            'object_ids': object_ids,
                                            'transformation_matrix': transformation_matrix_torch,
                                            'transformation_matrix_clean': transformation_matrix_clean_torch})

                if self.visualize:
                    origin_lidar = \
                        np.array(
                            downsample_lidar_minimum(pcd_np_list=origin_lidar))
                    origin_lidar = torch.from_numpy(origin_lidar)
                    output_dict[cav_id].update({'origin_lidar': origin_lidar})
                    ego_lidar = \
                        np.array(
                            downsample_lidar_minimum(pcd_np_list=ego_lidar))
                    ego_lidar = torch.from_numpy(ego_lidar)
                    output_dict[cav_id].update({'ego_lidar': ego_lidar})
                    if(cav_content['only_ego']==False):
                        collaboration_lidar = \
                            np.array(
                                downsample_lidar_minimum(pcd_np_list=collaboration_lidar))
                        
                        collaboration_lidar = torch.from_numpy(collaboration_lidar)
                        output_dict[cav_id].update({'collaboration_lidar': collaboration_lidar})
                    else:output_dict[cav_id].update({'collaboration_lidar': None})
                    output_dict[cav_id].update({'only_ego': cav_content['only_ego']})
            if self.sample_method == 'unc':
                output_dict['ego'].update({'non_ego_boxes':batch['ego']['non_ego_boxes']})
                output_dict['ego'].update({'non_ego_score':batch['ego']['non_ego_score']})
                output_dict['ego'].update({'non_ego_unc':batch['ego']['non_ego_unc']})

            output_dict['ego'].update({'points_num':batch['ego']['points_num']})
            return output_dict
        
        def collate_batch_train(self, batch):
            # Intermediate fusion is different the other two
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            processed_lidar_list = []
            image_inputs_list = []
            # used to record different scenario
            label_dict_list = []
            origin_lidar = []
            
            # heterogeneous
            lidar_agent_list = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []
            
            ### 2022.10.10 single gt ####
            if self.supervise_single:
                pos_equal_one_single = []
                neg_equal_one_single = []
                targets_single = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])
                if self.load_lidar_file:
                    processed_lidar_list.append(ego_dict['processed_lidar'])
                if self.load_camera_file:
                    image_inputs_list.append(ego_dict['image_inputs']) # different cav_num, ego_dict['image_inputs'] is dict.
                
                label_dict_list.append(ego_dict['label_dict'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

                ### 2022.10.10 single gt ####
                if self.supervise_single:
                    pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                    neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                    targets_single.append(ego_dict['single_label_dict_torch']['targets'])

                # heterogeneous
                if self.heterogeneous:
                    lidar_agent_list.append(ego_dict['lidar_agent'])

            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_lidar_list)

                if self.heterogeneous:
                    lidar_agent = np.concatenate(lidar_agent_list)
                    lidar_agent_idx = lidar_agent.nonzero()[0].tolist()
                    for k, v in merged_feature_dict.items(): # 'voxel_features' 'voxel_num_points' 'voxel_coords'
                        merged_feature_dict[k] = [v[index] for index in lidar_agent_idx]

                if not self.heterogeneous or (self.heterogeneous and sum(lidar_agent) != 0):
                    processed_lidar_torch_dict = \
                        self.pre_processor.collate_batch(merged_feature_dict)
                    output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})

            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')

                if self.heterogeneous:
                    camera_agent = 1 - lidar_agent
                    camera_agent_idx = camera_agent.nonzero()[0].tolist()
                    if sum(camera_agent) != 0:
                        for k, v in merged_image_inputs_dict.items(): # 'imgs' 'rots' 'trans' ...
                            merged_image_inputs_dict[k] = torch.stack([v[index] for index in camera_agent_idx])
                            
                if not self.heterogeneous or (self.heterogeneous and sum(camera_agent) != 0):
                    output_dict['ego'].update({'image_inputs': merged_image_inputs_dict})
            
            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)

            # for centerpoint
            label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask})

            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # add pairwise_t_matrix to label dict

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask,
                                    'label_dict': label_torch_dict,
                                    'object_ids': object_ids[0]})


            if self.visualize:
                origin_lidar = \
                    np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            if self.supervise_single:
                output_dict['ego'].update({
                    "label_dict_single" : 
                        {"pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                        "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                        "targets": torch.cat(targets_single, dim=0)}
                })

            if self.heterogeneous:
                output_dict['ego'].update({
                    "lidar_agent_record": torch.from_numpy(np.concatenate(lidar_agent_list)) # [0,1,1,0,1...]
                })

            return output_dict

        def post_process(self, data_dict, output_dict):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.

            output_dict :dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)
            #if self.store_boxes:
            if True:
                pred_box_tensor, pred_score, uncertainty ,diffs= \
                    self.post_processor.post_process(data_dict, output_dict)
                
                if uncertainty!=None:
                    uncertainty=torch.exp(uncertainty)
                    uncx=uncertainty[:,0]
                    uncy=uncertainty[:,1]
                    
                else:
                    uncx=None
                    uncy=None
                return pred_box_tensor, pred_score, gt_box_tensor,uncx,uncy,diffs[:,0],diffs[:,1]
            else:
                pred_box_tensor, pred_score = \
                    self.post_processor.post_process(data_dict, output_dict)
                return pred_box_tensor, pred_score, gt_box_tensor
            
            

            

    return EarlyFusionDataset

