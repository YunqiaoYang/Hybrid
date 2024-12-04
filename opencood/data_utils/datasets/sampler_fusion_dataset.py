# early fusion dataset
import torch
import numpy as np
from opencood.utils.pcd_utils import downsample_lidar_minimum
import math
from collections import OrderedDict
import os
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
from opencood.tools import train_utils, inference_utils

import random
def corner_pos2box(corner_pos):
        
        #corner_pos: (N,8,3)
        #output: (N,26,3)
        output=torch.zeros(corner_pos.shape[0],26,3)
        output[:,0:8,:]=corner_pos
        #calc face center
        output[:,8,:]=torch.mean(corner_pos[:,0:4,:],dim=1)
        output[:,9,:]=torch.mean(corner_pos[:,4:8,:],dim=1)
        output[:,10,:]=torch.mean(corner_pos[:,[0,1,4,5],:],dim=1)
        output[:,11,:]=torch.mean(corner_pos[:,[2,3,6,7],:],dim=1)
        output[:,12,:]=torch.mean(corner_pos[:,[1,2,5,6],:],dim=1)
        output[:,13,:]=torch.mean(corner_pos[:,[0,3,4,7],:],dim=1)
        #calc edge center
        output[:,14,:]=torch.mean(corner_pos[:,[0,1],:],dim=1)
        output[:,15,:]=torch.mean(corner_pos[:,[2,3],:],dim=1)
        output[:,16,:]=torch.mean(corner_pos[:,[4,5],:],dim=1)
        output[:,17,:]=torch.mean(corner_pos[:,[6,7],:],dim=1)
        output[:,18,:]=torch.mean(corner_pos[:,[0,3],:],dim=1)
        output[:,19,:]=torch.mean(corner_pos[:,[1,2],:],dim=1)
        output[:,20,:]=torch.mean(corner_pos[:,[4,7],:],dim=1)
        output[:,21,:]=torch.mean(corner_pos[:,[5,6],:],dim=1)
        output[:,22,:]=torch.mean(corner_pos[:,[0,4],:],dim=1)
        output[:,23,:]=torch.mean(corner_pos[:,[1,5],:],dim=1)
        output[:,24,:]=torch.mean(corner_pos[:,[2,6],:],dim=1)
        output[:,25,:]=torch.mean(corner_pos[:,[3,7],:],dim=1)

        return output
def angle_augment(output_dict):
        lidar_list=output_dict['ego']['processed_lidar']
        bbox_list=output_dict['ego']['bbox']
        new_lidar_list=[]
        for i,lidar in enumerate(lidar_list):
            bbox_center_3d=torch.mean(bbox_list[i],dim=0,keepdim=True)

            bbox_center_4d=torch.zeros(1,4)
            bbox_center_4d[:,:3]=bbox_center_3d
            bbox_center_4d[:,3]=0

            random_theta=(random.random()*2-1)*math.pi
            rotate_matrix_4D=torch.tensor([[math.cos(random_theta),-math.sin(random_theta),0,0],
                                        [math.sin(random_theta),math.cos(random_theta),0,0],
                                        [0,0,1,0],
                                        [0,0,0,1]])
            rotate_matrix_3D=rotate_matrix_4D[:3,:3]
            
            temp_lidar=lidar-bbox_center_4d
            temp_bbox=bbox_list[i]-bbox_center_3d
            temp_lidar=torch.matmul(temp_lidar,rotate_matrix_4D)
            temp_bbox=torch.matmul(temp_bbox,rotate_matrix_3D)
            lidar=temp_lidar+bbox_center_4d
            bbox_list[i]=temp_bbox+bbox_center_3d
            new_lidar_list.append(lidar)
        
        output_dict['ego']['processed_lidar']=new_lidar_list
        output_dict['ego']['bbox']=bbox_list   
        return output_dict
def getSamplerFusionDataset(cls):
    class SamplerFusionDataset(cls):
        """
        This dataset is used for early fusion, where each CAV transmit the raw
        point cloud to the ego vehicle.
        """
        def __init__(self, params, visualize, train=True):
            super(SamplerFusionDataset, self).__init__(params, visualize, train)
            print("here is the sampler dataset")
            self.supervise_single = True if ('supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) \
                                        else False
            #assert self.supervise_single is False
            self.proj_first = False if 'proj_first' not in params['fusion']['args']\
                                         else params['fusion']['args']['proj_first']
            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)
            self.heterogeneous = False
            self.is_train=train

            model_dir = params['model_dir']
            if train:
                self.detect_result=np.load(os.path.join(model_dir, 'train_all_boxes.npy'),allow_pickle=True).item()
            else:
                self.detect_result=np.load(os.path.join(model_dir, 'val_all_boxes.npy'),allow_pickle=True).item()
            
            print(self.detect_result.keys())


            if 'heter' in params:
                self.heterogeneous = True
                self.selector = AgentSelector(params['heter'], self.max_cav)

        

        def point_in_polygon(self, points, polygon):
            """
            Determine if a set of points are inside a set of polygons.

            Args:
            - points: Tensor of shape (N, 2) representing the N points
            - polygon: Tensor of shape (M, N_i, 2) representing the M polygons, where N_i is the number of polygon in the i-th polygon

            Returns:
            - in_polygon: Tensor of shape (N,) indicating if each point is inside the polygon.
            """
            
            # Repeat the first vertex of each polygon to the end so we can easily calculate the winding number.
            
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
            
            in_which_box=torch.argmax(in_poly.float(),dim=1)
            # meaningful only when this point is selected 
            # print(in_which_box)     
            # Sum the winding number along the edge dimension and check if the winding number is non-zero, which indicates the point is inside the polygon.
            in_polygon = in_poly.sum(1) > 0
            
            
            return in_polygon,in_which_box

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
            box=box[:,:4,:2].clone().detach()
            mask,index =self.point_in_polygon(inputlidar[:,0:2],box)
            #print(mask)
            
            inputlidar=inputlidar[mask]
            index = index[mask]


            points_num=inputlidar.shape[0]

            output=inputlidar

            
            return output,points_num,index

       

        #     return inputlidar,points_num
        def __getitem__(self, idx):
            #print('here sampler dataset')

            base_data_dict = self.retrieve_base_data(idx)
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
            box_num_stack = []
            object_stack = []
            object_id_stack = []
            bbox_stack = []
            sample_dict=self.detect_result[idx]
            gt_tensor=sample_dict['gt_box_tensor']
            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():

                distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                            ego_lidar_pose[0]) ** 2 + (
                                    selected_cav_base['params'][
                                        'lidar_pose'][1] - ego_lidar_pose[
                                        1]) ** 2)
                if distance > self.params['comm_range']:
                    continue

                selected_cav_processed = self.get_item_single_car(
                    selected_cav_base,
                    ego_lidar_pose)
                
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
            
            if gt_tensor.shape[0]> 0:
                

                projected_lidar_stack = np.vstack(projected_lidar_stack)
                filtered_lidar,points_num,index=self.box_filtering(projected_lidar_stack,gt_tensor)

                unique_index=torch.unique(index)

                num_groups=unique_index.shape[0]
                bbox_stack=gt_tensor

                
                
                result = [torch.empty((0,filtered_lidar.shape[1])) for _ in range(num_groups)]
                
                for i in range(num_groups):
                    box_mask = (index == unique_index[i]).unsqueeze(1).expand_as(filtered_lidar)
                    result[i] = torch.cat([result[i], filtered_lidar[box_mask].view(-1,filtered_lidar.shape[1])])
                    
                
                if bbox_stack.shape[1]!=8:
                    bbox_stack=bbox_stack.unsqueeze(0)

                surface_points=corner_pos2box(bbox_stack)
                processed_data_dict['ego'].update(
                    {                
                    'processed_lidar': result,
                    'bbox': gt_tensor,
                    'surface_points':surface_points
                    })

                if self.visualize:
                    processed_data_dict['ego'].update({'origin_lidar':
                                                        projected_lidar_stack})

                processed_data_dict['ego'].update({'points_num':points_num})

                return processed_data_dict
            else:
                return None

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
            # print("in test")
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            batch = batch[0] # only ego
            if batch is None:
                return None
            output_dict = {}

            for cav_id, cav_content in batch.items():
                output_dict.update({cav_id: {}})
                if self.visualize:
                    origin_lidar = [cav_content['origin_lidar']]

                output_dict[cav_id].update({'processed_lidar': cav_content['processed_lidar']})
                output_dict[cav_id].update({'bbox': cav_content['bbox']})
                output_dict[cav_id].update({'surface_points': cav_content['surface_points']})
                if self.visualize:
                    origin_lidar = \
                        np.array(
                            downsample_lidar_minimum(pcd_np_list=origin_lidar))
                    origin_lidar = torch.from_numpy(origin_lidar)
                    output_dict[cav_id].update({'origin_lidar': origin_lidar})
            
            if self.is_train :
                output_dict=angle_augment(output_dict) 

            return output_dict
        
        



            
        def collate_batch_train(self, batch):
            # Intermediate fusion is different the other two
            print("in train")
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
                
                if self.load_lidar_file:
                    processed_lidar_list.append(ego_dict['processed_lidar'])
 
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
                pred_box_tensor, pred_score,uncx,uncy = \
                    self.post_processor.post_process(data_dict, output_dict)
                return pred_box_tensor, pred_score, gt_box_tensor,uncx,uncy
            else:
                pred_box_tensor, pred_score = \
                    self.post_processor.post_process(data_dict, output_dict)
                return pred_box_tensor, pred_score, gt_box_tensor
            
            

            

    return SamplerFusionDataset

