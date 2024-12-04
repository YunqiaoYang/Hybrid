import torch
import numpy as np
import random

def point_in_polygon_2d(points, polygon):
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

def unc_filtering(inputlidar,tensors,scores,sampling_rate=1.0):

    
    
    box=tensors[:,:4,:2]
    average=torch.mean(box,keepdim=True,dim=1)

    diff=box-average

    diff[:,:,0]=diff[:,:,0]*(1+10*scores[:1]).T
    diff[:,:,1]=diff[:,:,1]*(1+10*scores[1:]).T

    box=average+diff

    mask,in_which_box=point_in_polygon_2d(inputlidar[:,0:2],box)
    inputlidar=inputlidar[mask]
    box_num=in_which_box[mask]

    xy_scores=(scores[0]**2+scores[1]**2).T

    prob=xy_scores[box_num]


    #normalization
    prob=(prob/sum(prob)).cpu().numpy()

    #2.1-downsapling-random
    # sampling_idx = np.sort(np.random.choice(range(points_num), int(points_num*self.sampling_rate)))
    # inputlidar=inputlidar[sampling_idx]
    
    #2.2-downsampling-fps
    #inputlidar = self.fps(inputlidar, self.sampling_num)
    
    #2.3-downsampling-uncertainty
    #print(self.sampling_rate)
    points_num=inputlidar.shape[0]
    
    if prob.shape[0]>0:
        sampling_idx=np.sort(random.choices(range(points_num), k=int(points_num*sampling_rate),weights=prob))
        inputlidar=inputlidar[sampling_idx]
    output=inputlidar.cpu().numpy()
    points_num=output.shape[0]
    
    return output,points_num