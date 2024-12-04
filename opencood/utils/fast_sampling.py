# import numpy as np

# def fast_sampling(input_pc,sample_num,neighbor_threshold=0.3):

    
#     origin_pc=input_pc
#     dist_mat=np.linalg.norm(input_pc[:, np.newaxis] - input_pc, axis=-1)
#     #get near points
#     near_points_pos=np.where(dist_mat<neighbor_threshold,np.ones_like(dist_mat),np.zeros_like(dist_mat))
#     #calc the center of near points
#     input_pc=input_pc.unsqueeze(0)
#     expanded_pc=input_pc.repeat(input_pc.shape[1],1,1)
#     selected_neighbor=np.mul(expanded_pc,near_points_pos.unsqueeze(-1))
#     neighbor_center=np.sum(selected_neighbor,dim=1)/np.sum(near_points_pos,dim=0).unsqueeze(-1)
#     weight=np.mul(origin_pc-neighbor_center,origin_pc-neighbor_center)
#     weight=np.sum(weight,dim=-1)
#     # weight=torch.exp(weight)
#     # select_index=torch.multinomial(weight, 1000, replacement=False)
#     select_index=np.argsort(weight,descending=True)[:sample_num]
    
    
#     return select_index



import numpy as np

def fast_sampling(input_pc, sample_num, neighbor_threshold=0.3):
    origin_pc = input_pc.copy()
    dist_mat = np.linalg.norm(input_pc[:, np.newaxis] - input_pc, axis=-1)

    # 获取附近的点
    near_points_pos = np.where(dist_mat < neighbor_threshold, np.ones_like(dist_mat), np.zeros_like(dist_mat))

    # 计算附近点的中心
    expanded_pc = np.expand_dims(input_pc, axis=0)
    selected_neighbor = np.multiply(expanded_pc, near_points_pos[:, :, np.newaxis])
    neighbor_center = np.sum(selected_neighbor, axis=1) / np.sum(near_points_pos, axis=1)[:, np.newaxis]

    weight = np.sum(np.square(origin_pc - neighbor_center), axis=-1)
    select_index = np.argsort(weight)[::-1][:sample_num]

    return select_index