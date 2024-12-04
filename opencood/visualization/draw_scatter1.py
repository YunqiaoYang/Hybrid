import numpy as np
import matplotlib.pyplot as plt
import yaml


# x_data = np.load('/GPFS/data/yunqiaoyang-1/Hybrid_opv2v_pose_error/diff_x.npy', allow_pickle=True)  
# y_data = np.load('/GPFS/data/yunqiaoyang-1/Hybrid_opv2v_pose_error/unc_x.npy', allow_pickle=True)  

# print(x_data)
# print(type(x_data))
x='x_all'
x1='x'
unc='y_ego_only'
note=x
with open(f"/GPFS/data/yunqiaoyang-1/Hybrid_opv2v_unc_diffxy/diff_{x}.yaml", "r", encoding="utf-8") as file:
    y_data = yaml.safe_load(file)
with open(f"/GPFS/data/yunqiaoyang-1/Hybrid_opv2v_unc_diffxy/unc_{x}.yaml", "r", encoding="utf-8") as file:
    x_data = yaml.safe_load(file)

mergedx_list = []
mergedy_list = []

for value in x_data.values():
    if isinstance(value, list):
        mergedx_list.extend(value)
for value in y_data.values():
    if isinstance(value, list):
        mergedy_list.extend(value)


plt.figure(figsize=(10, 6))  
plt.scatter(mergedx_list, mergedy_list, alpha=0.5) 
plt.xlim(0.00001,0.04)
plt.xscale('log')
plt.title(f'Diff_{x1}-Scatter Unc_{x1}')
plt.ylabel(f'Diff_{x1}')
plt.xlabel(f'Unc_{x1}(log)')
plt.savefig(f"/GPFS/data/yunqiaoyang-1/Hybrid_opv2v_unc_diffxy/diff{note}(log)_unc{note}.png")








# print(x_data)
# print(type(x_data))