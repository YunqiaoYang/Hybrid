import numpy as np
import matplotlib.pyplot as plt
import yaml


# x_data = np.load('/GPFS/data/yunqiaoyang-1/Hybrid_opv2v_pose_error/diff_x.npy', allow_pickle=True)  
# y_data = np.load('/GPFS/data/yunqiaoyang-1/Hybrid_opv2v_pose_error/unc_x.npy', allow_pickle=True)  

# print(x_data)
# print(type(x_data))
x='x_all'
x1='x'


with open(f"/GPFS/data/yunqiaoyang-1/Hybrid_opv2v_unc_diffxy/diff_{x}.yaml", "r", encoding="utf-8") as file:
    x_data = yaml.safe_load(file)
with open(f"/GPFS/data/yunqiaoyang-1/Hybrid_opv2v_unc_diffxy/unc_{x}.yaml", "r", encoding="utf-8") as file:
    y_data = yaml.safe_load(file)

mergedx_list = []
mergedy_list = []
over_dict={}

for value in x_data.values():
    if isinstance(value, list):
        mergedx_list.extend(value)
for value in y_data.values():
    if isinstance(value, list):
        mergedy_list.extend(value)
        
# indexes = [i for i, num in enumerate(mergedx_list) if num > 0.1]
# print(indexes)
# plt.hist(indexes, bins=10, edgecolor='black')
# plt.title('Elements Distribution Histogram')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.savefig(f"/GPFS/data/yunqiaoyang-1/Hybrid_opv2v_unc_diffxy/above_0.1_diff{x1}.png")
# input()
for key,numbers in x_data.items():
    over_dict[key]=sum(1 for num in numbers if num>0.2)

sub_dict = {key: value for key, value in over_dict.items() if value !=0}


# print(sub_dict)
# with open('/GPFS/data/yunqiaoyang-1/Hybrid_opv2v_unc_diffxy/diffx_above02.yaml', 'w') as f:
#        yaml.dump(over_dict, f, default_flow_style=False)

plt.figure(figsize=(10, 6)) 
plt.bar(over_dict.keys(), over_dict.values(), color='skyblue') 

plt.title('diffx_above0.2')
plt.xlabel('Sample')
plt.ylabel('N')
plt.savefig(f"/GPFS/data/yunqiaoyang-1/Hybrid_opv2v_unc_diffxy/diffx_above0.2.png")


# index=np.arange(len(mergedx_list))
# size=35058
# index_sampled=np.random.choice(index, size=size, replace=False)
# mergedx_list=[mergedx_list[i] for i in index_sampled]
# mergedy_list=[mergedy_list[i] for i in index_sampled]

# note=x+str(size)+'s'

# plt.figure(figsize=(10, 6))  
# plt.scatter(mergedx_list, mergedy_list, alpha=0.5,s=0.1) 
# plt.ylim(0.00001,0.04)
# plt.yscale('log')
# plt.title(f'Scatter Unc_{x1}-Diff_{x1}')
# plt.xlabel(f'Diff_{x1}')
# plt.ylabel(f'Unc_{x1}(log)')
# plt.savefig(f"/GPFS/data/yunqiaoyang-1/Hybrid_opv2v_unc_diffxy/unc{note}_diff{note}(log).png")








# print(x_data)
# print(type(x_data))