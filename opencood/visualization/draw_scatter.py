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

for value in x_data.values():
    if isinstance(value, list):
        mergedx_list.extend(value)
for value in y_data.values():
    if isinstance(value, list):
        mergedy_list.extend(value)
        
indexes = [i for i, num in enumerate(mergedx_list) if num > 0.1]
# print(indexes)
# plt.hist(indexes, bins=10, edgecolor='black')
# plt.title('Elements Distribution Histogram')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.savefig(f"/GPFS/data/yunqiaoyang-1/Hybrid_opv2v_unc_diffxy/above_0.1_diff{x1}.png")
# input()
index=np.arange(len(mergedx_list))
size=35058
index_sampled=np.random.choice(index, size=size, replace=False)
mergedx_list=[mergedx_list[i] for i in index_sampled]
mergedy_list=[mergedy_list[i] for i in index_sampled]

note=x+str(size)+'s'

draw_envelop=True
envelop_index=95
y_values=np.array(mergedy_list)
x_values=np.array(mergedx_list)

y_log_values = np.log(y_values)
log_min = np.min(y_log_values)
log_max = np.max(y_log_values)

num_intervals = 20
interval_width_log = (log_max - log_min) / num_intervals

quantiles_y=[]
quantiles_low_x = []
quantiles_high_x = []

for i in range(num_intervals):
    start = log_min + i * interval_width_log
    end = start + interval_width_log
    
    interval_data = y_values[(y_log_values >= start) & (y_log_values < end)]
    
    if len(interval_data) > 0:
        q_low_x = np.percentile(x_values[(y_log_values >= start) & (y_log_values < end)], (100-envelop_index)/2)
        q_hi_x = np.percentile(x_values[(y_log_values >= start) & (y_log_values < end)], (100+envelop_index)/2)
        
        quantiles_low_x.append(q_low_x)
        quantiles_high_x.append(q_hi_x)
        quantiles_y.append((start+end)/2)
quantiles_y=np.exp(quantiles_y)

if draw_envelop:
    plt.figure(figsize=(10, 6))  
    plt.scatter(mergedx_list, mergedy_list, alpha=0.5,s=0.1) 

    plt.plot(quantiles_low_x, quantiles_y, linestyle='--', color='r', label=f'  {(100-envelop_index)/2} Percentile Envelope')
    plt.plot(quantiles_high_x, quantiles_y, linestyle='--', color='b', label=f' {(100+envelop_index)/2} Percentile Envelope')

    plt.ylim(0.00001,0.04)
    plt.yscale('log')
    plt.title(f'Scatter Unc_{x1}-Diff_{x1}')
    plt.xlabel(f'Diff_{x1}')
    plt.ylabel(f'Unc_{x1}(log)')
    plt.legend()
    plt.savefig(f"/GPFS/data/yunqiaoyang-1/Hybrid_opv2v_unc_diffxy/unc{note}_diff{note}(log)_{envelop_index}envelop.png")
else:
    plt.figure(figsize=(10, 6))  
    plt.scatter(mergedx_list, mergedy_list, alpha=0.5,s=0.1) 
    
    plt.ylim(0.00001,0.04)
    plt.yscale('log')
    plt.title(f'Scatter Unc_{x1}-Diff_{x1}')
    plt.xlabel(f'Diff_{x1}')
    plt.ylabel(f'Unc_{x1}(log)')
    plt.legend()
    plt.savefig(f"/GPFS/data/yunqiaoyang-1/Hybrid_opv2v_unc_diffxy/unc{note}_diff{note}(log).png")







# print(x_data)
# print(type(x_data))