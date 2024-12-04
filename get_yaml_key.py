import yaml
from torch import tensor

print("ok")
yaml.add_constructor('tag:yaml.org,2002:python/object/apply:torch._utils._rebuild_tensor_v2', 
                     lambda loader, node: tensor(loader.construct_sequence(node))) 

data = yaml.safe_load(open('/GPFS/data/yunqiaoyang/Hybrid_opv2v_pose_error/val_boxes_down_key.yaml')) 

def get_keys(d, level=0):
    r = {}
    if level < 2:
        for k, v in d.items():
            print(k)
            r[k] = {}
            if isinstance(v, dict):
                r[k].update(get_keys(v, level+1))
    return r

result = get_keys(data)

with open('/GPFS/data/yunqiaoyang/Hybrid_opv2v_pose_error/val_boxes_down_key1.yaml', 'w') as f:
    yaml.dump(result, f)