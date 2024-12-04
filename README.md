# Hycomm
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This program is based on the [OpenCOOD.](https://mobility-lab.seas.ucla.edu/opv2v/)

## Main result
<p align="center">
<img src="images/image.png" width="600" alt="" class="img-responsive">
</p>
HyComm consistently achieves superior performance-bandwidth trade-off on both the simulated and real-world datasets, outperforming Where2comm with 2,006/1,317 ¡Á less communication cost on DAIR-V2X/OPV2V for AP50/70


## Installation
Please refer to [data introduction](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html)and [installation](https://opencood.readthedocs.io/en/latest/md_files/installation.html) guide to preparedata and install OpenCOOD. 
### Datasets
To see more details of OPV2V data, please check [the website.](https://mobility-lab.seas.ucla.edu/opv2v/)
To see more details of DAIR-V2X data, please check [the website.](checkhttps://thudair.baai.ac.cn/index)

## Quick Start
### Data sequence visualization
To quickly visualize the LiDAR stream in the OPV2V dataset, first modify the `validate_dir`
in your `opencood/hypes_yaml/visualization.yaml` to the opv2v data path on your local machine, e.g. `opv2v/validate`,
and the run the following commond:
```python
cd ~/OpenCOOD
python opencood/visualization/vis_data_sequence.py [--color_mode ${COLOR_RENDERING_MODE}]
```
Arguments Explanation:
- `color_mode` : str type, indicating the lidar color rendering mode. You can choose from 'constant', 'intensity' or 'z-value'.


### Train your model
OpenCOOD uses yaml file to configure all the parameters for training. To train your own model
from scratch or a continued checkpoint, run the following commonds:
```python
python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
```
Arguments Explanation:
- `hypes_yaml` or `-y`: the path of the training configuration file, e.g. `opencood/hypes_yaml/second_early_fusion.yaml`, meaning you want to train
an early fusion model which utilizes SECOND as the backbone. See [Tutorial 1: Config System](https://opencood.readthedocs.io/en/latest/md_files/config_tutorial.html) to learn more about the rules of the yaml files.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.

If you train DiscoNet with Knowledge Distillation, use opencood/tools/train_kd.py

### Test the model
Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder
refers to the testing dataset path, e.g. `opv2v_data_dumping/test`.

```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} [--sample_method] [--store_boxes] [--is_hybrid] [--pose_err]
```
Arguments Explanation:
- `model_dir`: the path to your saved model.
- `fusion_method`: indicate the fusion strategy, currently support 'early', 'late', and 'intermediate'.
- `sample_method`: the method to downsample the point cloud
- `store_boxes` : store detection boxes and gt boxes.
- `is_hybrid` : use hybrid method if implemented. (Tips: you have to first save boxes once for hybrid method)
- `pose_err` : add pose error when inference


## Tutorials
OpenCOOD provides a series of tutorials to help you understand OpenCOOD more. Please check the series of [tutorials](https://opencood.readthedocs.io/en/latest/md_files/config_tutorial.html).

