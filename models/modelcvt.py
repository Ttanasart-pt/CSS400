import tensorflow as tf
import torch
import pickle
from colorHandPose.posenet import PoseNet as Model
from util.checkpoint import save_checkpoint

model = Model()

file_name = 'colorHandPose/ColorHandPose3D_data_v3/weights/posenet3d-rhd-stb-slr-finetuned.pickle'
session = tf.compat.v1.Session()
exclude_var_list = list()
# exclude_var_list = ['HandSegNet/conv5_2/weights', 
#                     'HandSegNet/conv5_2/biases', 
#                     'HandSegNet/conv6_1/weights', 
#                     'HandSegNet/conv6_1/biases']

# read from pickle file
with open(file_name, 'rb') as fi:
    weight_dict = pickle.load(fi)
    weight_dict = {k: v for k, v in weight_dict.items() if not any([x in k for x in exclude_var_list])}
    
keys = [k for k, v in weight_dict.items() if 'PoseNet2D' in k]
keys.sort()

# [print(k, weight_dict[k].shape) for k in keys]
    
for name, module in model.named_children():
    key = 'PoseNet2D/{0}/'.format(name)
    if key + 'biases' in weight_dict:
        b = torch.tensor(weight_dict[key + 'biases'])
        w = torch.tensor(weight_dict[key + 'weights'])
        w = w.permute((3, 2, 0, 1))
        w = torch.nn.Parameter(w)
        b = torch.nn.Parameter(b)
        module.weight.data = w
        module.bias.data = b
        
save_checkpoint('colorHandPose/checkpoints/posenet.pth.tar', model)