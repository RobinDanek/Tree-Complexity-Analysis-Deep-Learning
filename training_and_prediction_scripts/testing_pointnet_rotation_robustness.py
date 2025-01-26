import torch
import numpy
import os
import sys

current_dir = os.getcwd()
sys.path.append(current_dir)

from modules.utils import rotation_robustness_eval
from modules.PointNet import PointNet
from modules.utils import get_device


model_name_1 = 'pointnet_10k_lr3_decay05'
model_name_2 = 'pointnet_10k_lr3_cosWR15_mult1_min6_newsplit_augmented'

print("\nBegin the dataloading...")
# Define the needed paths
data_path = os.path.join( current_dir, 'data', 'random_padding10k', 'testset' )
data_list = [os.path.join( data_path, f ) for f in os.listdir(data_path) if f.endswith('.npy')]

checkpoint_path = os.path.join( current_dir, 'model_saves', 'pointnet', f'{model_name_1}.pt')

print(f"Finished dataloading\n")

print("Loading the first model...")

print(checkpoint_path)

device = get_device()

# Load the model
model = PointNet(T2=64)
state_dict = torch.load( checkpoint_path )
model.load_state_dict( state_dict )
model.to(device)
print("Finished loading the first model\n")

predictions1 = rotation_robustness_eval( model, data_list, device )

print("Loading the second model...")

checkpoint_path = os.path.join( current_dir, 'model_saves', 'pointnet', f'{model_name_2}.pt')

print(checkpoint_path)

device = get_device()

# Load the model
model = PointNet(T2=64)
state_dict = torch.load( checkpoint_path )
model.load_state_dict( state_dict )
model.to(device)
print("Finished loading the second model\n")

predictions2 = rotation_robustness_eval( model, data_list, device )

print(f"First Model: {predictions1}")
print(f"Second Model: {predictions2}")