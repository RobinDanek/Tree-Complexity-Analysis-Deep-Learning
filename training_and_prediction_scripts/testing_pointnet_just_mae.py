import torch
import numpy
import os
import sys

current_dir = os.getcwd()
sys.path.append(current_dir)

from modules.utils import mae_eval
from modules.PointNet import PointNet
from modules.utils import get_device, TEST_SIZE, VAL_SIZE



model_name = 'pointnet_10k_lr3_cosWR15_mult1_min6_newsplit_augmented'


print("\nBegin the dataloading...")
# Define the needed paths
data_path = os.path.join( current_dir, 'data', 'random_padding10k', 'testset' )
checkpoint_path = os.path.join( current_dir, 'model_saves', 'pointnet', f'{model_name}.pt')

print(checkpoint_path)

# Get the testset
data_list = [os.path.join( data_path, f ) for f in os.listdir(data_path) if f.endswith('.npy')]
print(f"Finished dataloading. Predictions are being made for {len(data_list)} trees\n")

print("Loading the model...")
device = get_device()

# Load the model
model = PointNet(T2=64)
state_dict = torch.load( checkpoint_path )
model.load_state_dict( state_dict )
model.to(device)
print("Finished loading the model\n")

mae = mae_eval( data_list, model, device )