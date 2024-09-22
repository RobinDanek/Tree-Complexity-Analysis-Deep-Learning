import torch
import numpy
import os
import sys

current_dir = os.getcwd()
sys.path.append(current_dir)

from modules.utils import general_eval
from modules.PointNet import PointNet
from modules.LoadingTransforming import CloudSplitter, CloudLoader
from modules.utils import get_device, TEST_SIZE, VAL_SIZE




print("\nBegin the dataloading...")
# Define the needed paths
data_path = os.path.join( current_dir, 'data', 'random_padding10k', 'testset' )
checkpoint_path = os.path.join( current_dir, 'model_saves', 'pointnet', 'pointnet_10k_lr3_cosWR15_mult1_min6_newsplit.pt')

print(checkpoint_path)

# Get the testset
test_list = [os.path.join( data_path, f ) for f in os.listdir(data_path) if f.endswith('.npy')]
print(f"Finished dataloading. Predictions are being made for {len(test_list)} trees\n")

print("Loading the model...")
device = get_device()

# Load the model
model = PointNet()
state_dict = torch.load( checkpoint_path )
model.load_state_dict( state_dict )
model.to(device)
print("Finished loading the model\n")

test_loss = general_eval(test_list=test_list,  model=model, num_plot=50, device=device, prediction_path=os.path.join(current_dir, 'data', 'predictions10k'))
