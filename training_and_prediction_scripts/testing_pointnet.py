import torch
import numpy
import os
import sys

current_dir = os.getcwd()
sys.path.append(current_dir)

from modules.utils import general_eval, plott_eval, tree_type_eval
from modules.PointNet import PointNet
from modules.LoadingTransforming import CloudSplitter, CloudLoader
from modules.utils import get_device, TEST_SIZE, VAL_SIZE



model_name = 'pointnet_10k_lr3_FMSE_cosWR15_mult1_min6_aug'


print("\nBegin the dataloading...")
# Define the needed paths
data_path = os.path.join( current_dir, 'data', 'random_padding10k', 'testset' )
checkpoint_path = os.path.join( current_dir, 'model_saves', 'pointnet', f'{model_name}.pt')

print(checkpoint_path)

# Get the testset
test_list = [os.path.join( data_path, f ) for f in os.listdir(data_path) if f.endswith('.npy')]
print(f"Finished dataloading. Predictions are being made for {len(test_list)} trees\n")

print("Loading the model...")
device = get_device()

# Load the model
model = PointNet(T2=64)
state_dict = torch.load( checkpoint_path )
model.load_state_dict( state_dict )
model.to(device)
print("Finished loading the model\n")

plot_savepath_general = os.path.join( current_dir, 'plots', 'PredictionAnalytics', f'general_{model_name}.png' )

test_loss = general_eval(test_list=test_list,  model=model, num_plot=50, device=device, plot_savepath=plot_savepath_general)

plot_savepath_plott = os.path.join( current_dir, 'plots', 'PredictionAnalytics', f'plott_{model_name}.png' )

differences_per_plot = plott_eval(test_list=test_list,  model=model, num_plot=40, device=device, plot_savepath=plot_savepath_plott)

plot_savepath_tree_type = os.path.join( current_dir, 'plots', 'PredictionAnalytics', f'treeType_{model_name}.png' )

differences_per_tree_type = tree_type_eval(test_list=test_list,  model=model, num_plot=30, device=device, plot_savepath=plot_savepath_tree_type)
