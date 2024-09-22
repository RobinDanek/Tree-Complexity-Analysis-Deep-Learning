import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append(os.getcwd())
from modules.LoadingTransforming import CloudDataset
from modules.ArrayTypes import predictionType

########## STANDARDS ###############
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.1

########## DEVICE LOADER ###########

def get_device(cuda_preference=True):
        """Gets pytorch device object. If cuda_preference=True and 
            cuda is available on your system, returns a cuda device.
        
        Args:
            cuda_preference: bool, default True
                Set to true if you would like to get a cuda device
                
        Returns: pytorch device object
                Pytorch device
        """
        
        print('cuda available:', torch.cuda.is_available(), 
            '; cudnn available:', torch.backends.cudnn.is_available(),
            '; num devices:', torch.cuda.device_count())
        
        use_cuda = False if not cuda_preference else torch.cuda.is_available()
        device = torch.device('cuda:0' if use_cuda else 'cpu')
        device_name = torch.cuda.get_device_name(device) if use_cuda else 'cpu'
        print(f'Using device {device_name} \n')
        return device

########### MODEL EVALUATION ###########

def general_eval(test_list, model, num_plot, device, plot_savepath=None, prediction_path=None):
    """
    This  function reads  in a dataloader of test trees and usel model to predict 
    their label. It then computes the mean difference between the comparison
    and randomly picks num_plot trees to visualize the difference in predictions

    Args:
        test_loader: A dataloader of paths to test trees
        model: A model to do the evaluation on
        num_plot: The number of trees included in the plot.
        device: The device on which the computation is done
    """
    if num_plot > len(test_list):
        raise ValueError("Must have at least num_plot samples in the test list!")
 
    #  Initialize variables and lists for plotting and statistics
    mean_difference = 0.0
    differences = []

    testloader = DataLoader(CloudDataset(test_list), batch_size=32, shuffle=False)

    # Set model to evaluation mode
    model.eval()
    
    print("\nStarting to make predictions...")
    # Iterate over the test_loader
    with torch.no_grad():
        for i, batch in tqdm(enumerate(testloader), desc="Making predictions", total=len(testloader)):
            # Load tree and label
            trees, labels = batch
            trees, labels = trees.to(device), labels.to(device)

            # Iterate over samples in batch
            for j in range( trees.size(0) ):
                # Convert to torch tensors for inferece, pass to device and add batch dimension
                tree = trees[j].unsqueeze(0)
                label = labels[j].unsqueeze(0)

                label_pred = model(tree)
                # calculate absolute difference
                difference = torch.abs( label_pred - label ).item()
                # Append to variable
                mean_difference += difference

                # Store values for plotting
                # New way: appending as a tuple, which matches the dtype structure
                differences.append((os.path.splitext(os.path.basename(test_list[i*32 + j]))[0], label.cpu().item(), label_pred.cpu().item()))


    mean_difference = mean_difference / len(differences)
    print("Finished making predictions!\n")
    print( f"The mean difference is {mean_difference:.4f}\n" )

    print( f"Getting {num_plot} random samples...")
    # Now do the plotting
    idxs = np.arange( len(differences) )
    np.random.seed(SEED)
    random_idxs = np.random.choice( idxs, num_plot, replace=False )
    random_samples = []
    for idx in random_idxs:
        random_samples.append( differences[idx] )

    # Extract the parts for better readability
    tree_names = []
    actual_labels = []
    predicted_labels = []

    for samp in random_samples:
        tree_name, actual_label, predicted_label = samp
        tree_names.append( tree_name )
        actual_labels.append( actual_label )
        predicted_labels.append( predicted_label )
    print("Finished retrieving the samples!\n")

    fig, ax = plt.subplots(figsize=(10,5))
    # Loop over the collected samples
    for i, (actual_label, predicted_label) in enumerate(zip(actual_labels, predicted_labels)):
        ax.plot([i, i], [actual_label, predicted_label], linestyle='-', color='gray')  # Line connecting actual and predicted
        ax.scatter(i, actual_label, color='green', label='Actual' if i == 0 else "")  # Green dot for actual value
        ax.scatter(i, predicted_label, color='purple', label='Predicted' if i == 0 else "")  # Purple dot for predicted value

    # Set the xticks to the tree labels
    ax.set_xticks( range(num_plot) )
    ax.set_xticklabels( tree_names, rotation=45,  ha='right' )
    # Set the rest
    ax.set_ylabel( "Fractal dimension" )
    ax.set_title("Predicted and actual fractal dimensions")
    ax.legend()
    plt.tight_layout()
    if plot_savepath:
        plt.savefig(plot_savepath, dpi=600)
    plt.show()

    # Now safe the testset if output path is provided
    if prediction_path:
        differences = np.array(differences, dtype=predictionType)
        np.save(prediction_path, differences)

    return mean_difference