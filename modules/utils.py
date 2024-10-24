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
    print("\n Starting general evaluation")

    if num_plot > len(test_list):
        raise ValueError("Must have at least num_plot samples in the test list!")
 
    #  Initialize variables and lists for plotting and statistics
    mean_difference = 0.0
    differences = []

    print("\nLoading the data...")
    testloader = DataLoader(CloudDataset(test_list), batch_size=32, shuffle=False)
    print("Loaded the data!")

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
    ax.set_ylim(1.4,2.2)
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



def plott_eval(test_list, model, num_plot, device, plot_savepath=None, prediction_path=None):
    """
    This function reads in a dataloader of test trees and uses the model to predict their labels.
    It then computes the mean difference between the comparison and randomly picks num_plot trees 
    from each specified plot number to visualize the difference in predictions.

    Args:
        test_list: A list of paths to test tree files.
        model: The model to do the evaluation on.
        num_plot: The number of trees to include in each plot.
        device: The device on which the computation is done.
        plot_numbers: A list of plot numbers (integers) to filter and visualize.
        plot_savepath: Path to save the resulting plot(s).
        prediction_path: Path to save the predictions as a numpy file.
    """
    # Set model to evaluation mode
    model.eval()

    print("\nStarting evaluation of plotts!")
    
    # Dictionary to hold the differences for each plot
    plot_numbers = [3,4,6,8]
    differences_per_plot = {plot_number: [] for plot_number in plot_numbers}
    
    # Load test data
    print("\nLoading the data...")
    testloader = DataLoader(CloudDataset(test_list), batch_size=32, shuffle=False)
    print("Loaded the data!")

    print("\nStarting to make predictions...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(testloader), desc="Making predictions", total=len(testloader)):
            # Load trees and labels
            trees, labels = batch
            trees, labels = trees.to(device), labels.to(device)

            # Iterate over samples in the batch
            for j in range(trees.size(0)):
                tree = trees[j].unsqueeze(0)
                label = labels[j].unsqueeze(0)

                # Make prediction
                label_pred = model(tree)

                # Calculate absolute difference
                difference = torch.abs(label_pred - label).item()
                
                # Extract the plot number from the filename
                filename = os.path.basename(test_list[i * 32 + j])
                plot_number_str = filename.split('_')[0].split('.')[0]  # Extract 'plotnumber' part before the dot
                plot_number = int(plot_number_str)  # Convert to integer
                
                # Store the results if the plot number is in the specified plot_numbers
                if plot_number in plot_numbers:
                    differences_per_plot[plot_number].append(
                        (filename, label.cpu().item(), label_pred.cpu().item())
                    )

    print("Finished making predictions!\n")

    # Create a single figure with a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    axes = axes.flatten()  # Flatten the array of axes for easy iteration

    # Now create plots for each specified plot number
    for idx, plot_number in enumerate(plot_numbers):
        ax = axes[idx]  # Select the subplot axis

        # Get differences for the current plot number
        differences = differences_per_plot[plot_number]

        # If there are fewer samples than num_plot, raise an error
        if len(differences) < num_plot:
            raise ValueError(f"Not enough samples for plot number {plot_number}. Required: {num_plot}, Found: {len(differences)}")

        # Randomly select samples for plotting
        np.random.seed(SEED)
        random_idxs = np.random.choice(len(differences), num_plot, replace=False)
        random_samples = [differences[idx] for idx in random_idxs]

        # Extract the parts for better readability
        tree_names, actual_labels, predicted_labels = [], [], []
        for samp in random_samples:
            tree_name, actual_label, predicted_label = samp
            tree_names.append(tree_name)
            actual_labels.append(actual_label)
            predicted_labels.append(predicted_label)

        # Create the plot in the subplot
        for i, (actual_label, predicted_label) in enumerate(zip(actual_labels, predicted_labels)):
            ax.plot([i, i], [actual_label, predicted_label], linestyle='-', color='gray')  # Line connecting actual and predicted
            ax.scatter(i, actual_label, color='green', label='Actual' if i == 0 else "")  # Green dot for actual value
            ax.scatter(i, predicted_label, color='purple', label='Predicted' if i == 0 else "")  # Purple dot for predicted value

        # Set the xticks to the tree labels
        ax.set_xticks(range(num_plot))
        ax.set_xticklabels(tree_names, rotation=45, ha='right')
        ax.set_ylabel("Fractal dimension")
        ax.set_title(f"Plot {plot_number}")
        ax.set_ylim(1.4,2.2)
        ax.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure if a save path is provided
    if plot_savepath:
        plt.savefig(plot_savepath, dpi=600)

    plt.show()

    # Optionally save all differences as a numpy file
    if prediction_path:
        # Combine all differences
        all_differences = []
        for plot_number, differences in differences_per_plot.items():
            all_differences.extend(differences)
        all_differences = np.array(all_differences, dtype=predictionType)
        np.save(prediction_path, all_differences)

    return differences_per_plot



def tree_type_eval(test_list, model, num_plot, device, plot_savepath=None, prediction_path=None):
    """
    This function reads in a dataloader of test trees and uses the model to predict their labels.
    It then computes the mean difference between the comparison and randomly picks num_plot trees 
    from each specified tree type to visualize the difference in predictions.

    Args:
        test_list: A list of paths to test tree files.
        model: The model to do the evaluation on.
        num_plot: The number of trees to include in each plot.
        device: The device on which the computation is done.
        plot_savepath: Path to save the resulting plot(s).
        prediction_path: Path to save the predictions as a numpy file.
    """
    # Set model to evaluation mode
    model.eval()
    
    print("\nStarting evaluation by tree type!")
    
    # Define the tree types and initialize dictionary
    tree_types = ['1', '2', '3', '4', '5']
    differences_per_type = {tree_type: [] for tree_type in tree_types}

    # Load test data
    print("\nLoading the data...")
    testloader = DataLoader(CloudDataset(test_list), batch_size=32, shuffle=False)
    print("Loaded the data!")

    print("\nStarting to make predictions...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(testloader), desc="Making predictions", total=len(testloader)):
            # Load trees and labels
            trees, labels = batch
            trees, labels = trees.to(device), labels.to(device)

            # Iterate over samples in the batch
            for j in range(trees.size(0)):
                tree = trees[j].unsqueeze(0)
                label = labels[j].unsqueeze(0)

                # Make prediction
                label_pred = model(tree)

                # Calculate absolute difference
                difference = torch.abs(label_pred - label).item()
                
                # Extract the tree type from the filename
                filename = os.path.basename(test_list[i * 32 + j])
                plot_and_treetype = filename.split('_')[0]  # Extract 'plotnumber.treetype'
                tree_type_str = plot_and_treetype.split('.')[1]  # Extract 'treetype' part
                
                # Determine the base tree type (ignoring the 'b' suffix if present)
                base_tree_type = tree_type_str.rstrip('b')  # Remove 'b' if it exists

                # Store the results if the base tree type is in the specified tree_types
                if base_tree_type in tree_types:
                    differences_per_type[base_tree_type].append(
                        (filename, label.cpu().item(), label_pred.cpu().item())
                    )

    print("Finished making predictions!\n")

    # Create a single figure with a 1x5 grid of subplots
    fig, axes = plt.subplots(5, 1, figsize=(10, 20), sharey=True)
    fig.suptitle("Evaluation by Tree Type", fontsize=16)
    
    # Now create plots for each specified tree type
    for idx, tree_type in enumerate(tree_types):
        ax = axes[idx]  # Select the subplot axis

        # Get differences for the current tree type
        differences = differences_per_type[tree_type]

        # If there are fewer samples than num_plot, raise an error
        if len(differences) < num_plot:
            raise ValueError(f"Not enough samples for tree type {tree_type}. Required: {num_plot}, Found: {len(differences)}")

        # Randomly select samples for plotting
        np.random.seed(SEED)
        random_idxs = np.random.choice(len(differences), num_plot, replace=False)
        random_samples = [differences[idx] for idx in random_idxs]

        # Extract the parts for better readability
        tree_names, actual_labels, predicted_labels = [], [], []
        for samp in random_samples:
            tree_name, actual_label, predicted_label = samp
            tree_names.append(tree_name)
            actual_labels.append(actual_label)
            predicted_labels.append(predicted_label)

        # Create the plot in the subplot
        for i, (actual_label, predicted_label) in enumerate(zip(actual_labels, predicted_labels)):
            ax.plot([i, i], [actual_label, predicted_label], linestyle='-', color='gray')  # Line connecting actual and predicted
            ax.scatter(i, actual_label, color='green', label='Actual' if i == 0 else "")  # Green dot for actual value
            ax.scatter(i, predicted_label, color='purple', label='Predicted' if i == 0 else "")  # Purple dot for predicted value

        # Set the xticks to the tree labels
        ax.set_xticks(range(num_plot))
        ax.set_xticklabels(tree_names, rotation=45, ha='right')
        ax.set_ylabel("Fractal dimension")
        ax.set_title(f"Tree Type {tree_type}")
        ax.set_ylim(1.4, 2.2)
        ax.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate the suptitle

    # Save the figure if a save path is provided
    if plot_savepath:
        plt.savefig(plot_savepath, dpi=600)

    plt.show()

    # Optionally save all differences as a numpy file
    if prediction_path:
        # Combine all differences
        all_differences = []
        for tree_type, differences in differences_per_type.items():
            all_differences.extend(differences)
        all_differences = np.array(all_differences, dtype=predictionType)
        np.save(prediction_path, all_differences)

    return differences_per_type