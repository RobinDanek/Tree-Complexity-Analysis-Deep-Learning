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

def baseline_mae(train_list, test_list, device):
    """
    Computes the mean absolute error (MAE) of predicting the mean label
    from the train set for both train and test samples.

    Args:
        train_list: List of paths to training tree data
        test_list: List of paths to test tree data
        device: The device on which computation is done
    
    Returns:
        Tuple containing:
        - MAE of using the train mean label as a predictor for the train set.
        - MAE of using the train mean label as a predictor for the test set.
    """
    print(f"\nComputing baseline MAE using train set ({len(train_list)} samples) and evaluating on test set ({len(test_list)} samples)...")
    
    train_dataloader = DataLoader(CloudDataset(train_list), batch_size=32, shuffle=False)
    test_dataloader = DataLoader(CloudDataset(test_list), batch_size=32, shuffle=False)
    
    train_labels = []
    test_labels = []
    
    # Collect all train labels
    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc="Loading train labels", total=len(train_dataloader)):
            _, batch_labels = batch
            train_labels.extend(batch_labels.cpu().tolist())
    
    # Compute mean of all train labels
    mean_train_label = sum(train_labels) / len(train_labels)
    
    # Compute MAE for train set
    train_mae = sum(abs(label - mean_train_label) for label in train_labels) / len(train_labels)
    
    # Collect all test labels
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Loading test labels", total=len(test_dataloader)):
            _, batch_labels = batch
            test_labels.extend(batch_labels.cpu().tolist())
    
    # Compute MAE for test set using train mean label
    test_mae = sum(abs(label - mean_train_label) for label in test_labels) / len(test_labels)
    
    print(f"Train Mean Label: {mean_train_label:.4f}, Train Baseline MAE: {train_mae:.4f}, Test Baseline MAE: {test_mae:.4f}")
    return train_mae, test_mae

def mae_eval(data_list, model, device):
    """
    Evaluates a model on the given dataset and returns the Mean Absolute Error (MAE).

    Args:
        data_list: List of paths to tree data
        model: A model to perform the evaluation
        device: The device on which computation is done
    
    Returns:
        Mean Absolute Error (MAE) of the model on the given dataset.
    """
    print(f"\nEvaluating model on dataset ({len(data_list)} samples)...")
    
    dataloader = DataLoader(CloudDataset(data_list), batch_size=32, shuffle=False)
    mean_difference = 0.0
    differences = []
    
    # Set model to evaluation mode
    model.eval()
    
    # Perform predictions and compute MAE
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), desc="Making predictions", total=len(dataloader)):
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
                differences.append(difference)


    mean_difference = mean_difference / len(differences)
    print(f"Model MAE: {mean_difference:.3f}")
    return mean_difference

def rotation_robustness_eval(model, tree_paths, device):
    """
    Evaluates the robustness of the model to 90° rotations for multiple trees. For each tree,
    it computes the absolute prediction error between the original cloud and its rotated versions.

    Args:
        model: The initialized model to evaluate
        tree_paths: A list of paths to the tree .txt files
        device: The device on which computation is done
    
    Returns:
        A float representing the mean absolute deviation across all trees for the three rotations
    """
    total_error = 0.0
    num_trees = len(tree_paths)

    all_rotation_errors = []

    # Iterate over the list of tree paths
    for tree_path in tree_paths:
        # Get the predictions for this tree
        predictions = rotation_robustness_single_tree(model, tree_path, device)
        
        # Calculate absolute errors for the rotations (excluding the original)
        original_pred = predictions[0]
        rotation_errors = [abs(original_pred - pred) for pred in predictions[1:]]
        
        all_rotation_errors.append( rotation_errors )
        
    mean_deviations = np.mean( all_rotation_errors, axis=0 )

    return mean_deviations

def rotation_robustness_single_tree(model, tree_path, device):
    """
    Evaluates the robustness of a model to 90° rotations by predicting labels for a tree
    in its original orientation and three rotated versions.

    Args:
        model: The initialized model to evaluate
        tree_path: Path to the tree .txt file
        device: The device on which computation is done
    
    Returns:
        A list of predictions for the four orientations (original + 3 rotations)
    """
    

    # Load tree from file
    tree = np.load(tree_path)  # Assuming the tree is stored as a NumPy-compatible text file
    tree_points = tree[0][0]  # Adjust based on how the data is structured
    tree_tensor = torch.tensor(tree_points, dtype=torch.float32).to(device).transpose(0, 1).unsqueeze(0)
    
    # Prepare model
    model.eval()
    
    # Function to rotate a point cloud by 90 degrees along the Z-axis
    def rotate_cloud(cloud, angle):
        rotation_matrix = np.array([
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
            [0, 0, 1]
        ])
        rotated_cloud = np.dot(cloud, rotation_matrix.T)  # Apply the rotation
        return rotated_cloud
    
    predictions = []

    # Evaluate for the original orientation
    with torch.no_grad():
        predictions.append(model(tree_tensor).cpu().item())
    
    # Rotate the cloud by 90, 180, and 270 degrees and make predictions
    for angle in [90, 180, 270]:
        rotated_tree_points = rotate_cloud(tree_points, angle)
        rotated_tree_tensor = torch.tensor(rotated_tree_points, dtype=torch.float32).to(device).transpose(0, 1).unsqueeze(0)
        
        with torch.no_grad():
            predictions.append(model(rotated_tree_tensor).cpu().item())

    return predictions


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
    ax.set_ylabel( "Box-dimension", fontsize=18 )
    ax.set_xlabel("Tree ID", fontsize=18)
    ax.set_title("Comparison of predicted and actual box-dimensions", fontsize=20)
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

    plot_dict = {3: "Winnefeld", 4: "Nienover", 6: "Unterlüss", 8: "Göhrde 1"}

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
        ax.set_xticklabels(tree_names, rotation=45, ha='right', fontsize=10)  # Increase xtick label font size
        ax.set_ylabel("Box-dimension", fontsize=18)  # Increase ylabel font size
        ax.set_title(f"Plot {plot_dict[plot_number]}", fontsize=18)  # Increase subplot title font size
        ax.set_ylim(1.4, 2.2)
        ax.legend(fontsize=12)  # Increase legend font size

    # Add a main title for the entire figure
    fig.suptitle("Comparison of predicted and actual box-dimensions", fontsize=20)  # Increase main title font size


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
        ax.set_ylabel("Box-dimension")
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