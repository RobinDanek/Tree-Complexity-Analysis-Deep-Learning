import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch
import shutil








################# DATASET FUNCTIONS #######################

class CloudDataset(Dataset):
    # Loads set of pointclouds and passes one with according label
    def __init__(self, filepaths):
        self.filepaths = filepaths
        print(f"Initialized CloudDataset with {len(filepaths)} files.")

    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        #print(f"Fetching item index: {idx}")
        try:
            file = np.load(self.filepaths[idx])
            cloud = torch.tensor(file[0][0], dtype=torch.float32).transpose(0, 1)
            label = torch.tensor(file[0][1], dtype=torch.float32)
            #print(f"Cloud shape: {cloud.shape}, Label shape: {label.shape}, Cloud size: {cloud.element_size() * cloud.nelement() / (1024**2):.2f} MB")
            return cloud, label
        except Exception as e:
            print(f"Error loading file {self.filepaths[idx]}: {e}")
            raise
    
# Test Val Train Split. If an output directory is given, files are moved into subfolders:
# Three folders containing the sets for reuse and one folder containing all samples
def CloudSplitter(filepaths, test_size, val_size, output_dir=None, seed=42):
    # Helper function to delete all files in a directory
    def clear_directory(directory):
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Just in case there are subdirectories

    # Convert filepaths to a numpy array if it is not already
    filepaths = np.array(filepaths)
    
    if output_dir:
        # Create an 'all_samples' directory and move all files to this directory
        all_samples_dir = os.path.join(output_dir, 'all_samples')
        # Create train, validation, and test directories
        train_dir = os.path.join(output_dir, 'trainset')
        val_dir = os.path.join(output_dir, 'valset')
        test_dir = os.path.join(output_dir, 'testset')

        os.makedirs(all_samples_dir, exist_ok=True)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Clear all set-directories before execution
        clear_directory(train_dir)
        clear_directory(val_dir)
        clear_directory(test_dir)

        # Move all files to 'all_samples' directory
        for filepath in filepaths:
            dest_path = os.path.join(all_samples_dir, os.path.basename(filepath))
            if os.path.exists(dest_path):
                continue  # Skip this file if it already exists
            shutil.move(filepath, dest_path)

        # Update filepaths to point to the new 'all_samples' directory
        filepaths = np.array([os.path.join(all_samples_dir, os.path.basename(fp)) for fp in filepaths])

    # Split the indices randomly
    np.random.seed(seed)
    idxs = np.random.permutation(len(filepaths))
    np.random.shuffle(idxs)

    # First train-test split
    splt_test = int(np.floor(len(idxs) * test_size))
    idxs_test = idxs[:splt_test]
    idxs_train = idxs[splt_test:]

    # Now the validation split
    splt_val = int(np.floor(len(idxs_train) * val_size))
    idxs_val = idxs_train[:splt_val]
    idxs_train = idxs_train[splt_val:]

    # Perform the splits
    trainset = filepaths[idxs_train.tolist()]
    testset = filepaths[idxs_test.tolist()]
    valset = filepaths[idxs_val.tolist()]
    
    if output_dir:
        # Move files to their respective directories
        for filepath in trainset:
            shutil.copy(filepath, train_dir)

        for filepath in valset:
            shutil.copy(filepath, val_dir)

        for filepath in testset:
            shutil.copy(filepath, test_dir)

        print(f"Files have been moved and split into 'trainset', 'valset', 'testset', and 'all_samples' in {output_dir}")

    return trainset, valset, testset

# def CloudLoader(filepaths, batch_size, test_size,  val_size):
#     # First split the data
#     trainset, valset, testset = CloudSplitter(filepaths=filepaths, test_size=test_size, val_size=val_size)

#     # Create Datasets and DataLoaders
#     print("Initializing trainloader...")
#     trainloader = DataLoader(CloudDataset(trainset), batch_size=batch_size, shuffle=True, num_workers=0)
#     print("Initializing valloader...")
#     valloader = DataLoader(CloudDataset(valset), batch_size=batch_size, shuffle=True, num_workers=0)
#     print("Initializing testloader...")
#     testloader = DataLoader(CloudDataset(testset), batch_size=batch_size, shuffle=True, num_workers=0)
#     print("\nFinished initialization of the dataloaders! Some infos:\n")

#     # Verify DataLoader functionality
#     print(f"Number of batches in trainloader: {len(trainloader)}")
#     print(f"Number of batches in valloader: {len(valloader)}")
#     print(f"Number of batches in testloader: {len(testloader)}")

#     # Get insight over sizes
#     for batch_idx, (clouds, labels) in enumerate(trainloader):
#         # # Print the shapes of the batch
#         # print(f"First batch cloud shape: {clouds.shape}")
#         # print(f"First batch label shape: {labels.shape}")

#         # Print the size of one cloud
#         # Assuming clouds is a batch of shape [batch_size, channels, num_points]
#         # Example: clouds.shape = [2, 3, 213996]
#         one_cloud = clouds[0]  # Get the first cloud in the batch
#         # print(f"One cloud shape: {one_cloud.shape}")
#         print(f"One cloud size: {one_cloud.element_size() * one_cloud.nelement() / (1024**2):.2f} MB")

#         # Print the size of the first batch
#         print(f"First batch size: {clouds.element_size() * clouds.nelement() / (1024**2):.2f} MB\n")

#         if batch_idx == 0:  # Only need the first batch
#             break


#     return trainloader, valloader, testloader

def CloudLoader(filepaths, batch_size):

    # Create Datasets and DataLoaders
    loader = DataLoader(CloudDataset(filepaths), batch_size=batch_size, shuffle=True, num_workers=0)
    print("\nFinished initialization of the dataloaders! Some infos:\n")

    # Verify DataLoader functionality
    print(f"Number of batches in CloudLoader: {len(loader)}")

    # Get insight over sizes
    for batch_idx, (clouds, labels) in enumerate(loader):
        # # Print the shapes of the batch
        # print(f"First batch cloud shape: {clouds.shape}")
        # print(f"First batch label shape: {labels.shape}")

        # Print the size of one cloud
        # Assuming clouds is a batch of shape [batch_size, channels, num_points]
        # Example: clouds.shape = [2, 3, 213996]
        one_cloud = clouds[0]  # Get the first cloud in the batch
        # print(f"One cloud shape: {one_cloud.shape}")
        print(f"One cloud size: {one_cloud.element_size() * one_cloud.nelement() / (1024**2):.2f} MB")

        # Print the size of the first batch
        print(f"First batch size: {clouds.element_size() * clouds.nelement() / (1024**2):.2f} MB\n")

        if batch_idx == 0:  # Only need the first batch
            break


    return loader











################# PADDING FUNCTIONS #################

def RandomChoicePadding( data_dir, label_dir, output_dir, target_number, array_format ):
    np.random.seed(42)
    # Create output directory if not created allready
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False)

    # Load the labels and the according trees
    label_df = pd.read_csv( label_dir )
    fractal_dims = label_df['fractal_dim'].to_numpy()

    # Load the trees in label order 
    tree_list = [f'{tree}.txt' for tree in label_df['tree_id']]

    for i, t in enumerate(tree_list):
        output_path = os.path.join( output_dir, t.replace('.txt', '.npy') )
        tree_path = os.path.join(data_dir, t)
        tree = np.loadtxt(tree_path)

        # Ensure that all trees have 3 dimensions
        if tree.shape[1] != 3:
            print(f"Point cloud data in {t} does not have 3 coordinates per point, instead it has {tree.shape[1]} coordinates. Omitting unnecessary coordinates...")
            tree = tree[:,:3]

        tree_shape = tree.shape[0]

        # Check for too small tree
        if tree_shape < target_number:
            # Initialize new tree
            new_tree = tree

            # Check how many times the tree has to be recreated to match the target number
            # and append the tree so many times
            full_replications = target_number // tree_shape 
            for _ in range( full_replications - 1 ): # -1 somce the tree has been appended once already
                new_tree = np.vstack([ new_tree, tree ])

            # New draw the remaining points randomly without replacement
            number_remaining_points = target_number - new_tree.shape[0]
            drawn_point_indices = np.random.choice( tree.shape[0], size=number_remaining_points, replace=False )
            drawn_points = tree[ drawn_point_indices ]

            new_tree = np.vstack([ new_tree, drawn_points ])
            # Now save the filled tree
            if new_tree.shape[0] != target_number:
                print("Tree is too small")
        
        # Check for too large trees
        elif tree_shape > target_number:
            # draw samples from tree without replacement
            new_tree_indices = np.random.choice( tree.shape[0], size=target_number, replace=False )
            new_tree = tree[ new_tree_indices ]

            if new_tree.shape[0] != target_number:
                print("Tree is too large")

        # If the tree has the exact size just save it
        else:
            new_tree = tree

        #  save as cloud and label as .npy file
        new_npy = np.array( [(new_tree, fractal_dims[i])], dtype=array_format )
        np.save( output_path, new_npy )

    print("Random Padding Completed")

    return 


########### RANDOM AUGMENTATIONS ###########

def rotateTree(tree, angle):
    # Define rotation matrix along vertical axis, since trees should not be turned to the side
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    rotated_points = np.dot(tree, rotation_matrix)
    return rotated_points

def flipTree(tree):
    # Flips the tree horitontally
    tree[:, 0] = -tree[:, 0]  # Flip the x-coordinate
    return tree


def RandomChoiceAugmPadding( data_dir, label_dir, output_dir, target_number, array_format, num_draws, num_rots, horizontal_flip ):
    np.random.seed(42)
    # Create output directory if not created allready
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False)

    # Load the labels and the according trees
    label_df = pd.read_csv( label_dir )
    fractal_dims = label_df['fractal_dim'].to_numpy()

    # Load the trees in label order 
    tree_list = [f'{tree}.txt' for tree in label_df['tree_id']]

    # Go over tree list
    for i, t in enumerate(tree_list):
        # Draw num_draw times per tree
        for j in range(num_draws):
            output_path = os.path.join( output_dir, t.replace('.txt', f'_{j}.npy') )
            tree_path = os.path.join(data_dir, t)
            tree = np.loadtxt(tree_path)

            # Ensure that all trees have 3 dimensions
            if tree.shape[1] != 3:
                if j==0:
                    print(f"Point cloud data in {t} does not have 3 coordinates per point, instead it has {tree.shape[1]} coordinates. Omitting unnecessary coordinates...")
                tree = tree[:,:3]

            tree_shape = tree.shape[0]

            # Check for too small tree
            if tree_shape < target_number:
                # Initialize new tree
                new_tree = tree

                # Check how many times the tree has to be recreated to match the target number
                # and append the tree so many times
                full_replications = target_number // tree_shape 
                for _ in range( full_replications - 1 ): # -1 somce the tree has been appended once already
                    new_tree = np.vstack([ new_tree, tree ])

                # New draw the remaining points randomly without replacement
                number_remaining_points = target_number - new_tree.shape[0]
                drawn_point_indices = np.random.choice( tree.shape[0], size=number_remaining_points, replace=False )
                drawn_points = tree[ drawn_point_indices ]

                new_tree = np.vstack([ new_tree, drawn_points ])
                # Now save the filled tree
                if new_tree.shape[0] != target_number:
                    print("Tree is too small")
            
            # Check for too large trees
            elif tree_shape > target_number:
                # draw samples from tree without replacement
                new_tree_indices = np.random.choice( tree.shape[0], size=target_number, replace=False )
                new_tree = tree[ new_tree_indices ]

                if new_tree.shape[0] != target_number:
                    print("Tree is too large")

            # If the tree has the exact size just save it
            else:
                new_tree = tree

            # Save tree before augmentations and changing up the output path
            # save cloud and label as .npy file
            new_npy = np.array( [(new_tree, fractal_dims[i])], dtype=array_format )
            np.save( output_path, new_npy )

            # Save the horizontal flip of the drawn tree
            if horizontal_flip == True:
                aug_tree = flipTree( new_tree )
                output_path = os.path.join( output_dir, t.replace('.txt', f'_{j}_flip.npy') )
                new_npy = np.array( [(aug_tree, fractal_dims[i])], dtype=array_format )
                np.save( output_path, new_npy )

            # Now turn the trees and optionally flip them
            for k in range(num_rots):
                # Choose random turning angle
                angle = np.random.uniform( 0, 2*np.pi )

                aug_tree = rotateTree( new_tree, angle=angle )

                # Adapt output names in case of horizontal flip
                if horizontal_flip == True:
                    # First the rotation...
                    output_path = os.path.join( output_dir, t.replace('.txt', f'_{j}_{k}_0.npy') )
                    new_npy = np.array( [(aug_tree, fractal_dims[i])], dtype=array_format )
                    np.save( output_path, new_npy )
                    # ... then the flip
                    output_path = os.path.join( output_dir, t.replace('.txt', f'_{j}_{k}_1.npy') )
                    aug_tree = flipTree( aug_tree )
                    new_npy = np.array( [(aug_tree, fractal_dims[i])], dtype=array_format )
                    np.save( output_path, new_npy )
                
                # Now output without horizontal flip
                else:
                    output_path = os.path.join( output_dir, t.replace('.txt', f'_{j}_{k}.npy') )
                    new_npy = np.array( [(aug_tree, fractal_dims[i])], dtype=array_format )
                    np.save( output_path, new_npy )

                    

    print("Random Augmented Padding Completed")

    return 