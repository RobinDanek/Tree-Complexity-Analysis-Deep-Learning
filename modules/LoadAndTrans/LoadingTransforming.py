import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

################# DATASET FUNCTIONS #######################

# Cloud Dataset
class CloudDataset(Dataset):
    # Loads set of pointclouds and passes one with according label
    def __init__(self, cloudset, labels):
        self.clouds = cloudset
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        cloud = self.clouds[idx]
        label = self.labels[idx]
        return cloud, label
    
# Test Val Train Split
def CloudSplitter(cloudset, labels, test_size,  val_size, seed=42):
    # Split the indices randomly
    np.random.seed( seed )
    idxs = np.arange( len(labels) )
    np.random.shuffle( idxs )

    # First train test split
    splt = int(np.floor( len(idxs) * test_size ))
    idxs_test = idxs[ splt: ]
    idxs_train = idxs[ :splt ]

    # Now the validation split
    splt = int(np.floor( len(idxs_train) * val_size ))
    idxs_val = idxs_train[ splt: ]
    idxs_train = idxs_train[ :splt ]

    # Perform the splits
    trainset = cloudset[ idxs_train ]
    labels_train = labels[ idxs_train ]
    testset = cloudset[ idxs_test ]
    labels_test = labels[ idxs_test ]
    valset = cloudset[ idxs_val ]
    labels_val = labels[ idxs_val ]

    return trainset, valset, testset, labels_train, labels_test, labels_val

################# PADDING FUNCTIONS #################

def RandomChoicePadding( data_dir, output_dir, target_number ):
    # Create output directory if not created allready
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False)
    
    # Now load the trees, augment them and store them
    tree_list = [f for f in os.listdir( data_dir ) if f.endswith('.txt')]
    for t in tree_list:
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

        #  save as .npy file
        np.save( output_path, new_tree )

    print("Random Padding Completed")

    return 