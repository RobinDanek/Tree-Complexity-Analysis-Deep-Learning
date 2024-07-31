import numpy as np
import pandas as pd

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
    splt = np.floor( len(idxs) * test_size )
    idxs_test = idxs[ splt: ]
    idxs_train = idxs[ :splt ]

    # Now the validation split
    splt = np.floor( len(idxs_train) * val_size )
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