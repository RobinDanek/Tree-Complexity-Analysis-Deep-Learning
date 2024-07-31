import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

########### DEFINTION OF POINTNET AND TRAINING FUNCTIONS ############

# First define the transformation net
class TNet(nn.Module):
    def __init__(self, k):
        super(TNet, self).__init__()
        # Get dimension of k*k matrix
        self.k = k
        # Define the shared NLP and fully connected layers
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        # Define the batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

        # Initialize transformation as identity matrix
        self.fc3.weight.data.zero_() # Set weights of output layer to zero 
        self.fc3.bias.data.copy_(torch.eye(k).view(-1)) # Change bias for correspondences to identity matrix

    def forward(self, x):
        # Feed points through three shared MLPs
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        # Apply the maxpooling layer and reshape the outputs for the fully connected layers
        x = torch.max(x, 2, keepdim=True)[0] # dim=2 -> maximum feature value per channel of all points
        x = x.view(-1, 1024)
        # Feed output through fully connected layers and apply transformation
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        # Reshape to get the transformed points
        x = x.view(-1, self.k, self.k)
        return x
    

# Now define the PointNet
class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=64)
        
        # Define MLP for appliance after input transform
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        # Define MLP for appliance after feature transform
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        # Define the needed batchnormalizations
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Define the classification MLP
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1) 

        # Define the batchnormalizations for the regression MLP
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        # Define the dropout applied to the regression layer
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # First transform the input
        input_transform = self.input_transform(x) # Get the transformation matrix
        x = x.transpose(2, 1) # Swap feature and point dimensions for multiplication
        x = torch.bmm(x, input_transform) # perform batch multiplication
        x = x.transpose(2, 1) # Swap point and feature dimensions back

        # Extract features with first shared MLP
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # Now transform the features as above
        feature_transform = self.feature_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, feature_transform)
        x = x.transpose(2, 1)

        # Apply the second shared MLP and maxpooling
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        x = torch.max(x, 2, keepdim=True)[0] # Maximum feature value per feature

        x = x.view(-1, 1024) # Collapse for processing through fully connected layer

        # Apply regression MLP
        x = self.relu(self.bn6(self.fc1(x)))
        x = self.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
    


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