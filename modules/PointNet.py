# Important imports
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fastprogress
import seaborn as sns

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
    def __init__(self, T1=3, T2=64):
        super(PointNet, self).__init__()
        self.input_transform = TNet(k=T1)
        self.feature_transform = TNet(k=T2)
        
        # Define MLP for appliance after input transform
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, T2, 1)
        # Define MLP for appliance after feature transform
        self.conv3 = nn.Conv1d(T2, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        # Define the needed batchnormalizations
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(T2)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Define the classification MLP
        self.fc1 = nn.Linear(1024, 512)
        # self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1) 

        # self.fc4 = nn.Linear(256,256)
        # self.fc5 = nn.Linear(256,256)

        # Define the batchnormalizations for the regression MLP
        self.bn6 = nn.BatchNorm1d(512)
        # self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(256)

        # self.bn8 = nn.BatchNorm1d(256)
        # self.bn9 = nn.BatchNorm1d(256)

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
        # x = self.relu(self.bn8(self.fc4(x)))
        # x = self.relu(self.bn9(self.fc5(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
    

############### TRAINING FUNCTIONS ######################

def print_memory_summary():
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

def train(dataloader, optimizer, model, master_bar, device, loss_fn = nn.MSELoss()):
    """Run one training epoch.

    Args:
        dataloade: dataloader containing trainingdata
        optimizer: Torch optimizer object
        model: the model that is trained
        loss_fn: the loss function to be used -> nn.MSELoss()
        master_bar: Will be iterated over for each
            epoch to draw batches and display training progress

    Returns:
        Mean epoch loss and accuracy
    """
    scaler = GradScaler() # Initialize the gradient scaler for mixed precision training
    losses = []  # Use a list to store individual batch losses
    mean_absolute_errors = []
    MAErr = nn.L1Loss()

    for batch_idx, (x, y) in enumerate(fastprogress.progress_bar(dataloader, parent=master_bar)):
        # if batch_idx == 10:
        #     print(f"Memory allocated before batch: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        
        optimizer.zero_grad()
        model.train()

        # if batch_idx == 10:
        #     print(f"Memory allocated after model.train(): {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")


        # if batch_idx == 10:
        #     print_memory_summary()
        # # Forward pass
        # y_pred = model(x.to(device, non_blocking=True))
        # y_pred = y_pred.squeeze(dim=1)  # Removes dimension with index 1
        # if batch_idx == 10:
        #     print(f"Memory allocated after prediction: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        # if batch_idx == 10:
        #     print_memory_summary()

        # # Compute loss
        # batch_loss = loss_fn(y_pred, y.to(device, non_blocking=True))
        # if batch_idx == 10:
        #     print(f"Memory allocated after loss computation: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

        # # Backward pass
        # batch_loss.backward()
        # if batch_idx == 10:
        #     print(f"Memory allocated after backward pass: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

        # optimizer.step()

        with autocast():  # Enable mixed precision for the forward pass
            y_pred = model(x.to(device, non_blocking=True))
            y_pred = y_pred.squeeze(dim=1)
            # if batch_idx == 10:
            #     print(f"Memory allocated after prediction: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

            batch_loss = loss_fn(y_pred, y.to(device, non_blocking=True))
            MAE_loss = MAErr(y_pred, y.to(device, non_blocking=True))

        scaler.scale(batch_loss).backward()  # Scale the loss and perform the backward pass
        scaler.step(optimizer)
        scaler.update()  # Update the scaler for the next iteration

        # Save the batch loss for logging purposes
        losses.append(batch_loss.item())
        mean_absolute_errors.append(MAE_loss.item())

        # Clean up GPU memory
        # del x, y, y_pred, batch_loss
        # torch.cuda.empty_cache()

        # if batch_idx == 10:
        #     print(f"Memory allocated after batch: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

    # Calculate the mean loss for the epoch
    mean_loss = np.mean(losses)
    mean_MAE = np.mean(mean_absolute_errors)

    # Return the mean loss for the epoch
    return mean_loss, mean_MAE





def validate(dataloader, model, master_bar, device, loss_fn=nn.MSELoss()):
    """Compute loss and total prediction error on validation set.

    Args:
        dataloader: dataloader containing validation data
        model (nn.Module): the model to train
        loss_fn: the loss function to be used, defaults to MSELoss
        master_bar (fastprogress.master_bar): Will be iterated over to draw 
            batches and show validation progress

    Returns:
        Mean loss and total prediction error on validation set
    """
    epoch_loss = []
    epoch_MAE = []
    MAErr = nn.L1Loss()

    model.eval()
    with torch.no_grad():
        for x, y in fastprogress.progress_bar(dataloader, parent=master_bar):
            # make a prediction on validation set
            y_pred = model(x.to(device, non_blocking=True))
            y_pred = y_pred.squeeze(dim=1)  # Removes dimension with index 1

            # Compute loss
            loss = loss_fn(y_pred, y.to(device, non_blocking=True))
            MAE_loss = MAErr(y_pred, y.to(device, non_blocking=True))

            # For plotting the train loss, save it for each sample
            epoch_loss.append(loss.item())
            epoch_MAE.append(MAE_loss.item())

    # Return the mean loss, the accuracy and the confusion matrix
    return np.mean(epoch_loss), np.mean(epoch_MAE)


def test(dataloader, model, device, loss_fn=nn.MSELoss()):
    """Compute loss on testset.

    Args:
        dataloader: dataloader containing validation data
        model (nn.Module): the model to train
        loss_fn: the loss function to be used, defaults to MSELoss

    Returns:
        Mean loss 
    """
    epoch_loss = []

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            # make a prediction on test set
            y_pred = model(x.to(device, non_blocking=True))

            # Compute loss
            loss = loss_fn(y_pred, y.to(device, non_blocking=True))

            # For plotting the train loss, save it for each sample
            epoch_loss.append(loss.item())

    # Return the mean loss, the accuracy and the confusion matrix
    return np.mean(epoch_loss)



def plot(title, label, train_results, val_results, yscale='linear', save_path=None):
    """Plot learning curves.

    Args:
        title: Title of plot
        label: y-axis label
        train_results: Vector containing training results over epochs
        val_results: vector containing validation results over epochs
        yscale: Defines how the y-axis scales
        save_path: Optional path for saving file
    """
    
    epochs = np.arange(len(train_results)) + 1
    
    sns.set(style='ticks')

    plt.plot(epochs, train_results, epochs, val_results, linestyle='dashed', marker='o')
    legend = ['Train results', 'Validation results']
        
    plt.legend(legend)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.yscale(yscale)
    plt.title(title)
    
    sns.despine(trim=True, offset=5)
    plt.title(title, fontsize=15)
    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.show()


class EarlyStopper:
    """Early stops the training if validation accuracy does not increase after a
    given patience. Saves and loads model checkpoints.
    """
    def __init__(self, verbose=False, path='checkpoint.pt', patience=1):
        """Initialization.

        Args:
            verbose (bool, optional): Print additional information. Defaults to False.
            path (str, optional): Path where checkpoints should be saved. 
                Defaults to 'checkpoint.pt'.
            patience (int, optional): Number of epochs to wait for increasing
                accuracy. If accuracy does not increase, stop training early. 
                Defaults to 1.
        """
        ####################
        self.verbose = verbose
        self.path = path
        self.patience = patience
        self.counter = 0
        ####################

    @property
    def early_stop(self):
        """True if early stopping criterion is reached.

        Returns:
            [bool]: True if early stopping criterion is reached.
        """
        if self.counter == self.patience:
            return True

    def save_model(self, model):

        # Ensure the directory exists
        directory = os.path.dirname(self.path)
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist. Creating directory.")
            os.makedirs(directory, exist_ok=True)

        try:
            torch.save(model.state_dict(), self.path)
        except Exception as e:
            print(f"Error saving the model: {e}")
            
        return
        
    def check_criterion(self, loss_val_new, loss_val_old):
        if loss_val_old <= loss_val_new:
            self.counter += 1
        else:
            self.counter = 0

        
        return
    
    def load_checkpoint(self, model):
        # model = torch.jit.load(self.path)

        model.load_state_dict(torch.load(self.path))
        return model



def get_model_memory_size(model):
    num_params = sum(p.numel() for p in model.parameters())
    # Assuming float32 (4 bytes) for each parameter
    memory_size = num_params * 4  # in bytes
    memory_size = memory_size / (1024 ** 2)  # Convert to MB
    print(f'Model size: {memory_size:.2f} MB')
    return



def run_training(model, optimizer, num_epochs, train_dataloader, val_dataloader, device, 
                 loss_fn=nn.MSELoss(), patience=1, early_stopper=None, early_stopper_path=None, scheduler=None, verbose=False, plot_results=True, save_plots_path=None):
    """Run model training.

    Args:
        model: The model to be trained
        optimizer: The optimizer used during training
        loss_fn: Torch loss function for training -> nn.MSELoss()
        num_epochs: How many epochs the model is trained for
        train_dataloader:  dataloader containing training data
        val_dataloader: dataloader containing validation data
        verbose: Whether to print information on training progress

    Returns:
        lists containing  losses and total prediction errors per epoch for training and validation
    """

    if verbose:
        get_model_memory_size(model)

    start_time = time.time()
    master_bar = fastprogress.master_bar(range(num_epochs))
    train_losses, val_losses = [],[]

    if early_stopper:
        ES = EarlyStopper(verbose=verbose, patience = patience, path=early_stopper_path)

    # initialize old loss value varibale (choose something very large)
    val_MAE_loss_old = 1e6

    for epoch in master_bar:
        # Train the model
        epoch_train_loss, epoch_train_MAE_loss = train(dataloader=train_dataloader, optimizer=optimizer, model=model, 
                                                 master_bar=master_bar, device=device, loss_fn=loss_fn)
        # Validate the model
        epoch_val_loss, epoch_val_MAE_loss = validate(val_dataloader, model, master_bar, device, loss_fn)

        # Save loss and acc for plotting
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        if verbose:
            master_bar.write(f'Train loss: {epoch_train_loss:.3f}, train MAE: {epoch_train_MAE_loss:.3f}, val loss: {epoch_val_loss:.3f}, val MAE: {epoch_val_MAE_loss:.3f}')


        if early_stopper and epoch != 0:
            ES.check_criterion(epoch_val_MAE_loss, val_MAE_loss_old)
            if ES.early_stop:
                master_bar.write("Early stopping")
                model = ES.load_checkpoint(model)
                break
            if ES.counter > 0 and verbose:
                master_bar.write(f"Early stop counter: {ES.counter} / {patience}")

        # Save smallest loss
        if early_stopper and epoch_val_MAE_loss < val_MAE_loss_old:
            val_MAE_loss_old = epoch_val_MAE_loss
            ES.save_model(model)
            
        if scheduler:
            # scheduler.step(epoch_val_loss)
            # scheduler.step(epoch_val_MAE_loss)
            scheduler.step()

    time_elapsed = np.round(time.time() - start_time, 0).astype(int)
    print(f'Finished training after {time_elapsed} seconds.')

    if plot_results:
        plot("Loss", "Loss", train_losses, val_losses, save_path=save_plots_path)
    return train_losses, val_losses