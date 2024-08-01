import torch
import torch.nn as nn
import torch.optim as optim 

import os
import sys

import numpy as np
import pandas as pd

current_dir = os.getcwd()
sys.path.append(current_dir)
model_dir = os.path.join(current_dir, 'model_saves', 'pointnet')

from modules.LoadAndTrans.LoadingTransforming import CloudLoader
from modules.Models.PointNet import PointNet, run_training
from modules.Utils.utils import get_device





########## DEFINE NEEDED VARIABLES ##########

verbose = True

input_dir = os.path.join(current_dir, 'data', 'random_padding10k')

epochs = 100
lr = 10**-4
GPU = True

model_name = 'pointnet_10k_2'

plot_results = True
plot_dir = os.path.join(current_dir, 'plots', 'LossCurves', model_name+'png')

early_stopper = True
early_stopper_patience = 15

scheduler = True
scheduler_decay = 0.5
scheduler_patience = 5








########## TRAINING DEFINITION & RUNNING ############

def trainPointNet(data_dir, epochs, lr, gpu, model_name, early_stopper, early_stopper_patience, scheduler, scheduler_decay, scheduler_patience, verbose, plot_results, plot_dir):

    # Start with loading the data. Get list of all the trees
    cloud_list = [os.path.join( data_dir, f ) for f in os.listdir( data_dir ) if f.endswith('.npy') ]

    # Now get the dataloaders
    trainloader, valloader, testloader = CloudLoader(filepaths=cloud_list, batch_size=5, test_size=0.1, val_size=0.1)

    # Setup device and model
    device = get_device(cuda_preference=gpu)
    torch.cuda.empty_cache()
    model = PointNet().to(device)

    # Setup the model savepath
    model_savepath = os.path.join(model_dir, model_name+'.pt')

    # Setup the optimizer and scheduler
    num_epochs = epochs

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if scheduler:
        sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=scheduler_decay, patience=scheduler_patience)
    else:
        sched = None

    # Run the training
    train_losses, val_losses = run_training(model=model, num_epochs=num_epochs, optimizer=optimizer, train_dataloader=trainloader, val_dataloader=valloader,
            device=device, loss_fn=criterion, verbose=verbose, early_stopper=early_stopper, early_stopper_path=model_savepath, patience=early_stopper_patience,
            scheduler=sched, plot_results=plot_results, save_plots_path=plot_dir)

    return

# Now run the training
trainPointNet(data_dir=input_dir, epochs=epochs, lr=lr, gpu=GPU, model_name=model_name, early_stopper=early_stopper,
              early_stopper_patience=early_stopper_patience, scheduler=scheduler, scheduler_decay=scheduler_decay, 
              scheduler_patience=scheduler_patience, verbose=verbose, plot_results=plot_results, plot_dir=plot_dir)