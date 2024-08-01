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

input_dir = os.path.join(current_dir, 'data', 'random_padding10k_aug')

epochs = 200
lr = 10**-5
GPU = True

early_stopper = True
early_stopper_patience = 15

scheduler = True
scheduler_decay = 0.1
scheduler_patience = 5

batch_size = 24

model_name = f'pointnet_10k_rot_flip_lr5_decay01'

plot_results = True
plot_dir = os.path.join(current_dir, 'plots', 'LossCurves', model_name+'png')








########## TRAINING DEFINITION & RUNNING ############

def trainPointNet(data_dir, epochs, batch_size, lr, gpu, model_name, early_stopper, early_stopper_patience, scheduler, scheduler_decay, scheduler_patience, verbose, plot_results, plot_dir):

    # Start with loading the data. Get list of all the trees
    cloud_list = [os.path.join( data_dir, f ) for f in os.listdir( data_dir ) if f.endswith('.npy') ]

    # Now get the dataloaders
    trainloader, valloader, testloader = CloudLoader(filepaths=cloud_list, batch_size=batch_size, test_size=0.1, val_size=0.1)

    # Setup device and model
    device = get_device(cuda_preference=gpu)
    torch.cuda.empty_cache()
    model = PointNet().to(device)

    # Setup the model savepath
    model_savepath = os.path.join(model_dir, model_name+'.pt')

    # Setup the optimizer and scheduler
    num_epochs = epochs

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=10**-3)
    sched = None
    if scheduler:
        sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=scheduler_decay, patience=scheduler_patience)


    # Run the training
    train_losses, val_losses = run_training(model=model, num_epochs=num_epochs, optimizer=optimizer, train_dataloader=trainloader, val_dataloader=valloader,
            device=device, loss_fn=criterion, verbose=verbose, early_stopper=early_stopper, early_stopper_path=model_savepath, patience=early_stopper_patience,
            scheduler=sched, plot_results=plot_results, save_plots_path=plot_dir)

    return

# Now run the training
trainPointNet(data_dir=input_dir, epochs=epochs, lr=lr, batch_size=batch_size, gpu=GPU, model_name=model_name, early_stopper=early_stopper,
              early_stopper_patience=early_stopper_patience, scheduler=scheduler, scheduler_decay=scheduler_decay, 
              scheduler_patience=scheduler_patience, verbose=verbose, plot_results=plot_results, plot_dir=plot_dir)