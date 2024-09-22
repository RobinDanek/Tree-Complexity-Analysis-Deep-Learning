import torch
import torch.nn as nn
import torch.optim as optim 
from torchmetrics.regression import RelativeSquaredError

import os
import sys

import numpy as np
import pandas as pd

current_dir = os.getcwd()
sys.path.append(current_dir)
model_dir = os.path.join(current_dir, 'model_saves', 'pointnet')

from modules.LoadingTransforming import CloudLoader
from modules.PointNet import PointNet, run_training
from modules.utils import get_device
from modules.CustomLoss import WeightedMSELoss





########## DEFINE NEEDED VARIABLES ##########

verbose = True

# train_dir = os.path.join(current_dir, 'data', 'random_padding10k', 'trainset_augmented')
train_dir = os.path.join(current_dir, 'data', 'random_padding10k', 'trainset_augmented')
val_dir = os.path.join(current_dir, 'data', 'random_padding10k', 'valset')

epochs = 300
lr = 10**-3
GPU = True

early_stopper = True
early_stopper_patience = 25

scheduler = True
scheduler_decay = 0.5
scheduler_patience = 5

batch_size = 100

model_name = f'pointnet_10k_lr3_MAE_cosWR15_mult1_min6_aug'

plot_results = True
plot_dir = os.path.join(current_dir, 'plots', 'LossCurves', model_name+'png')

# criterion = nn.MSELoss()
# criterion = RelativeSquaredError() # https://lightning.ai/docs/torchmetrics/stable/regression/rse.html
# criterion = WeightedMSELoss( 0.2, 10 )
criterion = nn.L1Loss()






########## TRAINING DEFINITION & RUNNING ############

def trainPointNet(train_dir, val_dir, epochs, batch_size, lr, gpu, model_name, early_stopper, early_stopper_patience, scheduler, scheduler_decay, scheduler_patience, verbose, plot_results, plot_dir, criterion):

    # Start with loading the data. Get list of all the trees
    train_list = [os.path.join( train_dir, f ) for f in os.listdir( train_dir ) if f.endswith('.npy') ]
    val_list = [os.path.join( val_dir, f ) for f in os.listdir( val_dir ) if f.endswith('.npy') ]

    # Now get the dataloaders
    trainloader = CloudLoader(filepaths=train_list, batch_size=batch_size)
    valloader = CloudLoader(filepaths=val_list, batch_size=batch_size)

    # Setup device and model
    device = get_device(cuda_preference=gpu)
    torch.cuda.empty_cache()
    model = PointNet().to(device)

    # Setup the model savepath
    model_savepath = os.path.join(model_dir, model_name+'.pt')

    # Setup the optimizer and scheduler
    num_epochs = epochs

    criterion = criterion
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=10**-3)
    sched = None
    if scheduler:
        # sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=scheduler_decay, patience=scheduler_patience) # Standard scheduler
        # sched = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=15, eta_min=1e-6) # Cycle down and stay low
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=15, T_mult=1, eta_min=1e-6)


    # Run the training
    train_losses, val_losses = run_training(model=model, num_epochs=num_epochs, optimizer=optimizer, train_dataloader=trainloader, val_dataloader=valloader,
            device=device, loss_fn=criterion, verbose=verbose, early_stopper=early_stopper, early_stopper_path=model_savepath, patience=early_stopper_patience,
            scheduler=sched, plot_results=plot_results, save_plots_path=plot_dir)

    return

# Now run the training
trainPointNet(train_dir=train_dir, val_dir=val_dir, epochs=epochs, lr=lr, batch_size=batch_size, gpu=GPU, model_name=model_name, early_stopper=early_stopper,
              early_stopper_patience=early_stopper_patience, scheduler=scheduler, scheduler_decay=scheduler_decay, criterion=criterion,
              scheduler_patience=scheduler_patience, verbose=verbose, plot_results=plot_results, plot_dir=plot_dir)