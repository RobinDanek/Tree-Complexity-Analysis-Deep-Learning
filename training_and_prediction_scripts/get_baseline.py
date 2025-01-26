import torch
import numpy
import os
import sys

current_dir = os.getcwd()
sys.path.append(current_dir)

from modules.utils import baseline_mae
from modules.utils import get_device

test_path = os.path.join( current_dir, 'data', 'random_padding10k', 'testset' )
test_list = [os.path.join( test_path, f ) for f in os.listdir(test_path) if f.endswith('.npy')]

train_path = os.path.join( current_dir, 'data', 'random_padding10k', 'trainset' )
train_list = [os.path.join( train_path, f ) for f in os.listdir(train_path) if f.endswith('.npy')]

device = get_device

train_mae, test_mae = baseline_mae( train_list, test_list, device )

print(f"Train baseline: {train_mae:.3f}, Val baseline {test_mae:.3f}")