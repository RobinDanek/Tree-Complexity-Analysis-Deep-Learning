import os
import sys

import numpy as np
import pandas as pd

current_dir = os.getcwd()
sys.path.append(current_dir)

from modules.LoadingTransforming import CloudSplitter
from modules.utils import TEST_SIZE, VAL_SIZE

cloud_dir = os.path.join(current_dir, 'data', 'random_padding100k')

print(f"Splitting data found in {cloud_dir}")

cloud_list = [os.path.join( cloud_dir, f ) for f in os.listdir( cloud_dir ) if f.endswith('.npy') ]

trainset, valset, testset = CloudSplitter(cloud_list, test_size=TEST_SIZE, val_size=VAL_SIZE, output_dir=cloud_dir)

print(f"Completed split. Sizes:\ntrain: {len(trainset)}\tval: {len(valset)}\ttest: {len(testset)}")