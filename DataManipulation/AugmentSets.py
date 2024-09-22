import os
import sys

import numpy as np
import pandas as pd

current_dir = os.getcwd()
sys.path.append(current_dir)

from modules.LoadingTransforming import DataAugmentation
from modules.ArrayTypes import cloudLabelType_10k

# The directory in which the data that is to be augmented is stored
input_dir = os.path.join(current_dir, 'data', 'random_padding10k', 'trainset')
# The directory in which the augmented data should be stored
output_dir = os.path.join(current_dir, 'data', 'random_padding10k', 'trainset_augmented')


print(f"\nStarting augmentation of files found in:\n{input_dir}\nStoring in:\n{output_dir}")
DataAugmentation(input_dir=input_dir, output_dir=output_dir, array_format=cloudLabelType_10k, 
                 num_rots=3, horizontal_flip=True, flip_rots=True)
print("Finished augmentation!")

