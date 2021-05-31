# Dependencies
# ----------------------------------------------------------------------------
import cv2
import json
import urllib
import numpy as np
from tqdm import tqdm
from pathlib import Path
import PIL.Image as Image
from IPython.display import display
import matplotlib.pyplot as plt
from pylab import rcParams
from matplotlib import rc
import seaborn as sns

#%matplotlib inline
#%config InlineBackend.figure_format='retina'
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
rcParams['figure.figsize'] = 16, 10

import torch
import torchvision
from sklearn.model_selection import train_test_split

np.random.seed(111)

# Datasets
# ----------------------------------------------------------------------------

clothing = [ ]

with open('./dataset/clothing.json') as f:
    for line in f:
        clothing.append(json.loads(line))

print(clothing[0])