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

clothing = []

with open('./dataset/clothing.json') as f:
    for line in f:
        clothing.append(json.loads(line))

# print(clothing[0])

# Labels
# ----------------------------------------


# Images with multiple annotation ?
# for c in clothing:
#     if len(c['annotation']) > 1:
#         print(c)

categories = []

for c in clothing:
    for a in c['annotation']:
        categories.extend(a['label'])

# sorted list of labels
categories = list(set(categories))
categories.sort()

# Dataset Info
# ----------------------------------------

data_info = f"""

Labels : {categories}

"""
print(data_info)

# Train/ Test Split
# ----------------------------------------------------------------------------

train_set, dev_set = train_test_split(clothing, test_size = 0.1)
print(len(train_set), len(dev_set))