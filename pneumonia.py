"""
Created on Sun Nov 23 13:12:25 2025

@author: Mozhdeh
"""
# Importing the Required Libraries
import os
from pathlib import Path
import pydicom
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import glob
import ipdb
 

# Defining Folder PATH
folder_path = "D:/Courses/medical imaging classification with python/kaggle-pneumonia/" 

# Loading Labels and Checking Some Samples
labels = pd.read_csv(folder_path + "stage_2_train_labels.csv")
labels.head(10)

# Handling Data Duplications for Those with The Same IDs
labels = labels.drop_duplicates("patientId")
labels.head(10)

# Train Data Path
root_path = Path(folder_path + "stage_2_train_images/")
save_path = Path(folder_path + "processed_data/")

""" Loading the DCM Data """

fig, ax = plt.subplots(3,3, figsize=(9,9)) 
cntr = 0
for i in range(3):
    for j in range(3):
        # Selecting Each of The Data Samples
        patient_ID = labels.patientId.iloc[cntr]
        dcm_path = root_path/patient_ID
        dcm_path = dcm_path.with_suffix(".dcm")
        print(dcm_path)
        
        # Loading Their DICOM Images and The Labels (Target)
        dcm = pydicom.dcmread(dcm_path).pixel_array
        label = labels["Target"].iloc[cntr]
        
        ax[i][j].imshow(dcm, cmap = 'bone')
        ax[i][j].set_title(label)
               
        cntr += 1        
        
plt.tight_layout()  
plt.show()        
        
    
# Checking The Number of Data for Test and Train
train_len = len(os.listdir(folder_path + "stage_2_train_images"))
test_len = len(os.listdir(folder_path + "stage_2_test_images"))

""" Data Normalization and Train-Val Split"""

sums = 0
sums_sqrt = 0
split_cri = round(train_len * 0.9)

for cntr , patient_ID in enumerate(tqdm(labels.patientId)):
    dcm_path = root_path/patient_ID
    dcm_path = dcm_path.with_suffix(".dcm")
    
    if os.path.exists(dcm_path):
        dcm = pydicom.dcmread(dcm_path).pixel_array/255
        dcm_resized = cv2.resize(dcm , (224, 224)).astype(np.float16)
        label = labels.Target.iloc[cntr]
        
        train_val = "train" if cntr < split_cri else "val"
        
        my_save_path = save_path/train_val/str(label)
        my_save_path.mkdir(parents = True, exist_ok = True)
        np.save(my_save_path/patient_ID, dcm_resized)
 
        normalizer = dcm_resized.shape[0] * dcm_resized.shape[1]
        if train_val == "train":
            sums += np.sum(dcm_resized)/normalizer
            sums_sqrt += (np.power(dcm_resized, 2).sum())/normalizer































