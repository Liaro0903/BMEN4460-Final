import os
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import confusion_matrix, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader

import nibabel as nib

from torchvision import transforms

data_root_folder = '../../../data/TABSdataset/'
# data_root_folder = '../data/TABSdataset/'
class BasicDataset(TensorDataset):
    # This function takes folder name ('train', 'valid', 'test') as input and creates an instance of BasicDataset according to that fodler.
    # Also if you'dd like to have less number of samples (for evaluation purposes), you may set the `n_sample` with an integer.
    def __init__(self, folder, mode, n_sample=None):
        ## Get images ids and the training images dirs
        if (mode == 'train'):
            imgs_dir = os.path.join(data_root_folder, folder, 'Fold_2')
            img_dirs = sorted(glob.glob(os.path.join(imgs_dir, '*.gz')))
            imgs_dir = os.path.join(data_root_folder, folder, 'Fold_3')
            img_dirs.extend(sorted(glob.glob(os.path.join(imgs_dir, '*.gz'))))
            imgs_dir = os.path.join(data_root_folder, folder, 'Fold_4')
            img_dirs.extend(sorted(glob.glob(os.path.join(imgs_dir, '*.gz'))))
        elif (mode == 'valid'):
            imgs_dir = os.path.join(data_root_folder, folder, 'Fold_5')
            img_dirs = sorted(glob.glob(os.path.join(imgs_dir, '*.gz')))
        elif (mode == 'test'):
            imgs_dir = os.path.join(data_root_folder, folder, 'Fold_1')
            img_dirs = sorted(glob.glob(os.path.join(imgs_dir, '*.gz')))
        # print(len(img_dirs))
        # convert to dictionary for easier access
        self.img_dirs_dic = [(img[img.index('00'):img.index('_session')], [img]) for img in img_dirs]
        # print(self.img_dirs_dic[0:3])
        self.img_dirs_dic = {k: v for k, v in self.img_dirs_dic}
        img_dirs_ids = set(self.img_dirs_dic.keys())
        
        ## Get the truth masks dirs
        imgs_dir_fsl = os.path.join(data_root_folder, folder, 'Masks_dlbs')
        img_dirs_fsl = sorted(glob.glob(os.path.join(imgs_dir_fsl, '*.gz')))
        for imgdir in img_dirs_fsl:
            imgdir_id = imgdir[imgdir.index('00'):imgdir.index('_pve')]
            if (imgdir_id in img_dirs_ids):
                self.img_dirs_dic[imgdir_id].append(imgdir)

        for imgid in self.img_dirs_dic:
            assert len(self.img_dirs_dic[imgid]) == 4, 'There are some missing images or masks in {0}'.format(folder)
        
        # print(self.img_dirs_dic) 
        self.ids = sorted(list(self.img_dirs_dic.keys()))
        # print(self.ids)
        self.n_sample = len(self.img_dirs_dic.keys())
        
        # If n_sample is not None (It has been set by the user)
#         if not n_sample or n_sample > len(self.imgs_file):
#             n_sample = len(self.imgs_file)
        
#         self.n_sample = n_sample
#         self.ids = list([i+1 for i in range(n_sample)])
            
    # This function returns the lenght of the dataset (AKA number of samples in that set)
    def __len__(self):
        return self.n_sample
    
    
    # This function takes an index (i) which is between 0 to `len(BasicDataset)` (The return of the previous function), then returns RGB image, 
    # mask (Binary), and the index of the file name (Which we will use for visualization). The preprocessing step is also implemented in this function.
    def __getitem__(self, i):
        imgid = self.ids[i]
        # print(imgid)
        # print(self.img_dirs_dic[imgid][0])
        # print(self.img_dirs_dic[imgid][1])
        # print(self.img_dirs_dic[imgid][2])
        # print(self.img_dirs_dic[imgid][3])
        
        # Read the actual image
        img = nib.load(self.img_dirs_dic[imgid][0]).get_fdata()
        mask0 = nib.load(self.img_dirs_dic[imgid][1]).get_fdata()
        mask1 = nib.load(self.img_dirs_dic[imgid][2]).get_fdata()
        mask2 = nib.load(self.img_dirs_dic[imgid][3]).get_fdata()
        
        # Resize images to (192, 192, 192)
        img = img[:,13:205,:]
        img = np.append(img, np.zeros((10, 192, 182)), axis=0)
        img = np.append(img, np.zeros((192, 192, 10)), axis=2)
        mask0 = mask0[:,13:205,:]
        mask0 = np.append(mask0, np.zeros((10, 192, 182)), axis=0)
        mask0 = np.append(mask0, np.zeros((192, 192, 10)), axis=2)
        mask1 = mask1[:,13:205,:]
        mask1 = np.append(mask1, np.zeros((10, 192, 182)), axis=0)
        mask1 = np.append(mask1, np.zeros((192, 192, 10)), axis=2)
        mask2 = mask2[:,13:205,:]
        mask2 = np.append(mask2, np.zeros((10, 192, 182)), axis=0)
        mask2 = np.append(mask2, np.zeros((192, 192, 10)), axis=2)
        
        # Scale between 0 to 1
        # img = np.array(img) / 255.0
        # mask0 = np.array(mask0) / 255.0
        # mask1 = np.array(mask1) / 255.0
        # mask2 = np.array(mask2) / 255.0
        
        # Make sure that the mask are binary (0 or 1)
        mask0[mask0 <= 0.5] = 0.0
        mask0[mask0 > 0.5] = 1.0
        mask1[mask1 <= 0.5] = 0.0
        mask1[mask1 > 0.5] = 1.0
        mask2[mask2 <= 0.5] = 0.0
        mask2[mask2 > 0.5] = 1.0
        
        # Add an axis to the mask array so that it is in [channel, width, height] format.
        #mask = np.expand_dims(mask, axis=0)
        img = img[np.newaxis,:]
        mask = np.stack((mask0, mask1, mask2), axis=0)
        
        transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomHorizontalFlip(p=0.05),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            # transforms.ToTensor()
        ])

        return {
          'image': transform(torch.from_numpy(img).type(torch.FloatTensor)),
        #   'image': torch.from_numpy(img).type(torch.FloatTensor),
          'mask': torch.from_numpy(mask).type(torch.FloatTensor),
          'img_id': imgid
        }
