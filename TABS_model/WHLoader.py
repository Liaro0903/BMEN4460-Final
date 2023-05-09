import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

data_root_folder = '../../../data/WH_PVE_PPMI/'
# data_root_folder = '../data/WH_PVE_PPMI/'
class WHDataset(TensorDataset):
    # This function takes folder name ('train', 'valid', 'test') as input and creates an instance of BasicDataset according to that fodler.
    # Also if you'dd like to have less number of samples (for evaluation purposes), you may set the `n_sample` with an integer.
    def __init__(self, files, n_sample=None):
        ## Get images ids and the training images dirs

        img_path = os.path.join(data_root_folder, 'Step0_WH_T1_MNI152rigid_iso1mm')
        mask_path = os.path.join(data_root_folder, 'Step0_WB_T1_MNI152rigid_iso1mm_FAST_PVE')

        self.img_dirs_dic = {file: [
            os.path.join(img_path, file + '.nii.gz'), os.path.join(mask_path, file + '_pve_0.nii.gz'),
            os.path.join(mask_path, file + '_pve_1.nii.gz'), os.path.join(mask_path, file + '_pve_2.nii.gz')
        ] for file in files}

        # print(self.img_dirs_dic)
     
        self.ids = sorted(list(self.img_dirs_dic.keys()))
        # print(self.ids[0:10])
        self.n_sample = len(files)
            
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
        img = img[5:,21:213,:]
        img = np.append(img, np.zeros((192, 192, 3)), axis=2)
        mask0 = mask0[5:,21:213,:]
        mask0 = np.append(mask0, np.zeros((192, 192, 3)), axis=2)
        mask1 = mask1[5:,21:213,:]
        mask1 = np.append(mask1, np.zeros((192, 192, 3)), axis=2)
        mask2 = mask2[5:,21:213,:]
        mask2 = np.append(mask2, np.zeros((192, 192, 3)), axis=2)
        
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
        
        # transform = transforms.Compose([
        #     transforms.RandomVerticalFlip(p=0.5),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #     transforms.RandomRotation(degrees=15),
        #     # transforms.ToTensor()
        # ])

        return {
          'image': torch.from_numpy(img).type(torch.FloatTensor),
        #   'image': torch.from_numpy(img).type(torch.FloatTensor),
          'mask': torch.from_numpy(mask).type(torch.FloatTensor),
          'img_id': imgid
        }

def process_data(split_path=None):
    if (split_path == None):

        imgs_dir = os.path.join(data_root_folder, 'Step0_WH_T1_MNI152rigid_iso1mm')
        img_dirs = sorted(glob.glob(os.path.join(imgs_dir, '*.gz')))
        print('img_dirs len:', len(img_dirs))

        imgs_FAST_dir = os.path.join(data_root_folder, 'Step0_WB_T1_MNI152rigid_iso1mm_FAST_PVE')
        img_FAST_dirs = sorted(glob.glob(os.path.join(imgs_FAST_dir, '*.gz')))
        print('img_FAST len:', len(img_FAST_dirs))

        imgs_pveseg_dir = os.path.join(data_root_folder, 'Step0_WB_T1_MNI152rigid_iso1mm_FAST_PVE/pveseg')
        img_pveseg_dirs = sorted(glob.glob(os.path.join(imgs_pveseg_dir, '*.gz')))
        print('img_pveseg len:', len(img_pveseg_dirs))

        files = []
        files_FAST = []
        files_pveseg = []

        for i in range(len(img_dirs)):
            img_dir = img_dirs[i]

            cut_start = '_iso1mm'
            cut_end = '.nii.gz'
            idx_start = img_dir.index(cut_start)
            idx_end = img_dir.index(cut_end)
            img_dir = img_dir[idx_start + len(cut_start) + 1:idx_end]

            files.append(img_dir)
            # print(img_dir)
        
        for sample in range(len(img_FAST_dirs)):
            test = img_FAST_dirs[sample]

            cut_start = '_FAST_PVE'
            idx_start = test.index(cut_start)
            cut_ends = ['_mixeltype', '_pve', '_seg']

            for c in cut_ends:
                if test.find(c) != -1:
                    cut_end = c
                    break
                assert c != cut_ends[len(cut_ends)-1], 'found errors in ' + str(sample)
            idx_end = test.index(cut_end)

            test = test[idx_start + len(cut_start) + 1:idx_end]
            files_FAST.append(test)
            # print(test)

        for i in range(len(img_pveseg_dirs)):
            img_dir = img_pveseg_dirs[i]
            img_dir = img_dir[img_dir.index('pveseg')+len('pveseg')+1:img_dir.index('_pveseg')]
            
            files_pveseg.append(img_dir)
        #     print(img_dir)
        #     break

        print('files len:', len(files))
        print('files_FAST len:', len(files_FAST))
        print('files_pveseg len:', len(files_pveseg))

        # After unique
        print()
        print('After unique----\n')

        files = set(files)
        files_FAST = set(files_FAST)
        files_pveseg = set(files_pveseg)

        print('files len:', len(files))
        print('files_FAST len:', len(files_FAST))
        print('files_pveseg len:', len(files_pveseg))

        print('Are files and files_FAST the same?', files == files_FAST)
        print('Are files_FAST and files_pveseg the same?', files_FAST == files_pveseg)
        print()

        # How different 
        print('What are the files not in files_FAST or files_pveseg?\n')

        files_not_in = []
        for file in files:
            if file not in files_FAST:
                print(file)
                files_not_in.append(file)
                
        for file in files_not_in:
            files.remove(file)

        print()
        print('files after remove:', len(files))
        print('Do files and files_FAST equal?', files == files_FAST)

        files = list(files)

        # Train test valid split
        files_train, files_test = train_test_split(files, test_size=0.2, shuffle=True)
        files_train, files_valid = train_test_split(files_train, test_size=0.2, shuffle=True)

        with open('files_train.txt', 'w') as f:
            for file in files_train:
                f.write(file + '\n')

        with open('files_valid.txt', 'w') as f:
            for file in files_valid:
                f.write(file + '\n')

        with open('files_test.txt', 'w') as f:
            for file in files_test:
                f.write(file + '\n')
    else:
        print('Reading train, test, valid')
        with open(os.path.join(split_path, 'files_train.txt'), 'r') as file:
            files_train = file.readlines()
        files_train = [file.strip() for file in files_train]

        with open(os.path.join(split_path, 'files_test.txt'), 'r') as file:
            files_valid = file.readlines()
        files_valid = [file.strip() for file in files_valid]
        
        with open(os.path.join(split_path, 'files_test.txt'), 'r') as file:
            files_test = file.readlines()
        files_test = [file.strip() for file in files_test]

    print('files_train len:', len(files_train))
    print('files_valid len:', len(files_valid))
    print('files_test len:', len(files_test))
    return files_train, files_valid, files_test