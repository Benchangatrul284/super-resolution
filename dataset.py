import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import cv2
from natsort import natsorted
import random

# data augmentation
def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(a) for a in args]
class DIV2K(Dataset):
    def __init__(self, root,repeat=10,patch_size = 192,mode = 'train'):
        super(DIV2K, self).__init__()
        self.root = root
        self.repeat = repeat
        self.mode = mode
        # list all images
        if self.mode == 'train':
            self.images_hr = natsorted(os.listdir(os.path.join(self.root, 'DIV2K_HR_train')))
            self.images_lr = natsorted(os.listdir(os.path.join(self.root, 'DIV2K_LR_bicubic_train', 'X3')))
        elif self.mode == 'val':
            self.images_hr = natsorted(os.listdir(os.path.join(self.root, 'DIV2K_HR_valid')))
            self.images_lr = natsorted(os.listdir(os.path.join(self.root, 'DIV2K_LR_bicubic_valid')))
        
        self.hr_list = []
        self.lr_list = []
        # get full path of images
        if self.mode == 'train':
            for i in self.images_hr:
                self.hr_list.append(os.path.join(self.root, 'DIV2K_HR_train', i))
            for i in self.images_lr:
                self.lr_list.append(os.path.join(self.root, 'DIV2K_LR_bicubic_train', 'X3', i))
        elif self.mode == 'val':
            for i in self.images_hr:
                self.hr_list.append(os.path.join(self.root, 'DIV2K_HR_valid', i))
            for i in self.images_lr:
                self.lr_list.append(os.path.join(self.root, 'DIV2K_LR_bicubic_valid', i))
        self.patch_size = patch_size
        print('Number of images: {}'.format(len(self.hr_list)))

    def __getitem__(self, idx):
        img_hr, img_lr = self.load_file(idx) # load the lr and hr images into numpy array
        img_hr, img_lr = self.get_patches(img_hr, img_lr) # extract patches from images
        # img_hr, img_lr = augment(img_hr, img_lr) # data augmentation
        img_hr, img_lr = self.toTensor(img_hr, img_lr) # convert numpy array to tensor
        return img_hr, img_lr
    
    def __len__(self):
        return len(self.images_hr)*self.repeat
    
    def load_file(self, idx):
        '''
        load numpy array from file
        '''
        idx = idx % len(self.hr_list) #for repeat
        img_hr = np.load(self.hr_list[idx])
        img_lr = np.load(self.lr_list[idx])
        return img_hr, img_lr
    
    def get_patches(self, img_hr, img_lr):
        '''
        extract patches from images
        '''
        patch_size = self.patch_size
        scale = 3
        # t for target, i for input
        tp = patch_size
        ip = tp // scale
        ih, iw, _ = img_lr.shape

        ix = random.randrange(0, iw - ip+1)
        iy = random.randrange(0, ih - ip+1)
        tx, ty = scale * ix, scale * iy

        img_lr = img_lr[iy:iy + ip, ix:ix + ip, :]
        img_hr = img_hr[ty:ty + tp, tx:tx + tp, :]
        img_hr, img_lr = augment(img_hr, img_lr)
        return img_hr, img_lr
    
    def toTensor(self, img_hr, img_lr):
        '''
        convert numpy array to tensor, HWC->CHW
        '''
        img_hr = np.ascontiguousarray(img_hr.transpose(2, 0, 1)) # convert HWC to CHW
        tensor_hr = torch.from_numpy(img_hr).float() # convert numpy array to tensor
        tensor_hr.div_(255.0)
        img_lr = np.ascontiguousarray(img_lr.transpose(2, 0, 1)) # convert HWC to CHW
        tensor_lr = torch.from_numpy(img_lr).float() # convert numpy array to tensor
        tensor_lr.div_(255.0)
        return tensor_hr, tensor_lr
    
class FLR2K(Dataset):
    def __init__(self,root,repeat,patch_size = 192):
        super(FLR2K, self).__init__()
        self.root = root
        self.repeat = repeat
        self.images_hr = natsorted(os.listdir(os.path.join(self.root, 'Flickr2K_HR')))
        self.images_lr = natsorted(os.listdir(os.path.join(self.root, 'Flickr2K_LR_bicubic')))
        self.hr_list = []
        self.lr_list = []
        # get full path of images
    
        for image in self.images_hr:
            self.hr_list.append(os.path.join(self.root, 'Flickr2K_HR',image))
        for image in self.images_lr:
            self.lr_list.append((os.path.join(self.root, 'Flickr2K_LR_bicubic',image)))
        self.patch_size = patch_size
        print('Number of images: {}'.format(len(self.hr_list)))

    def __getitem__(self, idx):
        img_hr, img_lr = self.load_file(idx) # load the lr and hr images into numpy array
        img_hr, img_lr = self.get_patches(img_hr, img_lr) # extract patches from images
        img_hr, img_lr = augment(img_hr, img_lr) # data augmentation
        img_hr, img_lr = self.toTensor(img_hr, img_lr) # convert numpy array to tensor
        return img_hr, img_lr
    
    def __len__(self):
        return len(self.images_hr)*self.repeat
    
    def load_file(self, idx):
        '''
        load numpy array from file
        '''
        idx = idx % len(self.hr_list) #for repeat
        img_hr = np.load(self.hr_list[idx])
        img_lr = np.load(self.lr_list[idx])
        return img_hr, img_lr
    
    def get_patches(self, img_hr, img_lr):
        '''
        extract patches from images
        '''
        patch_size = self.patch_size
        scale = 3
        # t for target, i for input
        tp = patch_size
        ip = tp // scale
        ih, iw, _ = img_lr.shape
        # find the starting point of the patch
        ix = random.randrange(0, iw - ip+1) 
        iy = random.randrange(0, ih - ip+1) 
        tx, ty = scale * ix, scale * iy 
        #cut the patch out
        img_lr = img_lr[iy:iy + ip, ix:ix + ip, :]
        img_hr = img_hr[ty:ty + tp, tx:tx + tp, :]
        return img_hr, img_lr
    
    def toTensor(self, img_hr, img_lr):
        '''
        convert numpy array to tensor, HWC->CHW
        ''' 
        img_hr = np.ascontiguousarray(img_hr.transpose(2, 0, 1)) # convert HWC to CHW
        tensor_hr = torch.from_numpy(img_hr).float() # convert numpy array to tensor
        tensor_hr.div_(255.0)
        img_lr = np.ascontiguousarray(img_lr.transpose(2, 0, 1)) # convert HWC to CHW
        tensor_lr = torch.from_numpy(img_lr).float() # convert numpy array to tensor
        tensor_lr.div_(255.0)
        return tensor_hr, tensor_lr
    
class Butterfly(Dataset):
    def __init__(self,root,repeat,patch_size = 192):
        super(Butterfly, self).__init__()
        self.root = root
        self.repeat = repeat
        self.images_hr = natsorted(os.listdir(os.path.join(self.root, 'butterfly_HR')))
        self.images_lr = natsorted(os.listdir(os.path.join(self.root, 'butterfly_LR')))
        self.hr_list = []
        self.lr_list = []
        # get full path of images
    
        for i in self.images_hr:
            self.hr_list.append(os.path.join(self.root,'butterfly_HR',i))
        for i in self.images_lr:
            self.lr_list.append((os.path.join(self.root,'butterfly_LR',i)))
        self.patch_size = patch_size
        print('Number of images: {}'.format(len(self.hr_list)))

    def __getitem__(self, idx):
        img_hr, img_lr = self.load_file(idx) # load the lr and hr images into numpy array
        img_hr, img_lr = self.get_patches(img_hr, img_lr) # extract patches from images
        img_hr, img_lr = augment(img_hr, img_lr) # data augmentation
        img_hr, img_lr = self.toTensor(img_hr, img_lr) # convert numpy array to tensor
        return img_hr, img_lr
    
    def __len__(self):
        return len(self.images_hr)*self.repeat
    
    def load_file(self, idx):
        '''
        load numpy array from file
        '''
        idx = idx % len(self.hr_list) #for repeat
        img_hr = np.load(self.hr_list[idx])
        img_lr = np.load(self.lr_list[idx])
        return img_hr, img_lr
    
    def get_patches(self, img_hr, img_lr):
        '''
        extract patches from images
        '''
        patch_size = self.patch_size
        scale = 3
        # t for target, i for input
        tp = patch_size
        ip = tp // scale
        ih, iw, _ = img_lr.shape
        # find the starting point of the patch
        ix = random.randrange(0, iw - ip+1) 
        iy = random.randrange(0, ih - ip+1) 
        tx, ty = scale * ix, scale * iy 
        #cut the patch out
        img_lr = img_lr[iy:iy + ip, ix:ix + ip, :]
        img_hr = img_hr[ty:ty + tp, tx:tx + tp, :]
        return img_hr, img_lr
    
    def toTensor(self, img_hr, img_lr):
        '''
        convert numpy array to tensor, HWC->CHW
        ''' 
        img_hr = np.ascontiguousarray(img_hr.transpose(2, 0, 1)) # convert HWC to CHW
        tensor_hr = torch.from_numpy(img_hr).float() # convert numpy array to tensor
        tensor_hr.div_(255.0)
        img_lr = np.ascontiguousarray(img_lr.transpose(2, 0, 1)) # convert HWC to CHW
        tensor_lr = torch.from_numpy(img_lr).float() # convert numpy array to tensor
        tensor_lr.div_(255.0)
        return tensor_hr, tensor_lr