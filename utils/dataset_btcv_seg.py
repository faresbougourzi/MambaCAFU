import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import pickle


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class BTCV_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir



    def __len__(self):
        return len(self.sample_list)
    

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']

        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform :
            sample = self.transform(sample)
        


        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
    

class BTCV_datasetV2(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

        if self.split == "train":
            self.label_dir = "data_sets/Synapse/dataset13classes/train_npz_new"
        else:
            self.label_dir = "data_sets/Synapse/dataset13classes/test_vol_h5_new"


    def __len__(self):
        return len(self.sample_list)
    

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            # print(data_path)
            data = np.load(data_path)
            image = data['image']

            label_path = os.path.join(self.label_dir, slice_name+'.npz')
            # print(label_path)
            data_v1 = np.load(label_path)
            label = data_v1['label'] # get 13 classess
            # image, label = data['image'], data['label'] # get 8 classes
            

        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            # image, label = data['image'][:], data['label'][:]
            image = data['image'][:]

            filepath_label = self.label_dir + "/{}.npy.h5".format(vol_name) # get 13 classes
            data_v1 = h5py.File(filepath_label)
            label = data_v1['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform :
            sample = self.transform(sample)
        


        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
    


if __name__ == "__main__" :
    from torchvision import transforms
    from torch.utils.data import DataLoader
    db_train = BTCV_datasetV2(base_dir='data_sets/Synapse/train_npz'
                                , list_dir='data_sets/Synapse/lists/lists_Synapse'
                                , split="train"
                                , transform=transforms.Compose([RandomGenerator(output_size=[224, 224])])
                               
                            )
    
    trainloader = DataLoader(db_train, batch_size=100, shuffle=False, num_workers=1, pin_memory=True)

    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        print(i_batch)
        print("image_batch=",image_batch.shape)
        print("label_batch=",torch.min(label_batch), torch.max(label_batch))
        print("label_batch=",torch.unique(label_batch))
        break
        

    
