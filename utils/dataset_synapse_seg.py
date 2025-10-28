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


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None,pickle_format=True,pickle_load=True):
        self.transform = transform  # using transform in torch!
        self.pickle_format = pickle_format
        self.pickle_load = pickle_load
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.pickle_data_dir = self.data_dir.replace("train_npz","train_pickle")
        if self.split == "train" and pickle_format:
            if not os.path.exists(self.pickle_data_dir):
                os.makedirs(self.pickle_data_dir)
            self.save_pickle()


    def __len__(self):
        return len(self.sample_list)
    
    def save_pickle(self):
        
        for idx in range(len(self.sample_list)):
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            sample = {'image': image, 'label': label}
            if self.transform:
                sample = self.transform(sample)

            ### save picke so that preserve sample though many repeated trials
            pickle_path =  os.path.join(self.pickle_data_dir, slice_name+'.pickle')
            with open(pickle_path, 'wb') as handle:
                pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')

            # load .pickle so that removes the AUG method
            if  self.pickle_load:
                pickle_data_path = os.path.join(self.pickle_data_dir, slice_name+'.pickle')
                with open(pickle_data_path, 'rb') as handle:
                    data_p = pickle.load(handle)
                    image, label = data_p['image'], data_p['label']

            # load .npz
            else:
                data_path = os.path.join(self.data_dir, slice_name+'.npz')
                data = np.load(data_path)
                image, label = data['image'], data['label']

        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform and not self.pickle_load:
            sample = self.transform(sample)
        


        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
    

if __name__ == "__main__" :
    from torchvision import transforms
    from torch.utils.data import DataLoader
    db_train = Synapse_dataset(base_dir='data_sets/Synapse/train_npz'
                                , list_dir='data_sets/Synapse/lists/lists_Synapse'
                                , split="train"
                                , transform=transforms.Compose([RandomGenerator(output_size=[224, 224])])
                                , pickle_format=True
                                , pickle_load= False
                            )
    
    trainloader = DataLoader(db_train, batch_size=24, shuffle=False, num_workers=1, pin_memory=True)
    max = 0
    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        # print("image_batch=",image_batch.shape)
        # print("label_batch=",label_batch.shape)
        t_max = torch.max(label_batch)
        print(t_max)
        print(torch.unique(label_batch))

    
