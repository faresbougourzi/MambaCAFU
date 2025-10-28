from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
from einops.layers.torch import Rearrange
from scipy.ndimage.morphology import binary_dilation

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ===== normalize over the dataset 
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized
       
    
class weak_annotation(torch.nn.Module):
    def __init__(self, patch_size = 16, img_size = 256):
        super().__init__()
        self.arranger = Rearrange('c (ph h) (pw w) -> c (ph pw) h w', c=1, h=patch_size, ph=img_size//patch_size, w=patch_size, pw=img_size//patch_size)
    def forward(self, x):
        x = self.arranger(x)
        x = torch.sum(x, dim = [-2, -1])
        x = x/x.max()
        return x
    
def Bextraction(img):
    img = img[0].numpy()
    img2 = binary_dilation(img, structure=np.ones((7,7))).astype(img.dtype)
    img3 = img2 - img
    img3 = np.expand_dims(img3, axis = 0)
    return torch.tensor(img3.copy())

train_transform = A.Compose(
    [
        A.Resize(height=224, width=224),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)
######
val_transforms = A.Compose(
    [
        A.Resize(height=224, width=224),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)


## Temporary
class isic_loader(Dataset):
    """ dataset class for Brats datasets
    """
    def __init__(self, path_Data="data_sets/ISIC2017/processed/", train = True, Test = False):
        super(isic_loader, self)
        self.train = train
        if train:
            self.data   = np.load(path_Data+'data_train.npy')
            self.mask   = np.load(path_Data+'mask_train.npy')
            self.transf = train_transform
        else:
            if Test:
                self.data   = np.load(path_Data+'data_test.npy')
                self.mask   = np.load(path_Data+'mask_test.npy')
                self.transf = val_transforms
                
            else:
                self.data   = np.load(path_Data+'data_val.npy')
                self.mask   = np.load(path_Data+'mask_val.npy')  
                self.transf = val_transforms        
          
          
        # self.data   = dataset_normalized(self.data)
        # self.mask   = np.expand_dims(self.mask, axis=3)

        # self.mask   = self.mask /255.
        # self.weak_annotation = weak_annotation(patch_size = 16, img_size = 256) #224
         
    def __getitem__(self, indx):
        img = self.data[indx]
        seg = self.mask[indx]

        # if self.train:
        #     img, seg = self.apply_augmentation(img, seg)
        
        # seg = torch.tensor(seg.copy())
        # img = torch.tensor(img.copy())
        # img = img.permute( 2, 0, 1)
        # seg = seg.permute( 2, 0, 1)

        img = np.array(img)
        seg = np.array(seg)

        img = img.astype(np.uint8) 
        seg = seg.astype(np.uint8) 
 

        seg[seg > 0.0] = 1.0

        
        augmentations = self.transf(image=img, mask=seg)
        image = augmentations["image"]
        mask = augmentations["mask"]

        return {'image': image,
                'mask' : mask}
               
    def apply_augmentation(self, img, seg):
        if random.random() < 0.5:
            img  = np.flip(img,  axis=1)
            seg  = np.flip(seg,  axis=1)
        return img, seg

    def __len__(self):
        return len(self.data)
    

if __name__ == "__main__":
    db_train = isic_loader(path_Data="data_sets/ISIC2017/processed/", train = False, Test = True)

    trainloader = DataLoader(db_train, batch_size=30, shuffle=False, num_workers=1, pin_memory=True)

    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
        print(i_batch)
        print("image_batch=",image_batch.shape)
        print("label_batch=",label_batch.shape)
        # print("label_batch=",torch.min(label_batch), torch.max(label_batch))
        print("label_batch=",torch.unique(label_batch))
        print("image_batch=",torch.unique(image_batch))
        break