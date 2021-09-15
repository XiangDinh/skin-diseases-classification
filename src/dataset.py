import torch
import sys
from torch.utils.data import Dataset
import torchvision.transforms as transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class CustomImageDataset(Dataset):
    def __init__(self, image, label, transform=None, target_transform=None,diff=False,train=False):
        
        if diff:
            image_transformation_2 = [
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.ColorJitter(brightness=diff, contrast= diff),
                transforms.RandomAutocontrast(p=0.2),
                transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=(7,9),sigma=(0.1,2))]),p=0.3),
                transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.3)]),p=0.3),
                transforms.RandomRotation(degrees=180),
                transforms.RandomInvert(p=0.2),
                transforms.RandomPosterize(bits=2),
                transforms.RandomAdjustSharpness(sharpness_factor=2)   
            ]

        else:
            image_transformation_2 = []
        
        image_transformation_3 = [transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)]

        image_transformation_2 = transforms.Compose(image_transformation_2)
        image_transformation_3 = transforms.Compose(image_transformation_3)

        if transform:
            self.transform_2 = transform
        else:
            self.transform_2 = image_transformation_2
            self.transform_3 = image_transformation_3
        
        self.train = train
        self.img_data = image
        self.label_data = label
        self.target_transform = target_transform #TODO:

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        
        image = self.img_data[idx]
        label = self.label_data[idx]
        
        if self.train:
            image = self.transform_2(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.transform_3:
            image = self.transform_3(image)

        return image, label
    
    def get_labels(self): return self.label_data

    

    
