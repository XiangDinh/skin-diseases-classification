import torch
import sys,cv2,numpy as np
from torch.utils.data import Dataset
from src.utils import set_seed
import torchvision.transforms as transforms
import albumentations

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class CustomImageDataset(Dataset):
    def __init__(self, image, label, transform=None, target_transform=None,diff=False,train=False,image_size=256):
        set_seed(1)
        if diff:
            image_transformation_2 = [
                        albumentations.Transpose(p=0.5),
                        albumentations.VerticalFlip(p=0.5),
                        albumentations.HorizontalFlip(p=0.5),
                        albumentations.RandomBrightness(limit=0.2, p=0.75),
                        albumentations.RandomContrast(limit=0.2, p=0.75),
                        albumentations.OneOf([
                            albumentations.MotionBlur(blur_limit=5),
                            albumentations.MedianBlur(blur_limit=5),
                            albumentations.GaussianBlur(blur_limit=5),
                            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
                        ], p=0.7),

                        albumentations.OneOf([
                            albumentations.OpticalDistortion(distort_limit=1.0),
                            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
                            albumentations.ElasticTransform(alpha=3),
                        ], p=0.7),

                        albumentations.CLAHE(clip_limit=4.0, p=0.7),
                        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                        albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7), 
            ]

        else:
            image_transformation_2 = []
        
        image_transformation_3 = [albumentations.Normalize()]

        image_transformation_2 = albumentations.Compose(image_transformation_2)
        image_transformation_3 = albumentations.Compose(image_transformation_3)

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
        
        image_path = self.img_data[idx]
        label = self.label_data[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.train:
            transformed = self.transform_2(image=image)
            image = transformed["image"]

        if self.target_transform:
            label = self.target_transform(label)

        if self.transform_3:
            transformed = self.transform_3(image=image)
            image = transformed["image"]
    
        image = image.transpose(2, 0, 1)

        return torch.tensor(image).float(), label
    
    def get_labels(self): return self.label_data

    

    
