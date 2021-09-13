import torch
from torchvision.io import read_image
import os,sys
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406] #IMAGENET
IMAGENET_STD = [0.229, 0.224, 0.225] #IMAGENET


class DataGenerator:
    def __init__(self, df, data_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        img_size = [244,244]
        image_transformation_1 = [
            T.Resize(img_size)]

        self.img_labels = df
        self.img_dir = data_dir

        self.agu_label = torch.FloatTensor().to(self.device)
        self.transform_1 =  T.Compose(image_transformation_1)
        self.agu_img = torch.FloatTensor().to(self.device)
        self.agu_img = []
        self.agu_label = []
    
    def __len__(self):
        return len(self.img_labels)

    def __call__(self):
        image = torch.FloatTensor().to(self.device)
        for idx in range(len(self.img_labels)):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
            image = read_image(img_path)
            label = self.img_labels.iloc[idx, 1]

            # image = self.transform_1(image)
            self.agu_img.append(image)
            self.agu_label.append(label)

        return self.agu_img, self.agu_label

if __name__ == '__main__':
    test = DataGenerator()
