import os
import torch
from PIL import Image
from config import config
from torch.utils import data
import numpy as np
from torchvision import transforms as T
class customDataset(data.Dataset):
    def __init__(self, data_path, train = True):
        self.train = train    
        if self.train:
            self.transform = T.Compose([
                
                T.Resize((config.img_width, config.img_height)),
                T.RandomRotation(30),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAffine(45),
                T.ToTensor(),
                T.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
            ])
            folder_list = [data_path + folder for folder in os.listdir(data_path)]
            img_list = [[folder + os.sep + img_name for img_name in os.listdir(folder)] for folder in folder_list]
            list_str = str(img_list)
            list_str = list_str.replace('[', '')
            list_str = list_str.replace(']', '')
            self.img_path_list = list(eval(list_str))
        else:
            self.transform = T.Compose([
                
                T.Resize((config.img_width, config.img_height)),
                T.ToTensor(),
                T.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
            ])
            #self.img_path_list = [data_path + img_name for img_name in os.listdir(data_path)]
            folder_list = [data_path + folder for folder in os.listdir(data_path)]
            img_list = [[folder + os.sep + img_name for img_name in os.listdir(folder)] for folder in folder_list]
            list_str = str(img_list)
            list_str = list_str.replace('[', '')
            list_str = list_str.replace(']', '')
            self.img_path_list = list(eval(list_str))

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = Image.open(img_path)
        img = self.transform(img)
        #img_data = torch.from_numpy(np.asarray(img))
        if self.train:
            label = int(img_path.split('/')[-2])
            return img, label
        else:
            return img, img_path

    def __len__(self):
        return len(self.img_path_list)

def collate_fn(batch):
    imgs = []
    label = []
    for sample in batch:
        #print(sample[0].size(), sample[1])
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), \
           label

if __name__ == '__main__':
    c = customDataset('/dataset/speed_limitation/test/')
    print(c[1][0].size(), len(c))