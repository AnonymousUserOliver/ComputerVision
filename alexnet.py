import json
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.models import alexnet
import torchvision.transforms as transforms

class ImageNet(data.Dataset):
    def __init__(self, path):
        self.path = path
        self.folder_paths = glob("{}/*/".format(self.path))
        self.json_path = "{}/imagenet_class_index.json".format(self.path)

        with open("{}/imagenet_class_index.json".format(self.path), "r") as f:
            self.lbl_dic = json.load(f)
        self.lbl_dic = {v[0]: int(k) for k, v in self.lbl_dic.items()}

        self.img_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.imgs = []
        self.lbls = []
        for folder_path in self.folder_paths:
            image_paths = glob("{}/*".format(folder_path))
            self.imgs += image_paths
            self.lbls += [self.lbl_dic[folder_path.split("/")[-2]]] * len(image_paths)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        img = self.img_transforms(img)
        lbl = self.lbls[index]
        return img, lbl

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    image_net = ImageNet('./imagenet_12')
    dataloader = data.DataLoader(image_net)
    alex_net = alexnet(pretrained=True)
    alex_net = alex_net.eval()
    
    num_of_correct_prediction = 0
    num_of_tests = 0
    
    for iterator in dataloader: 

        num_of_tests += 1
        image = iterator[0]
        label = iterator[1]

        
        output = alex_net(image)
        predicted_label = torch.argmax(output)
        
        if torch.eq(predicted_label, label):
            num_of_correct_prediction += 1
        
        
    print("The Accurary is: ", num_of_correct_prediction/num_of_tests)

        
        
    
    
    