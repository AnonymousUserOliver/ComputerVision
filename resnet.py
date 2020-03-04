#!/usr/bin/env python
# coding: utf-8



import os
import json
import time
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.models import resnet18
import torchvision.transforms as transforms





class ImageNet(data.Dataset):
    def __init__(self, path, training):
        self.path = path
#         _training if training arg is set to true
        self.folder_paths = []
        self.imgs = []
        self.lbls = []
        if training:
            self.folder_paths = glob("{}/imagenet_12_train/*/".format(self.path))
        else:
            self.folder_paths = glob("{}/imagenet_12_val/*/".format(self.path))
#     Go into the two folders and get all the image paths. Glob() gets all the paths
        for folder_path in self.folder_paths:
            image_paths = glob("{}/*".format(folder_path))
            self.imgs += image_paths
#         determine labels from path names
        for path in self.imgs:
            if path.find("n02123394") >= 0:
                self.lbls.append(0)
            else:
                self.lbls.append(1)
            


# transform goes into get item
        self.img_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
#             here: add random crop 224*244
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
            
    
    def __getitem__(self, index):
#         laod the img with pillow
        img = Image.open(self.imgs[index]).convert("RGB")
        img = self.img_transforms(img)
        lbl = self.lbls[index]
        return img, lbl
    
    def __len__(self):
        return len(self.imgs)





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pretrained_resnet = resnet18(pretrained=False, progress=True)
#     Create a headless ResNet
        self.resnet = nn.Sequential(*list(self.pretrained_resnet.children())[:-1])
#     Create a linear layer
        self.linear = nn.Linear(512, 2)
    
    def forward(self, x):
    #     Pass input to headless ResNet
    #     size of x is [batch_size, 3, 224, 224]
    #     3 channels, r, g, and b. Their size is 224 by 224
    #     batch_size is the number of images sent each time
        
    #     size of output is [batch_size, 512, 1, 1]
    #     x = 1, y = 1, and there are 512 layers

    #     Resize the output of ResNet so that is can be feed to the linear layer
    
        x = self.resnet(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x





dataset = ImageNet(path="./../shared/imagenet_12", training=True)

training_size = int(0.8 * len(dataset.imgs))
validation_size = int(0.2 * len(dataset.imgs))

torch.manual_seed(0)
training_set, validation_set = data.random_split(dataset, [training_size, validation_size])

test_set = ImageNet(path="./../shared/imagenet_12", training=False)

model = Net().cuda()
loss_fn = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.003)
train_dataloader = data.DataLoader(training_set, shuffle=True, batch_size=16)
validation_dataloader = data.DataLoader(validation_set, shuffle=True, batch_size=16)
test_dataloader = data.DataLoader(test_set, shuffle=True, batch_size=16)





for epoch in range(50):
    
#     Training
    model.train()
    total_training_loss = 0
    correct = 0
    
    for batch_idx, (imgs, lbls) in enumerate(train_dataloader):

        imgs = imgs.cuda()
        lbls = lbls.cuda()
        
        optimizer.zero_grad()
        output = model(imgs)
        loss = loss_fn(output, lbls)
        loss.backward()
        optimizer.step()
        
        total_training_loss += loss.item()
        
        for idx, i in enumerate(output):
            if torch.argmax(i) == lbls[idx]:
                correct+=1
        
        
        print("Epoch Training {}/50 val: {}/{} loss: {:.5f}".format(
        epoch+1,
        batch_idx+1,
        len(train_dataloader),
        loss.item()), end="\r")
    print("Epoch {}/50 Training   acc: {:.5f} total loss: {:.5f}".format(epoch+1, correct/len(training_set), total_training_loss/len(training_set)))
    
#     Validation
    model.eval()
    total_val_loss = 0
    correct = 0
    
    for batch_idx, (imgs, lbls) in enumerate(validation_dataloader):
        imgs = imgs.cuda()
        lbls = lbls.cuda()
        
        output = model(imgs)
        loss = loss_fn(output, lbls)
        
        total_val_loss += loss.item()
        
        for idx, i in enumerate(output):
            if torch.argmax(i) == lbls[idx]:
                correct+=1
        
        print("Epoch Validation {}/50 val: {}/{} loss: {:.5f}".format(
        epoch+1,
        batch_idx+1,
        len(validation_dataloader),
        loss.item()), end="\r")
    print("Epoch {}/50 Validation acc: {:.5f} total loss: {:.5f}".format(epoch+1, correct/len(validation_set), total_val_loss/len(validation_set)))
    
#     Test
    model.eval()
    total_test_loss = 0
    correct = 0

    
    for batch_idx, (imgs, lbls) in enumerate(test_dataloader):
        imgs = imgs.cuda()
        lbls = lbls.cuda()
        
        output = model(imgs)
        loss = loss_fn(output, lbls)

        
        total_test_loss += loss.item()
        
        for idx, i in enumerate(output):
            if torch.argmax(i) == lbls[idx]:
                correct+=1
        
        print("Epoch Test {}/50 val: {}/{} total loss: {:.5f}".format(
        epoch+1,
        batch_idx+1,
        len(test_dataloader),
        loss.item()), end="\r")
        
    print("Epoch {}/50 Test       acc: {:.5f} total loss: {:.5f}".format(
    epoch+1,
    correct/len(test_set),
    loss.item()/len(test_set)))
    print()
    
    


