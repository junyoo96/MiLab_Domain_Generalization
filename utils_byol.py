# -*- coding: utf-8 -*-
import os
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
from torchvision import models, transforms

IMAGE_EXTS = ['.jpg', '.png', '.jpeg']


def expand_greyscale(t):
    return t.expand(3, -1, -1)


class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        
        self.folder = folder
        self.paths = []
        
        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

#         print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)


#train function
def byol_training(device, epochs, model,train_loaders,optimizer):
    
    print("BYOL Training Start!")
    
    batch=0
    train_loaders_len=0
    #train_loaders 전체 개수 계산 
    for _,t in train_loaders.items():
        train_loaders_len+=len(t.dataset)
    train_loaders_len=int(train_loaders_len/2)  
    
    #Epoch 시작 
    for epoch in range(epochs):
        
        batch=0
        train_loader_count=0
        train_loader_same_class=dict()
        
        for key,value in train_loaders.items():
            train_loader_count+=1

            train_loader_same_class[train_loader_count]=value

            if train_loader_count%2==0:
                train_loader_count=0

                train_loader1=iter(train_loader_same_class[1])
                train_loader2=iter(train_loader_same_class[2])
                train_loader_len=len(train_loader1)

                for i in range(train_loader_len):
                        
                    x1=next(train_loader1)
                    x2=next(train_loader2)

                    loss=model(x1,x2)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    model.update_moving_average()
                    
                    batch+=len(x1)
                    
                    print('BYOL Epoch: {} [{}/{} ({:.0f}%)],\t Loss: {:.6f}'.format(
                    epoch+1, batch, train_loaders_len,100. * batch / train_loaders_len,loss.item()))
