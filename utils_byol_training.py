# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms

from torchvision import models
from torchvision.datasets import ImageFolder, DatasetFolder
import torchvision.datasets as Datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn, optim
from collections import OrderedDict

import os
import copy

# +
import numpy as np
import matplotlib.pyplot as plt


from torch import nn
from torch.nn import functional as F

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1).mean()
        return b


# -

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class DGImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,domain, root, transform=None, target_transform=None,
                 loader=Datasets.folder.default_loader, is_valid_file=None):
        super(DGImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.domain = domain
    
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, self.domain, target


def plotting(losses, accuracies, used_model, save_dir, is_pretrained, try_check):
    plt.figure(figsize=(8,2))
    plt.subplots_adjust(wspace=0.2)

    plt.subplot(1,2,1)
    plt.title("$Loss$",fontsize = 18)
    plt.plot(losses)
    plt.grid()
    plt.xlabel("$epochs$", fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)

    plt.subplot(1,2,2)
    plt.title("$Accuracy$", fontsize = 18)
    plt.plot(accuracies)
    plt.grid()
    plt.xlabel("$epochs$", fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    
    if is_pretrained:
        plt.savefig(save_dir+'transfer_training_{}_{}.png'.format(used_model,str(try_check).zfill(2)), dpi=300)
    else:
        plt.savefig(save_dir+'scratch_training_{}_{}.png'.format(used_model,str(try_check).zfill(2)), dpi=300)
    plt.show()


def save_model(model, used_model, save_dir, is_pretrained, try_check):
    if is_pretrained:
        torch.save(model.state_dict(), 
                   save_dir+'transfer_{}_{}.pth'.format(used_model,str(try_check).zfill(2)))
    else:
        torch.save(model.state_dict(),
                   save_dir+'scratch_{}_{}.pth'.format(used_model,str(try_check).zfill(2)))


def classic_training(device, epochs, model,optimizer, criterion, train_loader, val_loader,lr_scheduler):
    
    val_losses = list()
    val_accuracies = list()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc=0.0
    avg_loss_val=0
    avg_acc_val=0
    
    val_set_size=len(val_loader.dataset)
    val_batches=len(val_loader)  
    

    for epoch in range(epochs[0]):
        epoch_loss = 0
        epoch_accuracy = 0
        batch = 0
        
        loss_val=0
        acc_val=0
        
        model.train(True)
        
        for idx, (x, y) in enumerate(train_loader):
            x = Variable(x.to(device))
            y = Variable(y.to(device))
            
            
            optimizer.zero_grad()
            
            output = model(x)
            loss = criterion(output,y)
            
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(output, 1)
            #preds = torch.argmax(output, dim=1)
            accuracy = torch.sum(preds == y.data)

            epoch_accuracy += accuracy.item()
            epoch_loss += loss.item()

            batch += len(x)
            print('Epoch: {} [{}/{} ({:.0f}%)],\tAccuracy: {:.1f}%,  \t Loss: {:.6f}'.format(
                epoch+1, batch, len(train_loader.dataset),100. * (idx+1) / len(train_loader), 100.*(accuracy.item()/len(x)), loss.item()))

        lr_scheduler.step()
#         torch.cuda.empty_cache()
         
        #validation
        model.train(False)
        model.eval()
        
        for i, data in enumerate(val_loader):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
                
            inputs, labels = data
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
    
            loss_val += loss.data
            acc_val += torch.sum(preds == labels.data)

        avg_loss_val = loss_val / val_set_size
        avg_acc_val = acc_val.item() / val_set_size
        
        val_losses.append(avg_loss_val)       
        val_accuracies.append(avg_acc_val)
        
        print("loss_val:",loss_val.item(),val_set_size)
        print("acc_val:",acc_val.item(),val_set_size)
        
        print("Validation","="*50)
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print()
        
        if avg_acc_val > best_acc:
            print("Best Weights Updated!")
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
            
    print("Best validation acc: {:.4f}".format(best_acc))    
    model.load_state_dict(best_model_wts)
            
    return model, val_losses, val_accuracies


def classic_test(device, model,criterion, test_loader,used_model, save_dir, try_check):
    batch = 0
    test_accuracy = 0
    
    f = open(save_dir+'test_{}_{}_log.txt'.format(used_model,str(try_check).zfill(2)),'w')
    
    #for multi-class accuracy
    y_pred=[]
    y_true=[]

    model.eval()
    for idx, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output,y)

        
        _, preds = torch.max(output.data, 1)
        accuracy = torch.sum(preds == y.data)
        
        #for multi-class accuracy
        y_pred.append(preds)
        y_true.append(y.data)

        test_accuracy += accuracy.item()
        batch += len(x)

        log = 'Test:  [{}/{} ({:.0f}%)],\tAccuracy: {:.1f}%,  \t Loss: {:.6f}'.format(
            batch, len(test_loader.dataset), 100.*(idx+1)/len(test_loader), 100.*(test_accuracy/batch), loss.item())
        print(log)
        f.write(log+'\n')
    f.close()
    
    return str(round(100.*(test_accuracy/batch),2)), y_pred, y_true


def di_training(device, epochs, model,optimizer, criterion, 
                train_loader, val_loader,lr_scheduler, 
                is_d_vec=True, gamma_d_loss=0.5, is_dc=True, entropy_weight = 0.0):
    val_losses = list()
    val_accuracies = list()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc=0.0
    avg_loss_val=0
    avg_acc_val=0
    
    val_set_size=len(val_loader.dataset)
    val_batches=len(val_loader)  
    entropy_criterion = HLoss()

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        batch = 0
        
        loss_val=0
        acc_val=0
        
        model.train(True)
        
        for idx, (x, domain, y) in enumerate(train_loader):
            
            x = Variable(x.to(device))
            domain = Variable(domain.to(device))
            y = Variable(y.to(device))
            
            optimizer.zero_grad()
            
            # 2 classification
            if is_dc:
                if is_d_vec:
                    output,domain_output = model(x, domain)
                else:
                    output,domain_output = model(x)
                    
                domain=domain.type(torch.cuda.LongTensor)
                _,domain=torch.max(domain,1)
                
                object_loss = criterion(output,y)
                domain_loss = criterion(domain_output,domain)
                entropy_loss = entropy_criterion(output)
                loss = object_loss + (domain_loss) * gamma_d_loss + entropy_loss*entropy_weight
                
                
            else:
                if is_d_vec:
                    output = model(x, domain)
                else:
                    output = model(x)
                object_loss = criterion(output,y)
                loss = object_loss 
            
            
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(output, 1)
            accuracy = torch.sum(preds == y.data)

            epoch_accuracy += accuracy.item()
            epoch_loss += loss.item()

            batch += len(x)
            print('Epoch: {} [{}/{} ({:.0f}%)],\tAccuracy: {:.1f}%,  \t Loss: {:.6f}'.format(
                epoch+1, batch, len(train_loader.dataset),100. * (idx+1) / len(train_loader), 100.*(accuracy.item()/len(x)), loss.item()))

        lr_scheduler.step()
        torch.cuda.empty_cache()
         
        
        ###################################
        # Validation
        ###################################
        
        model.train(False)
        model.eval()
        
        with torch.no_grad():
            for i, (x, domain, y) in enumerate(val_loader):
                if i % 100 == 0:
                    print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)

                x = Variable(x.to(device))
                domain = Variable(domain.to(device))
                y = Variable(y.to(device))

                optimizer.zero_grad()
                if is_dc:
                    if is_d_vec:
                        output,domain_output = model(x, domain)
                    else:
                        output,domain_output = model(x)

                    domain=domain.type(torch.cuda.LongTensor)
                    _,domain=torch.max(domain,1)

                    object_loss = criterion(output,y)
                    domain_loss = criterion(domain_output,domain)
                    loss = object_loss + (domain_loss) * gamma_d_loss


                else:
                    if is_d_vec:
                        output = model(x, domain)
                    else:
                        output = model(x)
                    object_loss = criterion(output,y)
                    loss = object_loss 

                _, preds = torch.max(output.data, 1)

                loss_val += loss.data
                acc_val += torch.sum(preds == y.data)
        
    #             del x, domain, labels, outputs, preds
    #             torch.cuda.empty_cache()
        
               
        avg_loss_val = loss_val / val_set_size
        avg_acc_val = acc_val.item() / val_set_size
        
        val_losses.append(avg_loss_val)       
        val_accuracies.append(avg_acc_val)
        
        print("loss_val:",loss_val.item(),val_set_size)
        print("acc_val:",acc_val.item(),val_set_size)
        
        print("Validation","="*50)
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print()
        
        if avg_acc_val > best_acc:
            print("Best Weights Updated!")
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
            
    print("Best validation acc: {:.4f}".format(best_acc))    
    model.load_state_dict(best_model_wts)
            
    return model, val_losses, val_accuracies


def cc_training(device, epochs, model,optimizer, criterion, 
                train_loader_stage1,train_loader_stage2, val_loader_stage1, val_loader_stage2,lr_scheduler, 
                is_d_vec=True, gamma_d_loss=0.5, is_dc=True, entropy_weight = 0.0):
    val_losses = list()
    val_accuracies = list()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    avg_loss_val = 0
    avg_acc_val = 0
    
    val_set_size = len(val_loader_stage1.dataset)
    val_batches = len(val_loader_stage1)  

    
    entropy_criterion = HLoss()
    
    stage_epochs={"stage1":epochs[0],"stage2":epochs[1]}

    
    ##########################################################################################
    ######## Stage 1
    
    for epoch in range(0,stage_epochs['stage1']):
        epoch_loss = 0
        epoch_accuracy = 0
        batch = 0
        
        loss_val=0
        acc_val=0
        
        model.train(True)
        
        for idx, (x, domain, y) in enumerate(train_loader_stage1):
            
            x = Variable(x.to(device))
            domain = Variable(domain.to(device))
            y = Variable(y.to(device))
            
            optimizer.zero_grad()
            
            # 2 classification
            if is_dc:
                if is_d_vec:
                    output,domain_output = model(x, domain)
                else:
                    output,domain_output = model(x)
                    
                domain=domain.type(torch.cuda.LongTensor)
                _,domain=torch.max(domain,1)
                
                object_loss = criterion(output,y)
                domain_loss = criterion(domain_output,domain)
                entropy_loss = entropy_criterion(output)
                loss = object_loss + (domain_loss) * gamma_d_loss + entropy_loss*entropy_weight
                
                
            else:
                if is_d_vec:
                    output = model(x, domain)
                else:
                    output = model(x)
                object_loss = criterion(output,y)
                loss = object_loss 
            
            
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(output, 1)
            accuracy = torch.sum(preds == y.data)

            epoch_accuracy += accuracy.item()
            epoch_loss += loss.item()

            batch += len(x)
            print('Epoch: {} [{}/{} ({:.0f}%)],\tAccuracy: {:.1f}%,  \t Loss: {:.6f}'.format(
                epoch+1, batch, len(train_loader_stage1.dataset),100. * (idx+1) / len(train_loader_stage1), 100.*(accuracy.item()/len(x)), loss.item()))

        lr_scheduler.step()
        torch.cuda.empty_cache()
         
        
        ###################################
        # Validation
        ###################################
        
        model.train(False)
        model.eval()
        
        with torch.no_grad():
            for i, (x, domain, y) in enumerate(val_loader_stage1):
                if i % 100 == 0:
                    print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)

                x = Variable(x.to(device))
                domain = Variable(domain.to(device))
                y = Variable(y.to(device))

                optimizer.zero_grad()
                if is_dc:
                    if is_d_vec:
                        output,domain_output = model(x, domain)
                    else:
                        output,domain_output = model(x)

                    domain=domain.type(torch.cuda.LongTensor)
                    _,domain=torch.max(domain,1)

                    object_loss = criterion(output,y)
                    domain_loss = criterion(domain_output,domain)
                    loss = object_loss + (domain_loss) * gamma_d_loss


                else:
                    if is_d_vec:
                        output = model(x, domain)
                    else:
                        output = model(x)
                    object_loss = criterion(output,y)
                    loss = object_loss 

                _, preds = torch.max(output.data, 1)

                loss_val += loss.data
                acc_val += torch.sum(preds == y.data)
        
    #             del x, domain, labels, outputs, preds
    #             torch.cuda.empty_cache()
        
               
        avg_loss_val = loss_val / val_set_size
        avg_acc_val = acc_val.item() / val_set_size
        
        val_losses.append(avg_loss_val)       
        val_accuracies.append(avg_acc_val)
        
        print("loss_val:",loss_val.item(),val_set_size)
        print("acc_val:",acc_val.item(),val_set_size)
        
        print("Validation","="*50)
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        
    
    
    
    
    
    
    ##########################################################################################
    ######## Stage 2
    
    val_set_size = len(val_loader_stage2.dataset)
    val_batches = len(val_loader_stage2)  
    
    for epoch in range(stage_epochs['stage1'],stage_epochs['stage2']):
        epoch_loss = 0
        epoch_accuracy = 0
        batch = 0
        
        loss_val=0
        acc_val=0
        
        model.train(True)
        
        for idx, (x, domain, y) in enumerate(train_loader_stage2):
            
            x = Variable(x.to(device))
            domain = Variable(domain.to(device))
            y = Variable(y.to(device))
            
            optimizer.zero_grad()
            
            # 2 classification
            if is_dc:
                if is_d_vec:
                    output,domain_output = model(x, domain)
                else:
                    output,domain_output = model(x)
                    
                domain=domain.type(torch.cuda.LongTensor)
                _,domain=torch.max(domain,1)
                
                object_loss = criterion(output,y)
                domain_loss = criterion(domain_output,domain)
                entropy_loss = entropy_criterion(output)
                loss = object_loss + (domain_loss) * gamma_d_loss + entropy_loss*entropy_weight
                
                
            else:
                if is_d_vec:
                    output = model(x, domain)
                else:
                    output = model(x)
                object_loss = criterion(output,y)
                loss = object_loss 
                
                
            #
            
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(output, 1)
            accuracy = torch.sum(preds == y.data)

            epoch_accuracy += accuracy.item()
            epoch_loss += loss.item()

            batch += len(x)
            print('Epoch: {} [{}/{} ({:.0f}%)],\tAccuracy: {:.1f}%,  \t Loss: {:.6f}'.format(
                epoch+1, batch, len(train_loader_stage2.dataset),100. * (idx+1) / len(train_loader_stage2), 100.*(accuracy.item()/len(x)), loss.item()))

        lr_scheduler.step()
        torch.cuda.empty_cache()
         
        
        ###################################
        # Validation
        ###################################
        
        model.train(False)
        model.eval()
        
        with torch.no_grad():
            for i, (x, domain, y) in enumerate(val_loader_stage2):
                if i % 100 == 0:
                    print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)

                x = Variable(x.to(device))
                domain = Variable(domain.to(device))
                y = Variable(y.to(device))

                optimizer.zero_grad()
                if is_dc:
                    if is_d_vec:
                        output,domain_output = model(x, domain)
                    else:
                        output,domain_output = model(x)

                    domain=domain.type(torch.cuda.LongTensor)
                    _,domain=torch.max(domain,1)

                    object_loss = criterion(output,y)
                    domain_loss = criterion(domain_output,domain)
                    loss = object_loss + (domain_loss) * gamma_d_loss


                else:
                    if is_d_vec:
                        output = model(x, domain)
                    else:
                        output = model(x)
                    object_loss = criterion(output,y)
                    loss = object_loss 

                _, preds = torch.max(output.data, 1)

                loss_val += loss.data
                acc_val += torch.sum(preds == y.data)
        
    #             del x, domain, labels, outputs, preds
    #             torch.cuda.empty_cache()
        
               
        avg_loss_val = loss_val / val_set_size
        avg_acc_val = acc_val.item() / val_set_size
        
        val_losses.append(avg_loss_val)       
        val_accuracies.append(avg_acc_val)
        
        print("loss_val:",loss_val.item(),val_set_size)
        print("acc_val:",acc_val.item(),val_set_size)
        
        print("Validation","="*50)
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print()
        
        if avg_acc_val > best_acc:
            print("Best Weights Updated!")
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
    print("Best validation acc: {:.4f}".format(best_acc))    
    model.load_state_dict(best_model_wts)
            
    return model, val_losses, val_accuracies

# +
from itertools import combinations
import random 

def prepare_feature_mix_batch_for_feature_mix(imgs,domains,labels,device):
    
    mix_candidate_pair = []
    mix_pairs_num = 32
    
    for combination in combinations([j for j in range(96)],2):
        _,first_domain_idx = torch.max(domains[combination[0]],0)
        _,second_domain_idx = torch.max(domains[combination[1]],0)
    
        if first_domain_idx.item() != second_domain_idx.item() and labels[combination[0]].item() == labels[combination[1]].item():
    #         print("append")
    #         print(first_domain_idx.item(),second_domain_idx.item())
    #         print("class",labels[combination[0]],labels[combination[1]])
            mix_candidate_pair.append(combination)
    
    
    mix_pairs = random.sample(mix_candidate_pair,min(len(mix_candidate_pair),mix_pairs_num))
    
    mixed_feature_labels_batch = labels[mix_pairs[0][0]].unsqueeze(0)
    mixed_feature_batch = torch.cat((imgs[mix_pairs[0][0]],imgs[mix_pairs[0][1]]),0).unsqueeze(0)
    
    for mix_pair in mix_pairs[1:]:
        mixed_features = torch.cat((imgs[mix_pair[0]],imgs[mix_pair[1]]),0).unsqueeze(0)
        mixed_feature_batch = torch.cat((mixed_feature_batch,mixed_features),dim=0)
        mixed_feature_labels_batch = torch.cat((mixed_feature_labels_batch, labels[mix_pair[0]].unsqueeze(0)), 0)
    
    return mixed_feature_batch, mixed_feature_labels_batch
   
    
def prepare_feature_mix_batch(imgs,domains,labels,device):
    
    mix_candidate_pair = []
    mix_pairs_num = 32
    

    for combination in combinations([j for j in range(96)],2):
        _,first_domain_idx = torch.max(domains[combination[0]],0)
        _,second_domain_idx = torch.max(domains[combination[1]],0)
    
        if first_domain_idx.item() != second_domain_idx.item() and labels[combination[0]].item() == labels[combination[1]].item():
    #         print("append")
    #         print(first_domain_idx.item(),second_domain_idx.item())
    #         print("class",labels[combination[0]],labels[combination[1]])
            mix_candidate_pair.append(combination)
    
    
    mix_pairs = random.sample(mix_candidate_pair,min(len(mix_candidate_pair),mix_pairs_num))
    mixed_feature_labels_batch = labels[mix_pairs[0][0]].unsqueeze(0)
    mixed_feature_batch_1 = imgs[mix_pairs[0][0]].unsqueeze(0)
    mixed_feature_batch_2 = imgs[mix_pairs[0][1]].unsqueeze(0)
    
    
    
    #original
    for mix_pair in mix_pairs[1:]:
        mixed_feature_1=imgs[mix_pair[0]].unsqueeze(0)
        mixed_feature_batch_1=torch.cat((mixed_feature_batch_1,mixed_feature_1),0)
        mixed_feature_2=imgs[mix_pair[1]].unsqueeze(0)
        mixed_feature_batch_2=torch.cat((mixed_feature_batch_2,mixed_feature_2),0)
        mixed_feature_labels_batch = torch.cat((mixed_feature_labels_batch, labels[mix_pair[0]].unsqueeze(0)), 0)
        
#     print(mixed_feature_batch_1.shape, mixed_feature_batch_2.shape)
    return mixed_feature_batch_1,mixed_feature_batch_2, mixed_feature_labels_batch


# -

def fm_training(device, epochs, model,mixer, optimizer, fm_optimizer,
                criterion, train_loader_stage1, train_loader_stage2,  
                val_loader_stage1, val_loader_stage2, lr_scheduler, is_d_vec=True, 
                gamma_d_loss=0.5, is_dc=True, entropy_weight = 0.0):

    val_losses = list()
    val_accuracies = list()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc=0.0
    avg_loss_val=0
    avg_acc_val=0

    val_set_size=len(val_loader_stage1.dataset)
    val_batches=len(val_loader_stage1)  
    entropy_criterion = HLoss()
        
    fm_criterion = nn.L1Loss()
    
    
    stage_epochs={"stage1":epochs[0],"stage1.5":epochs[1],
                  "stage2":epochs[2],"stage3":epochs[3]}
    
    #############################
    ########  stage 1
    #############################
    
    unseen = torch.Tensor([0,0,0,0,1]).type(torch.float32)
    unseen = unseen.unsqueeze(0)
    unseen_domains = unseen.clone()
    for i in range(31):
        unseen_domains = torch.cat((unseen_domains,unseen),0)
    unseen_domains = unseen_domains.to(device)
    
    unseen_domains_96 = unseen.clone()
    for i in range(95):
        unseen_domains_96 = torch.cat((unseen_domains_96,unseen),0)
    unseen_domains_96 = unseen_domains_96.to(device)
    
    unseen_domains_128 = unseen.clone()
    for i in range(127):
        unseen_domains_128 = torch.cat((unseen_domains_128,unseen),0)
    unseen_domains_128 = unseen_domains_128.to(device)
    #####################
    ### Training Loop
    
    for epoch in range(0,stage_epochs['stage1']):
        epoch_loss = 0
        epoch_accuracy = 0
        batch = 0

        loss_val=0
        acc_val=0

        model.train(True)
        for idx, (x, domain, y) in enumerate(train_loader_stage1):

            x = Variable(x.to(device))
            domain = Variable(domain.to(device))
            y = Variable(y.to(device))

            optimizer.zero_grad()
                
            features, d_feat = model.extract_features(x, unseen_domains_96)
            mixed_feature_batch_1,mixed_feature_batch_2, mix_feature_labels=prepare_feature_mix_batch( 
                features, domain, y, device
            )
            d_mixed_feature_batch_1,d_mixed_feature_batch_2, _=prepare_feature_mix_batch( 
                d_feat, domain, y, device
            )
            
            mixed_features=None
            mix_ratio_min=0.9
            #0.9도 범위에 포함시키기 위해 0.9001
            mix_ratio_max=0.9001

            mix_ratio= round(random.uniform(mix_ratio_min,mix_ratio_max),3)
            mixed_features = std_mix(mixed_feature_batch_1, mixed_feature_batch_2, ratio=mix_ratio)
            d_mixed_features = std_mix(d_mixed_feature_batch_1, d_mixed_feature_batch_2, ratio=mix_ratio)
            
            mix_output = model.forward_features(mixed_features)
            d_mix_output = model.domain_classfy(d_mixed_features)
#             domain = seen_domains
            
            # 2 classification
            if is_dc:
                if is_d_vec:
                    output,domain_output = model(x, domain)
                else:
                    output,domain_output = model(x)

#                 domain = seen_domains.long()
                _,domain = torch.max(domain,1)
                
                object_loss = criterion(output,y)
                domain_loss = criterion(domain_output,domain)
                entropy_loss = entropy_criterion(output)
                loss = object_loss + (domain_loss) * gamma_d_loss + entropy_loss*entropy_weight

            else:
                if is_d_vec:
                    output = model(x, domain)
                else:
                    output = model(x)
                object_loss = criterion(output,y)
                loss = object_loss 
            
            _,unseen_domains_labels = torch.max(unseen_domains,1)
            
            mix_object_loss = criterion(mix_output, mix_feature_labels)
            unseen_loss = criterion(d_mix_output, unseen_domains_labels.type(torch.long))
            #remove unseen_loss weight 3.0 
            loss = loss + gamma_d_loss*unseen_loss + 3.0*mix_object_loss 

            loss.backward()
            optimizer.step()


            _, preds = torch.max(mix_output, 1)
            mix_accuracy = torch.sum(preds == mix_feature_labels.data)
            epoch_accuracy += mix_accuracy.item()
           
            _, preds = torch.max(output, 1)
            accuracy = torch.sum(preds == y.data)
            epoch_accuracy += accuracy.item()
            
            epoch_loss += loss.item()

            batch += len(x)
            
            
            ############
            ## domain accuracy
            _, d_preds = torch.max(d_mix_output, 1)
            d_accuracy = torch.sum(d_preds == unseen_domains_labels.data)
            
            _, d_preds = torch.max(domain_output, 1)
            d_accuracy += torch.sum(d_preds == domain.data)
        
            print('Epoch: {} [{}/{} ({:.0f}%)],\tAccuracy: {:.1f}%, \tDomain_Accuracy: {:.1f}%,  \t Loss: {:.6f}'.format(
                epoch+1, batch, len(train_loader_stage1.dataset),100. * (idx+1) / len(train_loader_stage1), 
                100.*((accuracy.item()+mix_accuracy.item())/(len(x)+len(mixed_features))),
                100.*(d_accuracy.item())/(len(x)+len(mixed_features)), loss.item()))
            


        lr_scheduler.step()
        torch.cuda.empty_cache()        
        model.eval()
        
        with torch.no_grad():
            for i, (x, domain, y) in enumerate(val_loader_stage1):
                x = Variable(x.to(device))
                domain = Variable(domain.to(device))
                y = Variable(y.to(device))
                
                
                domain = unseen_domains_128
                
                if is_dc:
                    if is_d_vec:
                        output,domain_output = model(x, domain)
                    else:
                        output,domain_output = model(x)

#                     domain = seen_domains_128.type(torch.cuda.LongTensor)
                    _,domain = torch.max(domain,1)

                    object_loss = criterion(output,y)
                    domain_loss = criterion(domain_output,domain)
                    loss = object_loss + (domain_loss) * gamma_d_loss


                else:
                    if is_d_vec:
                        output = model(x, domain)
                    else:
                        output = model(x)
                    object_loss = criterion(output,y)
                    loss = object_loss 


                _, preds = torch.max(output.data, 1)
                acc_val += torch.sum(preds == y.data)
                loss_val += loss.data
                
        avg_loss_val = loss_val / val_set_size
        avg_acc_val = acc_val.item() / val_set_size

        val_losses.append(avg_loss_val)       
        val_accuracies.append(avg_acc_val)

        print("loss_val:",loss_val.item(),val_set_size)
        print("acc_val:",acc_val.item(),val_set_size)

        print("Validation","="*50)
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        
                
    #############################
    ########  stage 2
    #############################
    
    val_set_size=len(val_loader_stage2.dataset)
    val_batches=len(val_loader_stage2)  
    
    for epoch in range(stage_epochs['stage1'],stage_epochs['stage2']):
        epoch_loss = 0
        epoch_accuracy = 0
        batch = 0

        loss_val=0
        acc_val=0

        model.train(True)
        for idx, (x, domain, y) in enumerate(train_loader_stage2):

            x = Variable(x.to(device))
            domain = Variable(domain.to(device))
            y = Variable(y.to(device))

            optimizer.zero_grad()
                
            features, d_feat = model.extract_features(x, unseen_domains_96)
            mixed_feature_batch_1,mixed_feature_batch_2, mix_feature_labels=prepare_feature_mix_batch( 
                features, domain, y, device
            )
            d_mixed_feature_batch_1,d_mixed_feature_batch_2, _=prepare_feature_mix_batch( 
                d_feat, domain, y, device
            )
            
            mixed_features=None
            mix_ratio_min=0.7
            #0.9도 범위에 포함시키기 위해 0.9001
            mix_ratio_max=0.9001

            mix_ratio= round(random.uniform(mix_ratio_min,mix_ratio_max),3)
            mixed_features = std_mix(mixed_feature_batch_1, mixed_feature_batch_2, ratio=mix_ratio)
            d_mixed_features = std_mix(d_mixed_feature_batch_1, d_mixed_feature_batch_2, ratio=mix_ratio)
            
            mix_output = model.forward_features(mixed_features)
            d_mix_output = model.domain_classfy(d_mixed_features)
#             domain = seen_domains
            
            # 2 classification
            if is_dc:
                if is_d_vec:
                    output,domain_output = model(x, domain)
                else:
                    output,domain_output = model(x)

#                 domain = seen_domains.long()
                _,domain = torch.max(domain,1)
                
                object_loss = criterion(output,y)
                domain_loss = criterion(domain_output,domain)
                entropy_loss = entropy_criterion(output)
                loss = object_loss + (domain_loss) * gamma_d_loss + entropy_loss*entropy_weight

            else:
                if is_d_vec:
                    output = model(x, domain)
                else:
                    output = model(x)
                object_loss = criterion(output,y)
                loss = object_loss 
            
            _,unseen_domains_labels = torch.max(unseen_domains,1)
            
            mix_object_loss = criterion(mix_output, mix_feature_labels)
            unseen_loss = criterion(d_mix_output, unseen_domains_labels.type(torch.long))
            #remove 3.0 
            loss = loss + gamma_d_loss*unseen_loss + 3.0*mix_object_loss 

            loss.backward()
            optimizer.step()


            _, preds = torch.max(mix_output, 1)
            mix_accuracy = torch.sum(preds == mix_feature_labels.data)
            epoch_accuracy += mix_accuracy.item()
           
            _, preds = torch.max(output, 1)
            accuracy = torch.sum(preds == y.data)
            epoch_accuracy += accuracy.item()
            
            epoch_loss += loss.item()

            batch += len(x)
            
            
            ############
            ## domain accuracy
            _, d_preds = torch.max(d_mix_output, 1)
            d_accuracy = torch.sum(d_preds == unseen_domains_labels.data)
            
            _, d_preds = torch.max(domain_output, 1)
            d_accuracy += torch.sum(d_preds == domain.data)
        
            print('Epoch: {} [{}/{} ({:.0f}%)],\tAccuracy: {:.1f}%, \tDomain_Accuracy: {:.1f}%,  \t Loss: {:.6f}'.format(
                epoch+1, batch, len(train_loader_stage2.dataset),100. * (idx+1) / len(train_loader_stage2), 
                100.*((accuracy.item()+mix_accuracy.item())/(len(x)+len(mixed_features))),
                100.*(d_accuracy.item())/(len(x)+len(mixed_features)), loss.item()))
            


        lr_scheduler.step()
        torch.cuda.empty_cache()        
        model.eval()
        
        with torch.no_grad():
            for i, (x, domain, y) in enumerate(val_loader_stage2):
                x = Variable(x.to(device))
                domain = Variable(domain.to(device))
                y = Variable(y.to(device))
                
                
                domain = unseen_domains_128
                
                if is_dc:
                    if is_d_vec:
                        output,domain_output = model(x, domain)
                    else:
                        output,domain_output = model(x)

#                     domain = seen_domains_128.type(torch.cuda.LongTensor)
                    _,domain = torch.max(domain,1)

                    object_loss = criterion(output,y)
                    domain_loss = criterion(domain_output,domain)
                    loss = object_loss + (domain_loss) * gamma_d_loss


                else:
                    if is_d_vec:
                        output = model(x, domain)
                    else:
                        output = model(x)
                    object_loss = criterion(output,y)
                    loss = object_loss 


                _, preds = torch.max(output.data, 1)
                acc_val += torch.sum(preds == y.data)
                loss_val += loss.data
                
        avg_loss_val = loss_val / val_set_size
        avg_acc_val = acc_val.item() / val_set_size

        val_losses.append(avg_loss_val)       
        val_accuracies.append(avg_acc_val)

        print("loss_val:",loss_val.item(),val_set_size)
        print("acc_val:",acc_val.item(),val_set_size)

        print("Validation","="*50)
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        
            
    #############################
    ########  stage 3
    #############################
    
    for epoch in range(stage_epochs['stage2'],stage_epochs['stage3']):
        epoch_loss = 0
        epoch_accuracy = 0
        batch = 0

        loss_val=0
        acc_val=0

        model.train(True)
        for idx, (x, domain, y) in enumerate(train_loader_stage2):
            x = Variable(x.to(device))
            domain = Variable(domain.to(device))
            y = Variable(y.to(device))

            optimizer.zero_grad()
                
            features, d_feat = model.extract_features(x, unseen_domains_96)
            mixed_feature_batch_1,mixed_feature_batch_2, mix_feature_labels=prepare_feature_mix_batch( 
                features, domain, y, device
            )
            d_mixed_feature_batch_1,d_mixed_feature_batch_2, _=prepare_feature_mix_batch( 
                d_feat, domain, y, device
            )
            
            mixed_features=None
            mix_ratio_min=0.5
            #0.9도 범위에 포함시키기 위해 0.9001
            mix_ratio_max=0.9001

            mix_ratio= round(random.uniform(mix_ratio_min,mix_ratio_max),3)
            mixed_features = std_mix(mixed_feature_batch_1, mixed_feature_batch_2, ratio=mix_ratio)
            d_mixed_features = std_mix(d_mixed_feature_batch_1, d_mixed_feature_batch_2, ratio=mix_ratio)
            
            mix_output = model.forward_features(mixed_features)
            d_mix_output = model.domain_classfy(d_mixed_features)
#             domain = seen_domains
            
            # 2 classification
            if is_dc:
                if is_d_vec:
                    output,domain_output = model(x, domain)
                else:
                    output,domain_output = model(x)

#                 domain = seen_domains.long()
                _,domain = torch.max(domain,1)
                
                object_loss = criterion(output,y)
                domain_loss = criterion(domain_output,domain)
                entropy_loss = entropy_criterion(output)
                loss = object_loss + (domain_loss) * gamma_d_loss + entropy_loss*entropy_weight

            else:
                if is_d_vec:
                    output = model(x, domain)
                else:
                    output = model(x)
                object_loss = criterion(output,y)
                loss = object_loss 
            
            _,unseen_domains_labels = torch.max(unseen_domains,1)
            
            mix_object_loss = criterion(mix_output, mix_feature_labels)
            unseen_loss = criterion(d_mix_output, unseen_domains_labels.type(torch.long))
            #remove 3.0 
            loss = loss + gamma_d_loss*unseen_loss + 3.0*mix_object_loss 

            loss.backward()
            optimizer.step()


            _, preds = torch.max(mix_output, 1)
            mix_accuracy = torch.sum(preds == mix_feature_labels.data)
            epoch_accuracy += mix_accuracy.item()
           
            _, preds = torch.max(output, 1)
            accuracy = torch.sum(preds == y.data)
            epoch_accuracy += accuracy.item()
            
            epoch_loss += loss.item()

            batch += len(x)
            
            
            ############
            ## domain accuracy
            _, d_preds = torch.max(d_mix_output, 1)
            d_accuracy = torch.sum(d_preds == unseen_domains_labels.data)
            
            _, d_preds = torch.max(domain_output, 1)
            d_accuracy += torch.sum(d_preds == domain.data)
        
            print('Epoch: {} [{}/{} ({:.0f}%)],\tAccuracy: {:.1f}%, \tDomain_Accuracy: {:.1f}%,  \t Loss: {:.6f}'.format(
                epoch+1, batch, len(train_loader_stage2.dataset),100. * (idx+1) / len(train_loader_stage2), 
                100.*((accuracy.item()+mix_accuracy.item())/(len(x)+len(mixed_features))),
                100.*(d_accuracy.item())/(len(x)+len(mixed_features)), loss.item()))
            


        lr_scheduler.step()
        torch.cuda.empty_cache()        
        model.eval()
        
       
        
        with torch.no_grad():
            for i, (x, domain, y) in enumerate(val_loader_stage2):
                x = Variable(x.to(device))
                domain = Variable(domain.to(device))
                y = Variable(y.to(device))
                
                
                domain = unseen_domains_128
                
                if is_dc:
                    if is_d_vec:
                        output,domain_output = model(x, domain)
                    else:
                        output,domain_output = model(x)

#                     domain = seen_domains_128.type(torch.cuda.LongTensor)
                    _,domain = torch.max(domain,1)

                    object_loss = criterion(output,y)
                    domain_loss = criterion(domain_output,domain)
                    loss = object_loss + (domain_loss) * gamma_d_loss


                else:
                    if is_d_vec:
                        output = model(x, domain)
                    else:
                        output = model(x)
                    object_loss = criterion(output,y)
                    loss = object_loss 


                _, preds = torch.max(output.data, 1)
                acc_val += torch.sum(preds == y.data)
                loss_val += loss.data
                
        avg_loss_val = loss_val / val_set_size
        avg_acc_val = acc_val.item() / val_set_size

        val_losses.append(avg_loss_val)       
        val_accuracies.append(avg_acc_val)

        print("loss_val:",loss_val.item(),val_set_size)
        print("acc_val:",acc_val.item(),val_set_size)

        print("Validation","="*50)
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        
        if avg_acc_val >= best_acc:
            print("Best Weights Updated!")
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
    

    print("Best validation acc: {:.4f}".format(best_acc))    
    model.load_state_dict(best_model_wts)

    return model, val_losses, val_accuracies


def di_test(device, model, criterion, test_loader, used_model, save_dir,try_check, is_d_vec=True, is_dc=True):
    batch = 0
    test_accuracy = 0
    f = open(save_dir+'test_{}_{}_log.txt'.format(used_model,str(try_check).zfill(2)),'w')
    #for multi-class accuracy
    y_pred=[]
    y_true=[]
    
    model.eval()
    
    with torch.no_grad():
        for idx, (x, domain, y) in enumerate(test_loader):
            x = x.to(device)
            domain = Variable(domain.to(device))
            y = y.to(device)

            if is_dc:
                if is_d_vec:
                    output,domain_output = model(x, domain)
                else:
                    output,domain_output = model(x)

                domain=domain.type(torch.cuda.LongTensor)
                _,domain=torch.max(domain,1)

                object_loss = criterion(output,y)
                domain_loss = criterion(domain_output,domain)
                loss = object_loss + (domain_loss) 


            else:
                if is_d_vec:
                    output = model(x, domain)
                else:
                    output = model(x)
                object_loss = criterion(output,y)
                loss = object_loss 


            #_, preds = torch.max(output, 1)
            _, preds = torch.max(output.data, 1)
            #preds = torch.argmax(output, dim=1)        
    #         print("output feature size: ",len(output[0]),preds)
            accuracy = torch.sum(preds == y.data)

            #for multi-class accuracy
            y_pred.append(preds)
            y_true.append(y.data)

            test_accuracy += accuracy.item()
            batch += len(x)

            log = 'Test:  [{}/{} ({:.0f}%)],\tAccuracy: {:.1f}%,  \t Loss: {:.6f}'.format(
                batch, len(test_loader.dataset), 100.*(idx+1)/len(test_loader), 100.*(test_accuracy/batch), loss.item())
            print(log)
            f.write(log+'\n')
        f.close()
    
    
    return str(round(100.*(test_accuracy/batch),2)), y_pred, y_true

def save_route(test_idx, domains, dataset, save_name, used_model):
    dom = ''
    for i in range(4):
        if i==test_idx:
            continue
        dom += domains[i]
        dom += '+'
    save_dir = os.path.join(used_model,dataset)+'/{}/'.format(save_name)+dom[:-1]+'('+domains[test_idx]+')/'
    return save_dir


from sklearn.metrics import classification_report


def getClassificationReport(y_pred, y_true):
    real_y_true=[]
    real_y_pred=[]
    for tmp1 in y_pred:
        for tmp2 in tmp1.tolist():
            real_y_pred.append(tmp2)

    for tmp3 in y_true:
        for tmp4 in tmp3.tolist():
            real_y_true.append(tmp4)    
    
    print(classification_report(real_y_true, real_y_pred, target_names=classes))    


# +
def get_tf(color_jitter=False, augment=True):
    train_tf = 0
    if color_jitter:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.4, .4, .4, .4),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_tf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    if augment==False:
        train_tf = test_tf
    return train_tf, test_tf




# -

# Standard mix
def std_mix(x_1,x_2,ratio=0.5):
    return ratio*x_1 + (1.-ratio)*x_2


#모델 세팅 저장
def save_model_setting(settings,used_model,domains, dataset, save_name):
    text_file_name="model_parameter_setting_info.txt"
    save_dir = os.path.join(used_model,dataset,save_name)
        
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except:
        print('Error : Creating directory. '+ save_dir)
        
    save_dir=os.path.join(save_dir,text_file_name)
    
    with open(save_dir,"w") as f:
        for key,value in settings.items():
            f.write(key+" : "+str(value)+"\n") 

