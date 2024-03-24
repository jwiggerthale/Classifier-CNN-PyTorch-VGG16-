'''
This code implements training of convolutional neural network for image classification in PyTorch
	--> Images are grayscale (1 channel, 1024 x 2048 pixels)
For modeling, model following VGG16 architectire is used (c.f. https://arxiv.org/pdf/1409.1556.pdf, implemented from scratch in module vgg)
'''

import numpy as np
import os
from PIL import Image
import copy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models
import vgg
import csv

'''
Define transformations for training-, test- and validation images 
Transformations for validation- and test images identical 
Adapt if required
'''
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.09523337495093247],[0.1498455750945764])
    ])

transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.09523337495093247],[0.1498455750945764])    
    ])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.09523337495093247],[0.1498455750945764])
    ])

'''
Get image paths
Make sure, file exists 
Adapt reshaping to your specific file structure
'''
with open('image_paths.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
im_paths  = np.array(data)
im_paths = im_paths[1:, 1]
    
'''
Split paths in those for training, validation and testing
Shuffle previously to ensure balanced distribution
'''
np.random.shuffle(im_paths)
train_paths, rest = np.split(im_paths, [int(0.8*len(im_paths))])
test_paths, val_paths = np.split(rest, [int(0.5*len(rest))])



label_to_num = {'NOK.tiff':0,
                'OK.tiff':1}
num_to_label = {0: 'NOK',
               1: 'OK'}

'''
Class image dataset for loading images 
Create with: 
	File_Name: list of filepaths to images 
	transform: Instance of transforms.Compose (optional)
Methods: 
	getitem --> Returns image and corresponding label for index
		--> If transforms is not None, image is transformed by transforms
	len --> Returns number of image paths in filepaths
'''
class ImageDataset(Dataset):
    def __init__(self, File_Name, transform=False):
        self.image_paths = File_Name
        self.transform = transform

    def __len__(self):
        return(len(self.image_paths))


    def __getitem__(self, idx):
        im_path = self.image_paths[idx]
        #im_path = im_path.replace('\\', '/')
        #im_path = im_path.replace('tiff', 'csv')
        #im_path = im_path.replace('2023APIS2', 'Ims_APIS')
        label = im_path.split('_')[-1]
        label = label_to_num[label]

        im = Image.open(im_path)
        im = np.array(im)
        if(self.transform is not None):
            im = self.transform(im)#['Image']
        return im, label

'''
Create instances of ImageDataset for training-, test- and validation data
Put them in a DataLoader
'''
train_data = ImageDataset(train_paths, transform_train)
test_data = ImageDataset(test_paths, transform_test)
val_data = ImageDataset(val_paths, transform_val)

train_loader = DataLoader(train_data, batch_size =8, shuffle = True, num_workers = 0, drop_last = True)
test_loader = DataLoader(test_data, batch_size = 8, shuffle = False, num_workers = 0, drop_last = True)
val_loader = DataLoader(val_data, batch_size=8, shuffle = True, num_workers=0, drop_last = True)

'''
Function validation
Use for validation of model's performance (i.e. after every epoch of training)
Call with: 
	model: Model you are currently training
	val_loader: Data loader for validation images
	criterion: Loss function you apply
 	device: Device you use to train your model
  		--> Make sure, device is avaialbe on your hardware
Returns: 
	Average loss on validation images
	Accuracy on validation images
'''
def validation(model: nn.Module, 
	       val_loader: DataLoader, 
	       criterion: nn.Module,
		device: str = 'cpu'):
    model.eval()
    val_loss = 0
    acc = 0
    for ims, labels in iter(val_loader):
        ims = ims.to(device)
        labels = labels.to(device)

        output = model.forward(ims)
        val_loss += criterion(output, labels).item()
        probs = torch.exp(output)

        equality = labels.data == probs.max(dim=1)[1]
        acc+= equality.type(torch.FloatTensor).mean()
    val_loss /= len(val_loader)
    acc /= len(val_loader)
    return val_loss, acc


'''
Function for training model 
Conducts training and validation of model iteratively 
Call with: 
	model: Model you want to train 
	optimizer: Optimizer you apply
	loss_function: loss function you apply
	epoch: Number of epochs for training
	threshold_early_stopping: Number of epochs for early stopping when model does not improve
	device: Device you want to use for training 
		--> Make sure, device exists on your hardware
After every epoch: 
	Model is saved in folder 'results'
	Results are written to console
	Results are written to 'results/results.txt'
		--> Make sure, 'results/results.txt' exists
Returns: 
	  list of accuracy on validation images for every epoch
   	  list of validation losses for every epoch
	  list of training losses for every epoch
	  trained model (after last epoch)
	  file name for best model trained
'''
def train_classifier(model: nn.Module, 
		     optimizer, 
		     loss_function: nn.Module, 
		     epochs: int = 200, 
		     threshold_early_stopping: int = 50, 
		     device: str = 'cpu'):
      model = model
      optimizer = optimizer
      criterion = loss_function
      accs = []
      losses = []
      train_losses = []
      epochs = epochs
      best_epoch = 0
      best_acc = -1

      model.to(device)

      for epoch in range(epochs):
        model.train()
        running_loss = 0
        print('\n\nepoch: '+ str(epoch+1))
	j = 0
        for ims, labels in iter(train_loader):
            if(j/20 ==0):
		print(f'epoch {epoch} / {epochs}, batch {j} / {len(train_loader)}')
            ims = ims.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            output = model.forward(ims)
            print(output.shape)
            print(labels.shape)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
	#Set model in evaluation mode and call validation 
        model.eval()
        val_loss, acc = validation(model, val_loader, criterion)
        running_loss /= len(train_loader)
        train_losses.append(running_loss)
        losses.append(val_loss)
        accs.append(acc)
        print('Epoch: {}/{}.. '.format(epoch+1, epochs),
            'training_loss: {:.3f}..'.format(running_loss),
            'validation loss: {:.3f}..'.format(val_loss),
            'accuracy: {:.3f}'.format(acc))
        f = open('results/results.txt', 'a')
        f.write(f'Epoch: {epoch}/{epochs}. \ntraining_loss: {running_loss}.\nvalidation loss: {val_loss} \naccuracy: {acc}')

        file_name = 'results/vgg_cnn_epoch_{}_acc_{}.pth'.format(epoch, int(acc*100))
        torch.save(model.state_dict(), file_name)
        if(acc>best_acc):
            best_acc = acc
            best_epoch = epoch
            best_model = file_name
        elif(epoch-best_epoch > threshold_early_stopping):
            print('Training stopped')
            break
            
        running_loss = 0
        model.train()
      return(accs, losses, train_losses,  model, best_model)


'''
Define parameters for model and training here
'''
num_epochs = 100
num_classes = 2
device ='cpu'
model = vgg.VGG16_GS(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

accs, losses, train_losses, model, best_model =  train_classifier(model, optimizer, criterion, num_epochs, 30, device)







