# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 17:38:44 2018

@author: prizo
"""


import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision
import torchvision.models as models
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
from scipy import ndimage


import torchvision.transforms as transforms

from os import listdir
from os.path import isfile, join

from PIL import Image

"""
# #%%
"""
###############################################################################
###############       THIS IS THE DATA LOADER            ######################
###############################################################################
class Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        dirs = []
        start = 0
        mypath = 'data/png'
        X = np.empty((0, 1, 225, 225))
        y = np.empty((0))
        label = 0
        for f in listdir(mypath):
            dirs.append(mypath + '/' + f)
        for folder in dirs:
            print(folder)
            for file in listdir(folder):
                #print(file)
                img = io.imread(folder + '/' + file)
                new_image = Image.fromarray(img)
                new_image = new_image.resize((225, 225), resample=Image.LANCZOS)
                new_image = np.array(new_image, dtype=np.uint8)
                new_image = np.expand_dims(new_image, axis=0)
                new_image = new_image/255
                
                X = np.append(X, [new_image], axis=0)
                y = np.append(y, [label], axis=0)
                
                if start > 5:
                    start = 0
                    break
                start += 1

            if label > 3:
                    break
            label += 1
                    
        #X.shape
        #y.shape
        #x_data.shape
        self.len = X.shape[0]
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(y).float()
        self.transform = transform
        

    def __len__(self):
        
        return self.len

    def __getitem__(self, idx):
        if self.transform:
            #print(self.x_data[idx].shape)
            image = transforms.ToPILImage()(self.x_data[idx])
            #print(image)
            samplex = self.transform(image)
            #print(samplex.shape)
        else:
            samplex = self.x_data[idx]
            
        return samplex, self.y_data[idx]

###############################################################################
###############       THIS IS THE NET DEFINITION         ######################
###############################################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, 15, 3, 0)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(64, 128, 5, 1, 0)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 512, 7, 1, 0)
        self.drop = nn.Dropout(p=0.5)
        self.conv7 = nn.Conv2d(512, 512, 1, 1, 0)
        self.conv8 = nn.Conv2d(512, 250, 1, 1, 0)
        

    def forward(self, x):
        out = F.relu(self.conv1(x))
        #print("conv1",out.shape)
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        #print("conv2",out.shape)
        out = self.pool(out)
        out = F.relu(self.conv3(out))
        #print("conv3",out.shape)
        out = F.relu(self.conv4(out))
        #print("conv4",out.shape)
        out = F.relu(self.conv5(out))
        #print("conv5",out.shape)
        out = self.pool(out)
        out = F.relu(self.conv6(out))
        #print("conv6",out.shape)
        out = self.drop(out)
        out = F.relu(self.conv7(out))
        #print("conv7",out.shape)
        out = self.drop(out)
        out = self.conv8(out)
        #print("conv8",out.shape)
        
        return out
    
 
net = Net()


#transforms.Resize(224)
#transforms.CenterCrop(225)
#dataset = Dataset(transform=transforms.Compose([transforms.CenterCrop(225),transforms.ToTensor()]))
dataset = Dataset()
train_loader = DataLoader(dataset=dataset,batch_size=1,shuffle=True)

"""
for i, data in enumerate(train_loader, 0):
    print(i)
    print(data[0][0,0,:,:])
    imgplot = plt.imshow(data[0][0,0,:,:])
    plt.show()
    scipy.misc.imsave('test_sample.tif', data[0][0,0,:,:])
    break
"""


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
max_epochs = 10
loss_np = np.zeros((max_epochs))
accuracy = np.zeros((max_epochs))

for epoch in range(max_epochs):
    correct = 0
    epoch_loss = 0
    for i, data in enumerate(train_loader, 0):
        # Get inputs and labels from data loader 
        inputs, labels = data
        
        #print(inputs.shape)
        #print(labels)
        labels = labels.type(torch.LongTensor)
        #64x3x32x32 to 64x1x3072
        #inputs.resize_(len(inputs), 1, 3072)
        
        inputs, labels = Variable(inputs), Variable(labels)
        #inputs = torch.squeeze(inputs)
        # Feed the input data into the network 
        y_pred = net(inputs)
        y_pred = torch.squeeze(y_pred)
        #print(y_pred)
        
        
        # Calculate the loss using predicted labels and ground truth labels
        loss = criterion(y_pred, labels)
        
        #print("epoch: ", epoch, "loss: ", loss.data[0])
        
        # zero gradient
        optimizer.zero_grad()
        
        # backpropogates to compute gradient
        loss.backward()
        
        # updates the weghts
        optimizer.step()
        
        # convert predicted laels into numpy
        y_pred_np = y_pred.data.numpy()
        
        # calculate the training accuracy of the current model
        pred_np = np.argmax(y_pred_np,axis=1)
        label_np = labels.data.numpy().reshape(len(labels),1)
        
        for j in range(y_pred_np.shape[0]):
            if pred_np[j] == label_np[j,:]:
                correct += 1
        epoch_loss += loss.data.numpy()
            
        
        #accuracy[epoch] = float(correct)/float(len(label_np))    
        #loss_np[epoch] = loss.data.numpy()
    accuracy[epoch] = float(correct)/float(10000)    
    loss_np[epoch] = epoch_loss/float(len(train_loader))
    
    print("epoch: ", epoch, "loss: ", loss_np[epoch])

#print("finished: ", batch, "and ", learn_rate)
print("final training accuracy: ", accuracy[max_epochs-1])

epoch_number = np.arange(0,max_epochs,1)

# Plot the loss over epoch
plt.figure()
plt.plot(epoch_number, loss_np)
plt.title('loss over epoches')
plt.xlabel('Number of Epoch')
plt.ylabel('Loss')

# Plot the training accuracy over epoch
plt.figure()
plt.plot(epoch_number, accuracy)
plt.title('training accuracy over epoches')
plt.xlabel('Number of Epoch')
plt.ylabel('accuracy')
