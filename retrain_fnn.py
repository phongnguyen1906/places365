#import package
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
plt.ion()   # interactive mode
import glob
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm


#create dataset
def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample
data_dir = 'G:/Data_Science_Project/CenterNet_Paper/Raw'
#img_urls = list(sorted(glob.glob(f'{data_dir}/*.npy')))
#npy_loader(img_urls[0])
dataset = datasets.DatasetFolder(
    root=data_dir,
    loader = npy_loader,
    extensions=('.npy',)
)


# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

num_train = len(dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    sampler=valid_sampler, num_workers=num_workers)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
#     num_workers=num_workers)

class_names = dataset.classes
num_class = len(class_names)
# print(num_class)


class MyFNN(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(MyFNN,self).__init__()
        self.D_in = D_in
        self.H1 = H1
        self.H2 = H2
        self.D_out = D_out

        self.layer1 = nn.Linear(self.D_in, self.H1)
        self.layer2 = nn.Linear(self.H1, self.H2)
        self.layer3 = nn.Linear(self.H2, self.D_out)

        self.softmax = nn.Softmax(dim=1)

        self.bn1 = nn.BatchNorm1d(D_in)
        self.bn2 = nn.BatchNorm1d(H1)
        self.bn3 = nn.BatchNorm1d(H2)
        self.bn4 = nn.BatchNorm1d(D_out)

    def forward(self,x):
        # x -> BNorm
        x_bn = self.bn1(x)
        # x -> BNorm -> Linear(1).ELU
        h1_elu = F.elu(self.layer1(x_bn))
        # x -> BNorm -> Linear(1).ELU -> BNorm ->
        h1_elu_bn = self.bn2(h1_elu)
        # x -> BNorm -> Linear(1).ELU -> BNorm -> Linear(2).ELU
        h2_elu = F.elu(self.layer2(h1_elu_bn))
        #  x -> BNorm -> Linear(1).ELU -> BNorm -> Linear(2).ELU  -> BNorm
        h2_elu_bn = self.bn3(h2_elu)
        # x -> BNorm -> Linear(1).ELU -> BNorm -> Linear(2).ELU  -> BNorm -> Linear(3).ELU
        out = F.elu(self.layer3(h2_elu_bn))
        # # x -> BNorm -> Linear(1).ELU -> BNorm -> Linear(2).ELU  -> BNorm -> Linear(3).ELU -> BNorm
        out_bn = self.bn4(out)

        output = self.softmax(out_bn)
        return output

D_in, H1, H2, D_out = 512, 256, 128, num_class
learning_rate = 1e-4
#check if cuda is available:
#train_on_gpu = torch.cuda.is_available()
model = MyFNN(D_in, H1, H2, D_out)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = torch.nn.CrossEntropyLoss()

n_epochs = 100  # you may increase this number to train a final model
best_score = 0.0
best_loss = 1e18
# track change in validation loss

model_dir = 'G:/Data_Science_Project/CenterNet_Paper/models'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def train_model():
    best_score = 0.0
    best_loss = 1e18
    for epoch in tqdm(range(1, n_epochs + 1)):
        # model_index = 0
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            # if train_on_gpu:
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################

        correct = 0
        total = 0

        model.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            # if train_on_gpu:
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        accuracy = correct/total*100

        # calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        # print training/validation statistics
        if epoch %10 == 0:
            print('Epoch: {} \tTraining Loss: {:.2f} \tValidation Loss: {:.2f}'.format(
                epoch, train_loss, valid_loss))
            print('Accuracy on epoch {}: {:.2f}'.format(epoch, accuracy))
        # save model if validation loss has decreased
        if best_loss >=valid_loss and best_score <= accuracy:
            print('Validation loss decreased ({:.2f} --> {:.2f}).  Saving model ...'.format(
                best_loss,
                valid_loss))
            model_file_path = os.path.join(model_dir, 'model_best.pt')
            torch.save(model.state_dict(), model_file_path)
            best_loss = valid_loss
            best_score = accuracy

if __name__ == '__main__':

    train_model()

