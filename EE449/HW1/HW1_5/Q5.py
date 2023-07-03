
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
import torch
import torchvision
import json

from utils import visualizeWeights, part5Plots

from sklearn.model_selection import train_test_split

# parameters
BATCH_SIZE = 50
NUM_EPOCHS = 20
NUM_STEPS = 1
LEARNING_RATE = [0.1, 0.01, 0.001]
NUM_CLASSES = 10

# class definition


class cnn_3(torch.nn.Module):
    def __init__(self, num_classes):
        super(cnn_3, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding='valid')
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=5, stride=1, padding='valid')
        self.relu2 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=7, stride=1, padding='valid')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = torch.nn.Linear(16 * 3 * 3, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = x.view(BATCH_SIZE, 16 * 3 * 3)
        x = self.fc(x)
        return x


# device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{device} is available")
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# transform
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    torchvision.transforms.Grayscale()
])

# training set
trainset = torchvision.datasets.CIFAR10(
    './data', train=True, download=True, transform=transform)

# train set split
trainset, valset = train_test_split(trainset, test_size=0.1, random_state=42)
testset = torchvision.datasets.CIFAR10(
    './data', train=False, transform=transform)

# loader for training, validation and test set
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=BATCH_SIZE, shuffle=False)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False)

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# list initialization
tra_loss_curve = []
val_acc_curve = []

# loop for different learning rates
for ctr in LEARNING_RATE:
    train_loss_lr = []
    val_acc_lr = []
    model = cnn_3(10)

    model = model.to(device)
    model_name = model.__class__.__name__

    # optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=ctr)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # epoch loop
    for epoch in range(NUM_EPOCHS):
        print(
            f'Epoch {epoch+1}/{NUM_EPOCHS}  lr = {ctr} for {model_name} model')
        # train loader
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=BATCH_SIZE, shuffle=True)

        # batch loop
        for i, tr_data in enumerate(trainloader, 0):
            model.train()
            tr_inp, tr_lab = tr_data[0].to(device), tr_data[1].to(device)
            # forward + backward + optimize
            tr_out = model(tr_inp)
            tr_loss = criterion(tr_out, tr_lab)
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

            # record every 10th step
            if i % 10 == 9:
                model.eval()
                _, tr_pred = tr_out.max(1)
                n_samples = tr_lab.size(0)
                n_correct = (tr_pred == tr_lab).sum().item()
                training_acc = 100.0 * n_correct / n_samples
                train_loss = tr_loss.item()
                val_total = 0
                val_correct = 0
                # validation loop
                for j, val_data in enumerate(valloader, 0):
                    val_inp, val_lab = val_data[0].to(
                        device), val_data[1].to(device)
                    val_out = model(val_inp)
                    _, val_pred = val_out.max(1)
                    val_total += val_lab.size(0)
                    val_correct += (val_pred ==
                                    val_lab).sum().item()
                val_acc = 100.0 * val_correct / val_total

                train_loss_lr.append(train_loss)
                val_acc_lr.append(val_acc)

    tra_loss_curve.append(train_loss_lr)
    val_acc_curve.append(val_acc_lr)

# save the results
model_result = {
    'name': model_name,
    'loss_curve_1': tra_loss_curve[0],
    'loss_curve_01': tra_loss_curve[1],
    'loss_curve_001': tra_loss_curve[2],
    'val_acc_curve_1': val_acc_curve[0],
    'val_acc_curve_01': val_acc_curve[1],
    'val_acc_curve_001': val_acc_curve[2]
}

# save the results as json file
with open("Q5_JSON/Q5_"+"learning"+".json", "w") as outfile:
    json.dump(model_result, outfile)

# plot the results
part5Plots(model_result, save_dir=r'Q5_IMAGES', filename="learning")

print("Training Done")
