
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
import torch
import torchvision
import json

from utils import visualizeWeights

from sklearn.model_selection import train_test_split

# parameters
BATCH_SIZE = 50
NUM_EPOCHS = 30
NUM_STEPS = 1
LEARNING_RATE = 0.1
NUM_CLASSES = 10
LIMIT = 1000

# class  definition


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


# device selection
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

# splitting training set into training and validation set
trainset, valset = train_test_split(trainset, test_size=0.1, random_state=42)
testset = torchvision.datasets.CIFAR10(
    './data', train=False, transform=transform)


classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# initializing lists
test_acc_curve = []


ctr = 0

# dataloaders for training, validation and testing
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=BATCH_SIZE, shuffle=False)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False)
model = cnn_3(10)
model = model.to(device)
model_name = model.__class__.__name__
# optimizer
opt_name = "SGD"
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
# loss function
criterion = torch.nn.CrossEntropyLoss().to(device)
# epoch loop
for epoch in range(NUM_EPOCHS):
    print(
        f'Epoch {epoch+1}/{NUM_EPOCHS}  lr = {LEARNING_RATE} optimizer = {opt_name} for {model_name} model')
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

        if i % 10 == 9:
            # learning rate update
            if (ctr == 990):
                LEARNING_RATE = 0.01
                print(f'lr updated to {LEARNING_RATE}')
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=LEARNING_RATE)
            ctr += 1

# testing
with torch.no_grad():
    model.eval()
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False)
    n_correct = 0
    n_samples = 0
    # batch loop
    for test_images, test_labels in testloader:
        test_images, test_labels = test_images.to(
            device), test_labels.to(device)

        test_outputs = model(test_images)

        _, test_predicted = test_outputs.max(1)
        n_samples += test_labels.size(0)
        n_correct += (test_predicted == test_labels).sum().item()

    test_acc = 100.0 * n_correct / n_samples

test_acc_curve.append(test_acc)

# saving model
model_result = {
    'name': model_name,
    'loss_curve_1': test_acc_curve[0],
    'loss_curve_01': test_acc_curve[0],
    'loss_curve_001': test_acc_curve[0],
    'val_acc_curve_1': test_acc_curve[0],
    'val_acc_curve_01': test_acc_curve[0],
    'val_acc_curve_001': test_acc_curve[0]
}
# saving model results in json file
with open("Q5_JSON/Q5_"+"learning4"+".json", "w") as outfile:
    json.dump(model_result, outfile)

print("Training Done")
