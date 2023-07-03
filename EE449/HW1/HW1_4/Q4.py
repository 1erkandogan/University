
from utils import part4Plots
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
import torch
import torchvision
import json

from utils import visualizeWeights, part4Plots

from sklearn.model_selection import train_test_split

# parameters
BATCH_SIZE = 50
NUM_EPOCHS = 15
NUM_STEPS = 1
LEARNING_RATE = 0.01
NUM_CLASSES = 10

# class definitions


class mlp_1_relu(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(mlp_1_relu, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, 32)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class mlp_1_sigmo(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(mlp_1_sigmo, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, 32)
        self.sigmo = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.sigmo(x)
        x = self.fc2(x)
        return x


class mlp_2_relu(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(mlp_2_relu, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, 32)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, 64, bias=False)
        self.fc3 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class mlp_2_sigmo(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(mlp_2_sigmo, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, 32)
        self.sigmo = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(32, 64, bias=False)
        self.fc3 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.sigmo(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class cnn_3_relu(torch.nn.Module):
    def __init__(self, num_classes):
        super(cnn_3_relu, self).__init__()
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


class cnn_3_sigmo(torch.nn.Module):
    def __init__(self, num_classes):
        super(cnn_3_sigmo, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding='valid')
        self.sigmo1 = torch.nn.Sigmoid()
        self.conv2 = torch.nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=5, stride=1, padding='valid')
        self.sigmo2 = torch.nn.Sigmoid()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=7, stride=1, padding='valid')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = torch.nn.Linear(16 * 3 * 3, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmo1(x)
        x = self.conv2(x)
        x = self.sigmo2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = x.view(BATCH_SIZE, 16 * 3 * 3)
        x = self.fc(x)
        return x


class cnn_4_relu(torch.nn.Module):
    def __init__(self, num_classes):
        super(cnn_4_relu, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding='valid')
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=3, stride=1, padding='valid')
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=5, stride=1, padding='valid')
        self.relu3 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = torch.nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=5, stride=1, padding='valid')
        self.relu4 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = torch.nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)
        x = x.view(BATCH_SIZE, 16 * 4 * 4)
        x = self.fc(x)
        return x


class cnn_4_sigmo(torch.nn.Module):
    def __init__(self, num_classes):
        super(cnn_4_sigmo, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding='valid')
        self.sigmo1 = torch.nn.Sigmoid()
        self.conv2 = torch.nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=3, stride=1, padding='valid')
        self.sigmo2 = torch.nn.Sigmoid()
        self.conv3 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=5, stride=1, padding='valid')
        self.sigmo3 = torch.nn.Sigmoid()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = torch.nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=5, stride=1, padding='valid')
        self.sigmo4 = torch.nn.Sigmoid()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = torch.nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmo1(x)
        x = self.conv2(x)
        x = self.sigmo2(x)
        x = self.conv3(x)
        x = self.sigmo3(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.sigmo4(x)
        x = self.maxpool2(x)
        x = x.view(BATCH_SIZE, 16 * 4 * 4)
        x = self.fc(x)
        return x


class cnn_5_relu(torch.nn.Module):
    def __init__(self, num_classes):
        super(cnn_5_relu, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, stride=1, padding='valid')
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding='valid')
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=3, padding='valid')
        self.relu3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding='valid')
        self.relu4 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = torch.nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='valid')
        self.relu5 = torch.nn.ReLU()
        self.conv6 = torch.nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=3, stride=1, padding='valid')
        self.relu6 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = torch.nn.Linear(8 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool1(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.maxpool2(x)
        x = x.view(BATCH_SIZE, 8 * 4 * 4)
        x = self.fc(x)
        return x


class cnn_5_sigmo(torch.nn.Module):
    def __init__(self, num_classes):
        super(cnn_5_sigmo, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, stride=1, padding='valid')
        self.sigmo1 = torch.nn.Sigmoid()
        self.conv2 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding='valid')
        self.sigmo2 = torch.nn.Sigmoid()
        self.conv3 = torch.nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=3, padding='valid')
        self.sigmo3 = torch.nn.Sigmoid()
        self.conv4 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding='valid')
        self.sigmo4 = torch.nn.Sigmoid()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = torch.nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='valid')
        self.sigmo5 = torch.nn.Sigmoid()
        self.conv6 = torch.nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=3, stride=1, padding='valid')
        self.sigmo6 = torch.nn.Sigmoid()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = torch.nn.Linear(8 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmo1(x)
        x = self.conv2(x)
        x = self.sigmo2(x)
        x = self.conv3(x)
        x = self.sigmo3(x)
        x = self.conv4(x)
        x = self.sigmo4(x)
        x = self.maxpool1(x)
        x = self.conv5(x)
        x = self.sigmo5(x)
        x = self.conv6(x)
        x = self.sigmo6(x)
        x = self.maxpool2(x)
        x = x.view(BATCH_SIZE, 8 * 4 * 4)
        x = self.fc(x)
        return x


# device setting
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

# train-val split
trainset, valset = train_test_split(trainset, test_size=0.1, random_state=42)
testset = torchvision.datasets.CIFAR10(
    './data', train=False, transform=transform)

# dataloader for training, test, validation
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=BATCH_SIZE, shuffle=False)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False)

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# data list
relu_loss_curve = []
sigmoid_loss_curve = []
relu_grad_curve = []
sigmoid_grad_curve = []
plots = []
# model looper
for ctr in range(10):
    for step in range(NUM_STEPS):
        if ctr == 0:
            model = mlp_1_relu(1024, 10)
        elif ctr == 1:
            model = mlp_1_sigmo(1024, 10)
        elif ctr == 2:
            model = mlp_2_relu(1024, 10)
        elif ctr == 3:
            model = mlp_2_sigmo(1024, 10)
        elif ctr == 4:
            model = cnn_3_relu(10)
        elif ctr == 5:
            model = cnn_3_sigmo(10)
        elif ctr == 6:
            model = cnn_4_relu(10)
        elif ctr == 7:
            model = cnn_4_sigmo(10)
        elif ctr == 8:
            model = cnn_5_relu(10)
        elif ctr == 9:
            model = cnn_5_sigmo(10)

        model = model.to(device)
        model_name = model.__class__.__name__
        # optimizer and loss function
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        # epoch loop
        for epoch in range(NUM_EPOCHS):
            print(
                f'Epoch {epoch+1}/{NUM_EPOCHS}  Step {step+1}/{NUM_STEPS} for {model_name} model')
            # train loader
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=BATCH_SIZE, shuffle=True)
            # train loop
            for i, tr_data in enumerate(trainloader, 0):
                model.train()
                tr_inp, tr_lab = tr_data[0].to(device), tr_data[1].to(device)

                model.to('cpu')
                # get gradient for mlp and cnn
                if (ctr <= 3):
                    gradA = model.fc1.weight.data.numpy().flatten()
                else:
                    gradA = model.conv1.weight.data.numpy().flatten()
                model.to(device)

                # forward + backward + optimize
                tr_out = model(tr_inp)
                tr_loss = criterion(tr_out, tr_lab)
                optimizer.zero_grad()
                tr_loss.backward()
                optimizer.step()
                # record statistics every 10 steps for gradient and loss
                if i % 10 == 9:
                    model.to('cpu')
                    # record conditions for mlp and cnn
                    if (ctr <= 3):
                        gradB = model.fc1.weight.data.numpy().flatten()
                        grad_magnitude = float(np.linalg.norm(gradA - gradB))
                    else:
                        gradB = model.conv1.weight.data.numpy().flatten()
                        grad_magnitude = float(np.linalg.norm(gradA - gradB))

                    if (ctr == 0 or ctr == 2 or ctr == 4 or ctr == 6 or ctr == 8):
                        relu_grad_curve.append(grad_magnitude)
                        relu_loss_curve.append(tr_loss.item())
                    else:
                        sigmoid_grad_curve.append(grad_magnitude)
                        sigmoid_loss_curve.append(tr_loss.item())

                    model.to(device)
    # only record every 2nd loop since we have 10 models of doubles
    if (ctr % 2 == 1):
        model_result = {
            'name': model_name[0:5],
            'relu_loss_curve': relu_loss_curve,
            'sigmoid_loss_curve': sigmoid_loss_curve,
            'relu_grad_curve': relu_grad_curve,
            'sigmoid_grad_curve': sigmoid_grad_curve,
        }
        # clean up for next double model
        relu_loss_curve = []
        sigmoid_loss_curve = []
        relu_grad_curve = []
        sigmoid_grad_curve = []
        # save json
        with open("Q4_JSON/Q4_"+model_name[0:5]+".json", "w") as outfile:
            json.dump(model_result, outfile)

    # plot
    plots.append(json.load(model_result))
    part4Plots(plots, save_dir=r'Q4_IMAGES', filename=model+"_plot")


print("Training Done")
