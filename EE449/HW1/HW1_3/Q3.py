
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
import torch
import torchvision
import json
from utils import visualizeWeights, part3Plots
import json
from sklearn.model_selection import train_test_split

# Parameter Setting
BATCH_SIZE = 50
NUM_EPOCHS = 15
NUM_STEPS = 10
LEARNING_RATE = 0.01
NUM_CLASSES = 10

# Class Definitions


class mlp_1(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(mlp_1, self).__init__()
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


class mlp_2(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(mlp_2, self).__init__()
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


class cnn_4(torch.nn.Module):
    def __init__(self, num_classes):
        super(cnn_4, self).__init__()
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


class cnn_5(torch.nn.Module):
    def __init__(self, num_classes):
        super(cnn_5, self).__init__()
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


# device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{device} is available")
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# hyperparameters
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    torchvision.transforms.Grayscale()
])

# training set
trainset = torchvision.datasets.CIFAR10(
    './data', train=True, download=True, transform=transform)

# train set and validation set
trainset, valset = train_test_split(trainset, test_size=0.1, random_state=42)
testset = torchvision.datasets.CIFAR10(
    './data', train=False, transform=transform)

# data loader for training set, validation set and test set
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=BATCH_SIZE, shuffle=False)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False)

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
plots = []

# model loop
for ctr in range(5):
    model_train_loss = []
    model_train_acc = []
    model_val_acc = []
    best_weight = 0
    best_acc = 0

    # step loop
    for step in range(NUM_STEPS):
        if ctr == 0:
            model = mlp_1(1024, 10)
        elif ctr == 1:
            model = mlp_2(1024, 10)
        elif ctr == 2:
            model = cnn_3(10)
        elif ctr == 3:
            model = cnn_4(10)
        elif ctr == 4:
            model = cnn_5(10)

        model = model.to(device)
        model_name = model.__class__.__name__

        # loss function and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        # step lists
        step_train_loss = []
        step_train_acc = []
        step_val_acc = []
        step_test_acc = []

        # epoch loop
        for epoch in range(NUM_EPOCHS):
            print(
                f'Epoch {epoch+1}/{NUM_EPOCHS}  Step {step+1}/{NUM_STEPS} for {model_name} model')

            # training loader
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=BATCH_SIZE, shuffle=True)

            # training loop
            for i, tr_data in enumerate(trainloader, 0):
                model.train()
                tr_inp, tr_lab = tr_data[0].to(device), tr_data[1].to(device)

                # forward + backward + optimize
                tr_out = model(tr_inp)
                tr_loss = criterion(tr_out, tr_lab)
                optimizer.zero_grad()
                tr_loss.backward()
                optimizer.step()

                # print statistics each 10 steps
                if i % 10 == 9:
                    model.eval()
                    _, tr_pred = tr_out.max(1)
                    n_samples = tr_lab.size(0)
                    n_correct = (tr_pred == tr_lab).sum().item()
                    # training accuracy
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
                    # validation accuracy
                    val_acc = 100.0 * val_correct / val_total
                    # record statistics
                    step_train_loss.append(train_loss)
                    step_train_acc.append(training_acc)
                    step_val_acc.append(val_acc)

        # record statistics
        model_train_loss.append(step_train_loss)
        model_train_acc.append(step_train_acc)
        model_val_acc.append(step_val_acc)

        # test loop
        with torch.no_grad():
            model.eval()
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=BATCH_SIZE, shuffle=False)
            n_correct = 0
            n_samples = 0

            for test_images, test_labels in testloader:
                test_images, test_labels = test_images.to(
                    device), test_labels.to(device)

                test_outputs = model(test_images)

                _, test_predicted = test_outputs.max(1)
                n_samples += test_labels.size(0)
                n_correct += (test_predicted == test_labels).sum().item()

            # test accuracy
            test_acc = 100.0 * n_correct / n_samples
            step_test_acc.append(test_acc)

            # save best model
            if (test_acc > best_acc):
                best_acc = test_acc
                model.to('cpu')
                if (model.__class__.__name__ == 'mlp_1' or model.__class__.__name__ == 'mlp_2'):
                    best_weight = model.fc1.weight.data.numpy()
                else:
                    best_weight = model.conv1.weight.data.numpy()
                model.to(device)

    # average statistics
    avg_train_loss = [sum(x)/len(x) for x in zip(*model_train_loss)]
    avg_train_acc = [sum(x)/len(x) for x in zip(*model_train_acc)]
    avg_valid_acc = [sum(x)/len(x) for x in zip(*model_val_acc)]
    model_result = {
        'name': model_name,
        'loss_curve': avg_train_loss,
        'train_acc_curve': avg_train_acc,
        'val_acc_curve': avg_valid_acc,
        'test_acc': best_acc,
        'weights': best_weight.tolist(),
    }

    # save model results
    with open("Q3_JSON/Q3_"+model_name+".json", "w") as outfile:
        json.dump(model_result, outfile)

    # save model weights
    visualizeWeights(best_weight, save_dir='Q3_Images',
                     filename='input_weights_'+model_name)

    # save plots
    plots.append(json.load(morel_result))
    part3Plots(plots, save_dir=r'Q3_Images', filename='part3Plots')
print("Training Done")
