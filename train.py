import Model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import data_setup
import time
import mnist_model
from torch.utils.data import DataLoader
import os


def accuracy(model, dataset, device):
    """
    Compute the accuracy of `model` over the `dataset`.
    We will take the **most probable class**
    as the class predicted by the model.

    Parameters:
        `model` - A PyTorch MLPModel
        `dataset` - A data structure that acts like a list of 2-tuples of
                  the form (x, t), where `x` is a PyTorch tensor of shape
                  [1, 28, 28] representing an MedMNIST image,
                  and `t` is the corresponding binary target label

    Returns: a floating-point value between 0 and 1.
    """

    correct, total = 0, 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=100)
    for img, t in loader:
        # X = img.reshape(-1, 784)
        img = img.to(device)
        t = t.to(device)
        z = model(img)
        y = torch.sigmoid(z)
        pred = (y >= 0.5).int()

        correct += int(torch.sum(t == pred))
        total   += t.shape[0]
    return correct / total

criterion = nn.BCEWithLogitsLoss()

def train_model(model,                # an instance of MLPModel
                train_data,           # training data
                val_data,             # validation data
                learning_rate=0.005,
                batch_size=100,
                num_epochs=4,
                plot_every=2,        # how often (in # iterations) to track metrics
                plot=True,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):           # whether to plot the training curve
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True) # reshuffle minibatches every epoch
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)

    # these lists will be used to track the training progress
    # and to plot the training curve
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    try:
        for e in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                start = time.time()
                images = images.to(device)
                labels = labels.to(device)

                z = model(images).float()
                loss = criterion(z, labels.float())

                loss.backward() # propagate the gradients
                optimizer.step() # update the parameters
                optimizer.zero_grad() # clean up accumualted gradients


                iter_count += 1
                if iter_count % plot_every == 0:
                    iters.append(iter_count)
                    ta = accuracy(model, train_data, device)
                    va = accuracy(model, val_data, device)
                    train_loss.append(float(loss))
                    train_acc.append(ta)
                    val_acc.append(va)
                    end = time.time()
                    time_taken = round(end - start, 3)
                    print(iter_count, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va, 'Time taken:', time_taken)
    finally:
        if plot:
            plt.figure()
            plt.plot(iters[:len(train_loss)], train_loss)
            plt.title("Loss over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.savefig('training_loss.png')

            plt.figure()
            plt.plot(iters[:len(train_acc)], train_acc)
            plt.plot(iters[:len(val_acc)], val_acc)
            plt.title("Accuracy over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Accuracy")
            plt.legend(["Train", "Validation"])
            plt.savefig('accuracy.png')

train_data = data_setup.train_data
validation_data = data_setup.val_data
test_data = data_setup.test_data

if torch.cuda.is_available():
    device = torch.device("cuda")

print(torch.cuda.is_available())

model = Model.CNN(in1=3, out1=64, out2=128, out3=256, out4=512, fcb1=25088, fcb2=2048, fcb3=100, fcb4=1)
train_model(model, train_data, validation_data, batch_size=100, num_epochs=10)

test_accuracy = accuracy(model, test_data, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print(test_accuracy)

