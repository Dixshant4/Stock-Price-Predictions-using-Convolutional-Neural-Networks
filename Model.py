import torch
import torch.nn as nn




# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self, in1=4, out1=64, out2=128, out3=256, out4=512, fcb1=25088, fcb2=2048, fcb3=100, fcb4=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in1, out_channels=out1, kernel_size=7, stride=1, padding=4)
        self.conv2 = nn.Conv2d(in_channels=out1, out_channels=out1, kernel_size=7, stride=1, padding=4)
        self.conv3 = nn.Conv2d(in_channels=out1, out_channels=out2, kernel_size=7, stride=1, padding=4)
        self.conv4 = nn.Conv2d(in_channels=out2, out_channels=out2, kernel_size=7, stride=1, padding=4)
        self.conv5 = nn.Conv2d(in_channels=out2, out_channels=out3, kernel_size=7, stride=1, padding=4)
        self.conv6 = nn.Conv2d(in_channels=out3, out_channels=out3, kernel_size=7, stride=1, padding=4)
        self.conv7 = nn.Conv2d(in_channels=out3, out_channels=out4, kernel_size=7, stride=1, padding=4)
        self.conv8 = nn.Conv2d(in_channels=out4, out_channels=out4, kernel_size=7, stride=1, padding=4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AdaptiveAvgPool2d((7,7))
        self.fc1 = nn.Linear(fcb1, fcb2)  # Adjusted input size to match the output of conv3
        self.fc2 = nn.Linear(fcb2, fcb3)
        self.fc3 = nn.Linear(fcb3, fcb3)
        self.fc4 = nn.Linear(fcb3, fcb4) # Changed to 1 as per lab03 guidelines. (prev: 2 classes for classification)
        self.dropout = nn.Dropout(0.5, inplace=False)

    def forward(self, x):
        conv1_out = torch.relu(self.conv1(x))
        conv2_out = self.pool(torch.relu(self.conv2(conv1_out)))
        conv3_out = torch.relu(self.conv3(conv2_out))
        conv4_out = self.pool(torch.relu(self.conv4(conv3_out)))
        conv5_out = torch.relu(self.conv5(conv4_out))
        conv6_out = torch.relu(self.conv6(conv5_out))
        conv7_out = self.pool(torch.relu(self.conv7(conv6_out)))
        conv8_out = self.avg(torch.relu(self.conv8(conv7_out)))
        conv8_out = conv8_out.reshape(conv8_out.size(0), -1)  # Flatten the output from convolutional layers
        x = torch.relu(self.fc1(conv8_out))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        out = self.fc4(x)
        return out
