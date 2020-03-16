import torch
import torch.nn as nn

class LeNet_Enocder(nn.Module):
    def __init__(self):
        super(LeNet_Enocder, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)

        self.flatten = nn.Flatten()
        
        self.relu = nn.LeakyReLU()


    def forward(self , input):
          out = self.conv1(input)
          out = self.relu(out)
          out = self.pool(out)
          out = self.conv2(out)
          out = self.relu(out)
          out = self.pool(out)

          out = self.flatten(out)
          return out

class Discrminator(nn.Module):
    def __init__(self):
        super(Discrminator, self).__init__()
        self.fc1 = nn.Linear(400,500)
        self.fc2 = nn.Linear(500,500)
        self.fc3 = nn.Linear(500,1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self ,input):
        out = self.relu(self.fc1(input))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        return out
        

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(400,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        self.softmax = nn.Softmax()
        self.relu = nn.LeakyReLU()

    def forward(self,input):
        out = self.relu(self.fc1(input))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.softmax(out)
        return out
