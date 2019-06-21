from torch import nn
from torchvision import models
#from config import config
import  torch.nn.functional as F
from model.vgg11 import *

def generate_model(num_classes):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, stride = 2)
            self.conv2 = nn.Conv2d(16, 32, 3, stride = 2)
            self.conv3 = nn.Conv2d(32, 16, 3, stride = 2)
            self.fc = nn.Linear(7 * 7 * 16, num_classes)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    net = Net()
    return net

def get_net(num_classes, model_name = None):
    #model = eval('models.' + config.model_name + '(pretrained = True)')
    #model.fc = nn.Linear(512, config.num_classes)
    if model_name == None:
        model = generate_model(num_classes)
    else:
        model = eval(model_name + '(%d)' % num_classes)
    return model
