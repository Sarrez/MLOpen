import torch
import torch.nn as nn
import torch.nn.functional as fn

class AlexNet(nn.Module):
    def __init__(self, n_classes=10):
        super(AlexNet, self).__init__()
        self.structure = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=4, padding=0 ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features= 9216, out_features= 4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features= 4096, out_features= 4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096 , out_features=n_classes),
        )
    def forward(self, x):
        x = self.structure(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        probs = fn.softmax(x, dim=1)
        return x, probs