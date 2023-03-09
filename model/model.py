import torch
import torch.nn as nn
import torchvision.models as models

class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__()
        
        # Load pre-trained ResNet-50 model as the backbone
        resnet = models.resnet50(pretrained=True)
        
        # Remove the fully connected layer and average pooling layer
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        
        # Add custom regression head layers
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
