import torch
import torch.nn as nn
import torchvision.models as models

def load_model(model_path, num_classes=38, device="cpu"):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
