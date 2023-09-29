import torch

class Accuracy:
    """
    accuracy metric for torch fit model
    """
    def __init__(self, name='accuracy'):
        self.name = name
        
    def __call__(self, outputs, labels):
        _, preds = torch.max(outputs.data, 1)
        correct = (preds == labels).sum().item()
        return correct / preds.size(0)