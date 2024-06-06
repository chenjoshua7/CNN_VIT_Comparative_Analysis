import numpy as np
from torchvision import datasets, transforms
import torch.utils.data as data
import torch
import tqdm as tqdm

def predict_testing(testing_data, model, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        valid_loss, valid_correct, valid_total = 0.0, 0, 0
        for x, y in tqdm(testing_data):
            x, y = x.to(device), y.to(device)
            outputs = model(x, y)
            valid_output = outputs.logits.argmax(dim=1)
            valid_loss += outputs.loss
            valid_correct += (valid_output == y).sum().item()
            valid_total += y.size(0)
            
            predictions.extend(valid_output)

        val_accuracy = valid_correct / valid_total
        valid_loss /= len(testing_data)  # Normalize validation loss
        print(f'Testing Loss: {valid_loss:.4f} | Accuracy: {val_accuracy:.4f}')
        return predictions

class InvarianceCheckerPytorch:
    def __init__(self) -> None:
        self.augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomChoice([
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270))
            ]),
            transforms.ToTensor(),
        ])
        
        self.normal_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def transform_test_set(self, path: str, batch_size: int = 64):
        test_rotated_ds = datasets.ImageFolder(path, transform=self.rotated_transform)
        test_normal_ds = datasets.ImageFolder(path, transform=self.normal_transform)
        
        test_rotated_loader = data.DataLoader(test_rotated_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        test_normal_loader = data.DataLoader(test_normal_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return test_rotated_ds, test_normal_ds, test_rotated_loader, test_normal_loader
    
    def predict_models(self, model, test_rotated_loader, test_normal_loader):
        rotated_predictions = predict_testing(test_rotated_loader, model = model)
        normal_predictions = predict_testing(test_normal_loader, model = model)
        
        return rotated_predictions, normal_predictions
    
    def calculate_invariance(rotated_predictions, normal_predictions):
        count = 0 
        
        for i, v in zip(rotated_predictions, normal_predictions):
            if i == v:
                count += 1
        return (count)/len(rotated_predictions) 