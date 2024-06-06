import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import tqdm


from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
import torch.utils.data as data

def pytorch_data_loader(path, batch_size, augment:bool = False):
    if augment:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomRotation((90, 90)),
                    transforms.RandomRotation((180, 180)),
                    transforms.RandomRotation((270, 270))
                ])
            ], p=0.75),
            transforms.ToTensor(),
        ])
        ds = datasets.ImageFolder('data/train/', transform=transform)
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        ds = datasets.ImageFolder('data/train/', transform=transform)
        loader = data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
        
        return loader


def save_model(path, epoch, model, optimizer):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()

    torch.save({
        'model_state': model_state,
        'optimizer_state': optimizer_state,
        'epoch': epoch
    }, path)
    
    print(f"Model saved to {path}")
    return

def load_model(path, model):
    model.eval()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    return model
    
def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

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
        print(f'Validation Loss: {valid_loss:.4f} | Accuracy: {val_accuracy:.4f}')
        return predictions
    
def show_images(images, num_images=3):
    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
    for i, image in enumerate(images):
        img = image.numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1) 
        axs[i].imshow(img)
        axs[i].axis('on') 
    plt.show()
