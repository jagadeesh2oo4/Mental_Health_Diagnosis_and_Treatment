import torch
from IPython.core.pylabtools import figsize
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def load_data(data_dir, batch_size=32, augment=False):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if augment:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


    # Load the dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader, dataset.classes

def visualize_data(data_loader, classes, num_samples=5):
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    # Reverse normalization for display purposes
    images = images.numpy().transpose(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    images = (images * 0.5) + 0.5  # De-normalize to [0, 1]

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axes[i].imshow(np.clip(images[i], 0, 1))
        axes[i].axis('off')
        axes[i].set_title(f"Label: {classes[labels[i].item()]}")
    plt.show()


