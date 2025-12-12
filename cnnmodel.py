from preprocessing import load_data, visualize_data
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from classification_model import DeepAnn, CNN


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)

    train_losses = []
    val_losses = []
    val_accuracies = []
    print("training")
    for epoch in range(num_epochs):
        model.train()
        running_loss=0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss=criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss/len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    print("Training completed.")

    return model
def main():
    # Path to the dataset folder (adjust this path as needed)
    data_dir = r"C:\Users\Battula Jagadeesh\OneDrive\Desktop\Capstone_project\Mental_Health\images\train"

    # Load data and visualize
    train_loader, class_names = load_data(data_dir, batch_size=32, augment=True)

    print("Class names:", class_names)
    visualize_data(train_loader, class_names)

    # choose a model
    input_size = 128 * 128 * 3
    num_classes = len(class_names)
    hidden_size = 128

    model = CNN( num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizers = {
        # "SGD": torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8),
        "Adam": torch.optim.Adam(model.parameters(), lr=0.0005),
        # "RMSProp": torch.optim.RMSprop(model.parameters(), lr=0.0007)
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for optimizer_name, optimizer in optimizers.items():
        print(f"Training with optimizer: {optimizer_name}")
        trained_model = train_model(model, train_loader, train_loader, criterion, optimizer, device, num_epochs=5)


    torch.save(trained_model.state_dict(), "cnn_model.pth")
    print("Model saved as 'cnn_model.pth'.")
    predict(trained_model, train_loader, device, class_names)



def predict(model, val_loader, device, class_names):
    model.to(device)
    model.eval()
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    images, labels = images[:5].to(device), labels[:5].to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    images = (images * 0.5) + 0.5

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(np.clip(images[i], 0, 1))
        axes[i].axis('off')
        color = 'green' if predictions[i] == labels[i] else 'red'
        axes[i].set_title(f"Pred: {class_names[predictions[i]]}\nTrue: {class_names[labels[i]]}", color=color)
    plt.show()

if __name__ == "__main__":

    main()