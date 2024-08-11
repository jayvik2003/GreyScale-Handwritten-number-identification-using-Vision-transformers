import torch
import torch.nn as nn # Correctly import torch.nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model.Vit_former import VisionTransformer
from model.Cnn import CNN

# Training and Testing Functions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

#model = VisionTransformer(img_size=32, patch_size=4, num_classes=10, embed_dim=32, depth=2, num_heads=8, mlp_dim=64).to(device)
model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Load dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)), # These normalization values are for grayscale images
    # Add a channel dimension for grayscale images
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  
])

if isinstance(model, VisionTransformer):
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
else:
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

train_losses = []
test_losses = []
test_accuracies = []

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            train_losses.append(loss.item())

def test():
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    accuracy = 100. * correct / len(test_loader.dataset)
    test_accuracies.append(accuracy)

    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')

# Training loop
for epoch in range(1, 2):
    train(epoch)
    test()

# Plot training loss
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss over Batches')
plt.legend()
plt.show()

# Plot test loss
plt.figure()
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Loss over Epochs')
plt.legend()
plt.show()

# Plot test accuracy
plt.figure()
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.show()

# Visualize some test images along with predicted and true labels
def visualize_predictions():
    model.eval()
    with torch.no_grad():
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        
        # Move data and predictions back to CPU for visualization
        data, target, pred = data.cpu(), target.cpu(), pred.cpu()
        
        plt.figure(figsize=(10, 10))
        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.imshow(data[i].permute(1, 2, 0).squeeze(), cmap='gray')
            plt.title(f'True: {target[i].item()}, Pred: {pred[i].item()}')
            plt.axis('off')
        plt.show()

visualize_predictions()