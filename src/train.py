import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import os
import traceback

class SimpleTimer:
    def __init__(self):
        self.start_time = time.time()
        self.last_tick = self.start_time

    def tick(self):
        now = time.time()
        total_elapsed = now - self.start_time
        delta_elapsed = now - self.last_tick
        self.last_tick = now
        print(f"Total elapsed: {total_elapsed:.2f} s | Since last tick: {delta_elapsed:.2f} s")

def main():
    timer = SimpleTimer()
    # Hyperparameters
    batch_size = 128
    epochs = 3
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    timer.tick()
    print("Loading dataset")

    # Data transforms for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='E:/ml_datasets', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='E:/ml_datasets', train=False, download=True, transform=transform_test)

    # num_workers = min(8, os.cpu_count()/2)  # You can experiment with 4, 8, 12...

    num_workers = 1 # one worker is optimal for Windows because it loads the entire python environment for each worker.
    pin_memory = False # has no impact on performance. True may increase epoch time by 0.1s
    persistent_workers = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    timer.tick()
    print("Creating model")

    # Load ResNet-18 and adjust for CIFAR-10 (10 classes, smaller images)
    model = models.resnet18(weights=None)

    # This is a standard adaptation for CIFAR-10, widely used in literature (e.g., original ResNet paper’s CIFAR experiments). It maintains the 64 output channels, ensuring compatibility with the rest of the architecture.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # This is a common practice for CIFAR-10 with ResNet. The original ResNet paper (He et al., 2016) and many implementations (e.g., PyTorch community models) skip max pooling for small inputs to avoid excessive downsampling.
    model.maxpool = nn.Identity()

    # This is standard and correct for adapting ResNet-18 to CIFAR-10.
    model.fc = nn.Linear(model.fc.in_features, 10)

    model.to(device)

    timer.tick()
    print("Starting training loop")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100. * correct / total

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100. * correct / total
        timer.tick()
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    timer.tick()
    print("Exiting")

if __name__ == '__main__':
    try:
        main()
    except any as e:
        print(e)
        traceback.print_exc()


