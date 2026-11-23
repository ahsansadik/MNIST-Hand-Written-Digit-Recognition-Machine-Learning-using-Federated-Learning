import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from step2_split import get_split_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DigitNet(nn.Module):
    """Lightweight CNN that takes a 1x28x28 image and outputs scores for digits 0-9."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def main(save_path="sub3_model.pt", epochs=3, batch_size=64, lr=0.01):
    _, _, client_subsets = get_split_datasets()
    subset = client_subsets["Sub-system 03"]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    model = DigitNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining Sub-system 03 on {len(subset)} local samples using {device}...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        train_acc = 100.0 * correct / total
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Local Train Acc: {train_acc:.2f}%")

    torch.save(model.state_dict(), save_path)
    print(f"Saved Sub-system 03 weights to {save_path}")


if __name__ == "__main__":
    main()