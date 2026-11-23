import torch
from torch.utils.data import DataLoader, Subset
from step2_split import get_split_datasets
from step7_train_sub3 import DigitNet, device


def evaluate_subset(model, test_data, digits, pos_digits, label_name):
    """Evaluate exact accuracy and how often predictions fall into Sub-system 03's trained digits."""
    indices = [
        idx for idx, label in enumerate(test_data.targets) if int(label) in digits
    ]
    subset = Subset(test_data, indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    predicted_as_pos = 0
    pos_tensor = torch.tensor(pos_digits, device=device)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            predicted_as_pos += torch.isin(preds, pos_tensor).sum().item()

    exact_acc = 100.0 * correct / total if total > 0 else 0.0
    pos_rate = 100.0 * predicted_as_pos / total if total > 0 else 0.0

    print(
        f"{label_name} (digits {digits}, n={total}): "
        f"Exact Match Acc = {exact_acc:.2f}% | "
        f"Predicted as {pos_digits} = {pos_rate:.2f}%"
    )


def main(model_path="sub3_model.pt"):
    _, test_data, _ = get_split_datasets()

    model = DigitNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    pos_digits = [7, 8, 9]
    neg_digits = [0, 1, 2, 3, 4, 5, 6]

    print("\n--- Testing Sub-system 03 on Unseen Test Data ---")
    evaluate_subset(model, test_data, pos_digits, pos_digits, "Positive Test Set")
    evaluate_subset(model, test_data, neg_digits, pos_digits, "Negative (Unseen) Test Set")


if __name__ == "__main__":
    main()