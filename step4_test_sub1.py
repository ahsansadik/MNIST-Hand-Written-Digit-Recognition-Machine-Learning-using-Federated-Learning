import torch
from torch.utils.data import DataLoader, Subset
from step2_split import get_split_datasets
from step3_train_sub1 import DigitNet, device


def evaluate_subset(model, test_data, digits, label_name):
    """Evaluate exact multi-class accuracy on a specific set of digits."""
    indices = [
        idx for idx, label in enumerate(test_data.targets) if int(label) in digits
    ]
    subset = Subset(test_data, indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    predicted_as_012 = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            predicted_as_012 += torch.isin(preds, torch.tensor([0, 1, 2], device=device)).sum().item()

    exact_acc = 100.0 * correct / total if total > 0 else 0.0
    rate_012 = 100.0 * predicted_as_012 / total if total > 0 else 0.0

    print(
        f"{label_name} (digits {digits}, n={total}): "
        f"Exact Match Acc = {exact_acc:.2f}% | "
        f"Predicted as [0,1,2] = {rate_012:.2f}%"
    )


def main(model_path="sub1_model.pt", title="Sub-system 01"):
    _, test_data, _ = get_split_datasets()

    model = DigitNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    print(f"\n--- Testing {title} on Unseen Test Data ---")
    evaluate_subset(model, test_data, [0, 1, 2], "Positive Test Set")
    evaluate_subset(model, test_data, [4, 5, 6, 7, 8, 9], "Negative (Unseen) Test Set")


if __name__ == "__main__":
    main()