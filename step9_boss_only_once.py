import torch
from torch.utils.data import DataLoader, Subset
from step2_split import get_split_datasets
from step3_train_sub1 import DigitNet, device


def fed_avg_once(state_dicts, sample_counts):
    """Aggregate client weights once using sample-weighted averaging."""
    total_samples = sum(sample_counts)
    merged_state = {}

    for key in state_dicts[0].keys():
        weighted_sum = None
        for sd, count in zip(state_dicts, sample_counts):
            weight_ratio = count / total_samples
            part = sd[key].float() * weight_ratio
            weighted_sum = part if weighted_sum is None else weighted_sum + part

        orig_dtype = state_dicts[0][key].dtype
        if orig_dtype.is_floating_point:
            merged_state[key] = weighted_sum.to(orig_dtype)
        else:
            merged_state[key] = torch.round(weighted_sum).to(orig_dtype)

    return merged_state


def evaluate_boss(model, test_data, digits=None, label="All Digits (0-9)"):
    """Evaluate only the boss model on the test set."""
    if digits is None:
        subset = test_data
    else:
        indices = [
            idx for idx, target in enumerate(test_data.targets) if int(target) in digits
        ]
        subset = Subset(test_data, indices)

    loader = DataLoader(subset, batch_size=256, shuffle=False)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"{label} (n={total}): Boss Accuracy = {acc:.2f}%")


def main():
    _, test_data, client_subsets = get_split_datasets()

    subsystem_names = ["Sub-system 01", "Sub-system 02", "Sub-system 03"]
    weight_files = ["sub1_model.pt", "sub2_model.pt", "sub3_model.pt"]
    sample_counts = [len(client_subsets[name]) for name in subsystem_names]

    # 1. Pull updated weights from all 3 sub-systems (read-only; nothing is sent back)
    received_weights = [
        torch.load(path, map_location="cpu") for path in weight_files
    ]

    # 2. Boss system blends the 3 weight dictionaries once
    boss_weights = fed_avg_once(received_weights, sample_counts)
    boss_model = DigitNet().to(device)
    boss_model.load_state_dict({k: v.to(device) for k, v in boss_weights.items()})
    torch.save(boss_model.state_dict(), "boss_once_model.pt")

    # 3. Test only the boss model (sub-systems remain untouched)
    print("\n--- Boss System Evaluation (1-Pass Merge, No Feedback to Sub-systems) ---")
    evaluate_boss(boss_model, test_data, digits=None, label="All Digits [0-9]")
    evaluate_boss(boss_model, test_data, digits=[0, 1, 2], label="Sub-sys 01 Group [0, 1, 2]")
    evaluate_boss(boss_model, test_data, digits=[3, 4, 5, 6], label="Sub-sys 02 Group [3, 4, 5, 6]")
    evaluate_boss(boss_model, test_data, digits=[7, 8, 9], label="Sub-sys 03 Group [7, 8, 9]")
    print("\nDone. Saved boss_once_model.pt; sub-systems were not updated.")


if __name__ == "__main__":
    main()