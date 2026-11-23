import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from step2_split import get_split_datasets
from step3_train_sub1 import DigitNet, device


def fed_avg(state_dicts, sample_counts):
    """Aggregate client weights using sample-weighted averaging."""
    total_samples = sum(sample_counts)
    merged_state = {}

    for key in state_dicts[0].keys():
        weighted_sum = None
        for sd, count in zip(state_dicts, sample_counts):
            ratio = count / total_samples
            part = sd[key].float() * ratio
            weighted_sum = part if weighted_sum is None else weighted_sum + part

        orig_dtype = state_dicts[0][key].dtype
        if orig_dtype.is_floating_point:
            merged_state[key] = weighted_sum.to(orig_dtype)
        else:
            merged_state[key] = torch.round(weighted_sum).to(orig_dtype)

    return merged_state


def evaluate_accuracy(model, test_data, digits=None):
    """Evaluate exact accuracy across all digits (0-9) or a specific digit subset."""
    if digits is None:
        subset = test_data
    else:
        indices = [
            idx for idx, label in enumerate(test_data.targets) if int(label) in digits
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

    return 100.0 * correct / total if total > 0 else 0.0


def train_one_round(model, loader, lr=0.005):
    """Train a sub-system for 1 local epoch on its private digits and return weights."""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return {k: v.cpu() for k, v in model.state_dict().items()}


def main(rounds=10, lr=0.005):
    _, test_data, client_subsets = get_split_datasets()

    sub_names = ["Sub-system 01", "Sub-system 02", "Sub-system 03"]
    model_paths = ["sub1_model.pt", "sub2_model.pt", "sub3_model.pt"]

    sub_loaders = {
        name: DataLoader(client_subsets[name], batch_size=64, shuffle=True)
        for name in sub_names
    }
    sample_counts = [len(client_subsets[name]) for name in sub_names]

    # Start boss model from the initial FedAvg blend of the 3 specialist checkpoints
    initial_states = [torch.load(p, map_location="cpu") for p in model_paths]
    boss_state = fed_avg(initial_states, sample_counts)

    boss_model = DigitNet().to(device)
    boss_model.load_state_dict({k: v.to(device) for k, v in boss_state.items()})

    sub_models = {
        name: copy.deepcopy(boss_model).to(device) for name in sub_names
    }

    print(f"\n=== Running {rounds}-Round Federated Learning Loop (lr={lr}) ===\n")

    for rnd in range(1, rounds + 1):
        client_updates = []

        # 1. Send current boss weights to each sub-system & run 1 local epoch
        boss_cpu_state = {k: v.cpu() for k, v in boss_model.state_dict().items()}
        for name in sub_names:
            sub_models[name].load_state_dict(
                {k: v.to(device) for k, v in copy.deepcopy(boss_cpu_state).items()}
            )
            updated_sd = train_one_round(sub_models[name], sub_loaders[name], lr=lr)
            client_updates.append(updated_sd)

        # 2. Boss server averages the 3 updated weight dictionaries
        new_boss_state = fed_avg(client_updates, sample_counts)
        boss_model.load_state_dict({k: v.to(device) for k, v in new_boss_state.items()})

        # 3. Redistribute updated boss model back to all 3 sub-systems
        for name in sub_names:
            sub_models[name].load_state_dict(copy.deepcopy(boss_model.state_dict()))

        # 4. Evaluate Boss and all 3 Sub-systems across all 10 digits (0-9)
        boss_acc = evaluate_accuracy(boss_model, test_data)
        s1_acc = evaluate_accuracy(sub_models["Sub-system 01"], test_data)
        s2_acc = evaluate_accuracy(sub_models["Sub-system 02"], test_data)
        s3_acc = evaluate_accuracy(sub_models["Sub-system 03"], test_data)
        s1_unseen = evaluate_accuracy(sub_models["Sub-system 01"], test_data, [7, 8, 9])

        print(
            f"Round {rnd:02d}/{rounds:02d} | "
            f"Boss (0-9): {boss_acc:6.2f}% | "
            f"Sub1 (0-9): {s1_acc:6.2f}% [7-9: {s1_unseen:5.2f}%] | "
            f"Sub2 (0-9): {s2_acc:6.2f}% | "
            f"Sub3 (0-9): {s3_acc:6.2f}%"
        )

    torch.save(boss_model.state_dict(), "boss_fl_10rounds.pt")
    print("\nFinished 10 federated rounds. Saved final boss model to boss_fl_10rounds.pt")


if __name__ == "__main__":
    main()