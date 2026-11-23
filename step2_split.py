from collections import Counter
from torch.utils.data import Subset
from torchvision import datasets, transforms


def get_split_datasets(data_dir="./data"):
    """Load MNIST train/test sets and partition training data into 3 non-IID subsets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_data = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    client_groups = {
        "Sub-system 01": [0, 1, 2],
        "Sub-system 02": [3, 4, 5, 6],
        "Sub-system 03": [7, 8, 9],
    }

    client_subsets = {}
    targets = train_data.targets

    for client_name, allowed_labels in client_groups.items():
        indices = [
            idx for idx, label in enumerate(targets) if int(label) in allowed_labels
        ]
        client_subsets[client_name] = Subset(train_data, indices)

    return train_data, test_data, client_subsets


def main():
    train_data, test_data, client_subsets = get_split_datasets()

    print("\n--- Split Dataset Summary ---")
    print(f"Total original training samples: {len(train_data)}")
    print(f"Total testing samples:           {len(test_data)}\n")

    total_split = 0
    for client_name, subset in client_subsets.items():
        # Count how many samples each digit has inside this client's subset
        label_counts = Counter(
            int(train_data.targets[i]) for i in subset.indices
        )
        sorted_counts = dict(sorted(label_counts.items()))
        total_split += len(subset)

        print(f"{client_name}: {len(subset)} samples | Breakdown: {sorted_counts}")

    print(f"\nSum of split training samples:   {total_split}")


if __name__ == "__main__":
    main()