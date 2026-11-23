from torchvision import datasets, transforms


def main():
    # Convert 28x28 images into PyTorch tensors (pixel values scaled to 0.0 - 1.0)
    transform = transforms.ToTensor()

    print("Downloading / loading MNIST dataset...")
    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    print("\n--- Dataset Summary ---")
    print(f"Training images count: {len(train_data)}")
    print(f"Testing images count:  {len(test_data)}")

    # Inspect one sample image and its label
    sample_image, sample_label = train_data[0]
    print(f"Single image tensor shape: {sample_image.shape} (Channels, Height, Width)")
    print(f"First sample true label:   {sample_label}")


if __name__ == "__main__":
    main()