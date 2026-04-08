"""Local KNN evaluation for VAE latent space quality on MNIST."""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import Counter
import numpy as np
import yaml, os, sys, argparse

from config import Config
from trainer_vae import VAETrainer
from utils.data_utils import set_seed

def knn_classify(train_z, train_labels, test_z, test_labels, k=3):
    """KNN classification in latent space."""
    # train_z: (N_train, D), test_z: (N_test, D)
    correct = 0
    total = test_z.size(0)
    
    # Compute pairwise distances
    # Using batched computation to avoid OOM
    batch_size = 256
    for i in range(0, total, batch_size):
        batch = test_z[i:i+batch_size]  # (B, D)
        dists = torch.cdist(batch, train_z)  # (B, N_train)
        _, topk_idx = dists.topk(k, dim=1, largest=False)  # (B, k)
        topk_labels = train_labels[topk_idx]  # (B, k)
        
        for j in range(batch.size(0)):
            labels_list = topk_labels[j].tolist()
            vote = Counter(labels_list).most_common(1)[0][0]
            if vote == test_labels[i + j].item():
                correct += 1
    
    return correct / total


def few_shot_sample(dataset, n_shots, n_classes=10, seed=42):
    """Sample n_shots per class."""
    rng = np.random.RandomState(seed)
    indices_by_class = {c: [] for c in range(n_classes)}
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
        indices_by_class[label].append(idx)
    
    selected = []
    for c in range(n_classes):
        chosen = rng.choice(indices_by_class[c], size=n_shots, replace=False)
        selected.extend(chosen)
    
    return Subset(dataset, selected)


def encode_dataset(model, dataset, device, batch_size=256):
    """Encode a dataset through VAE encoder, return (mu, labels)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_mu = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            z, mu, logvar = model.encode(data)
            all_mu.append(mu.cpu())
            all_labels.append(labels)
    
    return torch.cat(all_mu, dim=0), torch.cat(all_labels, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='configs/config_vae.yaml')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--n_shots', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--n_test', type=int, default=1000, help='Number of test samples')
    args = parser.parse_args()

    set_seed(42)

    with open(args.config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
        config = Config(config_dict=config_dict)

    trainer = VAETrainer(config=config, output_dir=args.output_dir)
    
    # Load trained model
    model_path = f"{trainer.output_dir}/vae_{trainer.dataset.lower()}.pth"
    print(f"Loading model from {model_path}")
    model = trainer.load_model(model_path, map_location=trainer.device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    
    if config.data.dataset.lower() == 'mnist':
        trainset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    else:
        trainset = datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
        testset = datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)

    # Few-shot train set
    few_shot_train = few_shot_sample(trainset, n_shots=args.n_shots, seed=42)
    print(f"Few-shot train: {len(few_shot_train)} samples ({args.n_shots}-shot)")

    # Subsample test set for speed 
    rng = np.random.RandomState(42)
    test_indices = rng.choice(len(testset), size=min(args.n_test, len(testset)), replace=False)
    test_subset = Subset(testset, test_indices)
    print(f"Test set: {len(test_subset)} samples")

    # Encode
    train_z, train_labels = encode_dataset(model, few_shot_train, trainer.device)
    test_z, test_labels = encode_dataset(model, test_subset, trainer.device)

    # KNN
    for k in [1, 3, 5]:
        acc = knn_classify(train_z, train_labels, test_z, test_labels, k=k)
        print(f"  {args.n_shots}-shot KNN (k={k}): accuracy = {acc:.4f}")

    # Also test with full test set
    print("\nFull test set (10000):")
    test_z_full, test_labels_full = encode_dataset(model, testset, trainer.device)
    for k in [1, 3, 5]:
        acc = knn_classify(train_z, train_labels, test_z_full, test_labels_full, k=k)
        print(f"  {args.n_shots}-shot KNN (k={k}): accuracy = {acc:.4f}")


if __name__ == '__main__':
    main()
