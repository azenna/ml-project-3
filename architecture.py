import torch.nn as nn


def shallow_mlp_mnist(dropout=0.5):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 10),
    )


def medium_mlp_mnist(dropout=0.5):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 10),
    )


def deep_mlp_mnist(dropout=0.5):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 10),
    )


def simple_conv_mnist():
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(16 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def enh_conv_mnist():
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(128 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def shallow_mlp_cifar(dropout=0.5):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(32 * 32 * 3, 1024),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, 10),
    )


def medium_mlp_cifar(dropout=0.5):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 1024),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 10),
    )


def deep_mlp_cifar(dropout=0.5):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 2048),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 10),
    )


def simple_conv_cifar():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(16 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def enh_conv_cifar():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def get_architectures(dropout=0.5):
    return {
        "cifar": {
            "shallow_mlp": shallow_mlp_cifar(dropout),
            "medium_mlp": medium_mlp_cifar(dropout),
            "deep_mlp": deep_mlp_cifar(dropout),
            "simple_conv": simple_conv_cifar(),
            "enhanced_conv": enh_conv_cifar(),
        },
        "mnist": {
            "shallow_mlp": shallow_mlp_mnist(dropout),
            "medium_mlp": medium_mlp_mnist(dropout),
            "deep_mlp": deep_mlp_mnist(dropout),
            "simple_conv": simple_conv_mnist(),
            "enhanced_conv": enh_conv_mnist(),
        },
    }
