import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import random_split
import architecture
import sys
import argparse
import json
from datetime import datetime, timezone
from torchvision.datasets import MNIST
import os
import urllib.request
import gzip
import shutil

device = torch.device("cuda")


def load_train_test(dataset_str, batch_size, validation=False):

    if dataset_str == "mnist":
        dataset = MNIST
        validation_split = [50000, 10000]
    elif dataset_str == "cifar":
        dataset = torchvision.datasets.CIFAR10
        validation_split = [45000, 5000]

    ts = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    full_train = dataset(root="./data", train=True, download=False, transform=ts)

    if validation:
        train, val = random_split(full_train, validation_split)
        train_loader = DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=8
        )
        val_loader = DataLoader(val, batch_size=batch_size, num_workers=8)

        return (train_loader, val_loader)
    else:
        test = dataset(root="./data", train=False, download=False, transform=ts)
        test_loader = DataLoader(test, batch_size=batch_size, num_workers=8)

        train_loader = DataLoader(
            full_train, batch_size=batch_size, shuffle=True, num_workers=8
        )
        return (train_loader, test_loader)


def train(model, loader, loss_fn, optimizer, num_epochs=1):
    model.train()

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}"
                    f" ({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )


def evaluate(model, loader):
    model.eval()
    all_preds = []
    targets = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            outputs = model(data)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            targets.extend(target.numpy())

    return classification_report(targets, all_preds, digits=4, output_dict=True)


def run(args):
    arch = architecture.architectures[args.dataset][args.arch]

    (train_load, eval_load) = load_train_test(
        args.dataset, args.batch_size, validation=args.tune
    )

    model = arch.to(device)

    if args.optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr=args.learning_rate)

    loss_fn = nn.CrossEntropyLoss()

    train(model, train_load, loss_fn, opt)

    classification_report = evaluate(model, eval_load)

    timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "timestamp": timestamp,
        "dataset": args.dataset,
        "architecture": args.arch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "classification_report": classification_report,
    }


# Program that keeps a set of logs of each run
# stores hyperparmaters used, validation results or test results based on --tune
# architecture used and on which dataset etc.
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tune", action="store_true")
    parser.add_argument("-a", "--arch", required=True)
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-o", "--optimizer", default="adam")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001)

    args = parser.parse_args()
    run_log = run(args)

    with open("run_log.jsonl", "a") as f:
        f.write(json.dumps(run_log) + "\n")


if __name__ == "__main__":
    main()
