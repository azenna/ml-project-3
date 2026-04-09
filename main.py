import torch
import torchvision
import torchvision.transforms as transforms
from types import SimpleNamespace
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
import configuration
import time

device = torch.device("cuda")


def load_train_test(dataset_str, batch_size, validation=False):
    ts = transforms.Compose([transforms.ToTensor()])

    if dataset_str == "mnist":
        validation_split = [50000, 10000]

        full_train = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=ts
        )

        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=ts
        )

    elif dataset_str == "cifar":
        full_train = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=ts
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=ts
        )
        validation_split = [45000, 5000]

    if validation:
        train, val = random_split(full_train, validation_split)
        train_loader = DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=8
        )
        val_loader = DataLoader(val, batch_size=batch_size, num_workers=8)

        return (train_loader, val_loader)
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

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

    print(classification_report(targets, all_preds, digits=4))
    return classification_report(targets, all_preds, digits=4, output_dict=True)


def train_and_eval(args):
    arch = architecture.get_architectures(args.dropout)[args.dataset][args.architecture]

    (train_load, eval_load) = load_train_test(
        args.dataset, args.batch_size, validation=args.tune
    )

    model = arch.to(device)

    if args.optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr=args.learning_rate)

    loss_fn = nn.CrossEntropyLoss()

    start = time.perf_counter()
    train(model, train_load, loss_fn, opt, num_epochs=args.num_epochs)
    end = time.perf_counter()

    classification_report = evaluate(model, eval_load)

    timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "timestamp": timestamp,
        "dataset": args.dataset,
        "tune": args.tune,
        "architecture": args.architecture,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "num_epochs": args.num_epochs,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "classification_report": classification_report,
        "runtime": f"{end - start:.06f}s",
    }


def run(args):
    print("=" * 32 + f"{args.dataset} {args.architecture}" + "=" * 32)
    run_log = train_and_eval(args)

    with open("run_log.jsonl", "a") as f:
        f.write(json.dumps(run_log) + "\n")


def run_all_configs():
    for config in configuration.configurations:
        with open("run_log.jsonl", "r") as f:
            runs = [json.loads(line) for line in f.readlines()]
            if any(
                all(run.get(k) == v for k, v in config.items() if k != "tune")
                for run in runs
            ):
                print("Skipping: ", config)
                continue

        run(SimpleNamespace(config))


# Program that keeps a set of logs of each run
# stores hyperparmaters used, validation results or test results based on --tune
# architecture used and on which dataset etc.
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--configs", action="store_true")
    parser.add_argument("-a", "--architecture")
    parser.add_argument("-d", "--dataset")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-n", "--num_epochs", type=int, default=16)
    parser.add_argument("-o", "--optimizer", default="adam")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    args = parser.parse_args()

    if args.configs and args.tune:
        run_all_configs(tune_configurations)
    else:
        run_all_configs(test_configurations)


if __name__ == "__main__":
    main()
