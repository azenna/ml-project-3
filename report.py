import argparse
import json
from prettytable import PrettyTable


def mlp_report(runs):
    table = PrettyTable()
    table.field_names = [
        "Arch",
        "LR",
        "Batch",
        "Opt",
        "Dropout",
        "Acc",
        "Runtime",
    ]
    table.sortby = "Acc"
    table.reversesort = True

    for run in runs:
        try:
            table.add_row(
                [
                    run["architecture"],
                    run["learning_rate"],
                    run["batch_size"],
                    run["optimizer"],
                    run["dropout"],
                    run["classification_report"]["accuracy"],
                    run["runtime"],
                ]
            )
        except:
            continue

    return table


def conv_report(runs):
    table = PrettyTable()
    table.field_names = ["Arch", "LR", "Batch", "L2", "Acc", "Runtime"]
    table.sortby = "Acc"
    table.reversesort = True

    for run in runs:
        try:
            table.add_row(
                [
                    run["architecture"],
                    run["learning_rate"],
                    run["batch_size"],
                    run["weight_decay"],
                    run["classification_report"]["accuracy"],
                    run["runtime"],
                ]
            )
        except:
            continue

    return table


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--arch")

    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    filters = [lambda run: run["num_epochs"] >= 8]

    if args.arch:
        filters.append(lambda run: args.arch in run["architecture"])
    if args.dataset:
        filters.append(lambda run: run["dataset"] == args.dataset)
    if args.test:
        filters.append(lambda run: not run["tune"])

    with open("run_log.jsonl", "r") as file:
        lines = file.readlines()
        runs = [json.loads(line) for line in lines]

        filtered_runs = []
        for run in runs:
            try:
                if all(f(run) for f in filters):
                    filtered_runs.append(run)
            except:
                continue

        if "mlp" in args.arch:
            table = mlp_report(filtered_runs)
        elif "conv" in args.arch:
            table = conv_report(filtered_runs)

        print(args.arch, args.dataset)
        print(table)


if __name__ == "__main__":
    main()
