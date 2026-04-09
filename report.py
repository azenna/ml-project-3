import argparse
import json
from prettytable import PrettyTable


def filter_runs(arch, dataset, runs):
    # filter for arch and dataset
    runs = [
        run for run in runs if run["architecture"] == arch and run["dataset"] == dataset
    ]

    # filter for non trivial runs
    runs = [
        run for run in runs if "num_epochs" in run.keys() and run["num_epochs"] >= 8
    ]

    return runs


def mlp_report(runs):
    table = PrettyTable()
    table.field_names = ["LR", "Batch", "Opt", "Dropout", "Acc", "Runtime"]
    table.sortby = "Acc"
    table.reversesort = True

    for run in runs:
        try:
            table.add_row(
                [
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
    table.field_names = ["LR", "Batch", "L2", "Acc", "Runtime"]
    table.sortby = "Acc"
    table.reversesort = True

    for run in runs:
        try:
            table.add_row(
                [
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

    parser.add_argument("-a", "--arch", required=True)
    parser.add_argument("-d", "--dataset", required=True)

    args = parser.parse_args()

    with open("run_log.jsonl", "r") as file:
        lines = file.readlines()
        runs = [json.loads(line) for line in lines]
        filtered_runs = filter_runs(args.arch, args.dataset, runs)

        if "mlp" in args.arch:
            table = mlp_report(filtered_runs)
        elif "conv" in args.arch:
            table = conv_report(filtered_runs)

        print(args.arch, args.dataset)
        print(table)


if __name__ == "__main__":
    main()
