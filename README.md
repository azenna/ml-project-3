# Project Structure

```
project3/
├── README.md                                 # you are here
├── report.md                                 # report source
├── report.pdf                                # compiled report
├── ai_transcript.pdf                         # my conversations with gemini :()
├── pyproject.toml                            # dependencies for uv
├── uv.lock                                   # uv lockfile
├── main.py                                   # entrypoint (run me)
├── report.py                                 # table generation
├── architecture.py                           # all model architectures
├── configuration.py                          # configurations for hyperparamaters for tuning and testing
├── data/                                     # data container
└── run_log.jsonl                             # append only training log
```

# Build & Run

Install uv and use `uv sync` to install the projects dependencies.

The training harness is in `main.py`, `uv run main.py --help`. The main script, runs a specified training or a set of trainings defined in `./configuration.py`, and appends results to `./run_log.jsonl`.

A full list of available arguments for the program can be found by running `uv run main.py --help`. The main script can be used in several modes:

- `uv run main.py --configs --tune` can be used to train and validate all of the model `tune_configurations` defined in `./configuration.py`
- `uv run main.py --configs' will train and test model `test_configurations` also in `./configuration.py`
- `uv run main.py --architecture (shallow_mlp | medium_mlp | deep_mlp | simple_conv | enhanced_conv) --dataset (mnist | cifar10)` will train and test a specified model on a specified dataset. Use the `--tune` paramater to put the script and validation mode. A full list of hyperparmater arguments can be found with `uv run main.py --help`.

There is also a reports script `uv run report.py --help` for analyzing training results:

- `uv run report.py --arch (mlp | conv | shallow_mlp | medium_mlp | deep_mlp | simple_conv | enhanced_conv) --dataset (mnist | cifar) [--test]` by default this will only report data from tuning runs, but if you supply the `--test` argument it will build reports from final test set runs.
