# Project Structure

```
project3/
├── README.md                                 # you are here
├── report.md                                 # report source
├── report.pdf                                # compiled report
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

## uv

```bash
uv sync

uv run main.py
```
