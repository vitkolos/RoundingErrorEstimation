# Rounding Error Estimation

## Setup

```sh
# download the repository
git clone https://github.com/vitkolos/RoundingErrorEstimation.git
cd RoundingErrorEstimation

# install dependencies (using one of the following commands)
uv sync             # if uv is available
pip install -e .    # otherwise
```

To correctly install cuOpt on Linux without uv, you may need to add `--extra-index-url=https://pypi.nvidia.com`.

## Contents

- `appmax` – implementation of the AppMax method
- `tests` covering most of the `appmax` implementation
- `lin_opt_replication` – fragments from the [original repository](https://github.com/PetraVidnerova/RoundingErrorEstimation/)

## Executable modules

These modules can be run using `uv run -m` or `python -m`.

- `appmax` – the main module
- `appmax.visualization` – produces charts and tables
- `appmax.trainable` – trains a model
