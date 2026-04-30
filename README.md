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

## Contents

- `appmax` – implementation of the AppMax method
- `tests` covering most of the `appmax` implementation
- `lin_opt_replication` – fragments from the [original repository](https://github.com/PetraVidnerova/RoundingErrorEstimation/)
