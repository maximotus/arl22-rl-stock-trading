# arl22-rl-stock-trading

## Setup

Use poetry as the package manager. Install it using e.g.:

```bash
pip install poetry
```

## Usage

```bash
poetry install
python experiments/main.py --conf ./config/template-dqn.yaml --finnhub-key FINNHUB_API_KEY
```

## Data

In order to fetch data please make sure your `MetaTrader5`Terminal is running and has a registered and activated account
with your broker.

The package was tested with an [Admiral Markets](https://admiralmarkets.com/) Investement Demo Account (_Sign up with
Admirals, then go to the **Dashboard** and **ADD ACCOUNT** for the **Invest** option_)

You will also need an account for the [Finnhub](https://finnhub.io/) API.
