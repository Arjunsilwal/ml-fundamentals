# make_churn.py
import numpy as np
import pandas as pd


def make_churn(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)                       # months as a customer
    monthly = rng.uniform(20, 120, n).round(2)            # monthly bill
    calls = rng.poisson(1.5, n)                           # support calls
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["dsl", "fiber", "none"], n, p=[0.35, 0.45, 0.20])

    score = (
        -1.6
        - 0.04 * tenure
        + 0.015 * (monthly - 70)
        + 0.25 * calls
        + 1.3 * (contract == "month-to-month")
        - 0.8 * (contract == "two-year")
        + 0.5 * (internet == "fiber")
    )
    churn = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly,
        "support_calls": calls,
        "contract": contract,
        "internet": internet,
        "churn": churn,
    })

    df.loc[rng.random(n) < 0.05, "monthly_charges"] = np.nan   # 5% missing values
    return df