
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------------------
# 1. Project configuration
# ----------------------------------------

SEED = 42
N = 300

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)


# ----------------------------------------
# 2. Generate independent variable
# ----------------------------------------

X = rng.normal(loc=10, scale=2, size=N)


# ----------------------------------------
# 3. Define structural regimes
# ----------------------------------------

regimes = [
    {
        "start": 0,
        "end": 100,
        "alpha": 2,
        "beta": 0.5,
        "sigma": 1
    },
    {
        "start": 100,
        "end": 200,
        "alpha": 5,
        "beta": 1.2,
        "sigma": 1
    },
    {
        "start": 200,
        "end": 300,
        "alpha": 3,
        "beta": 0.3,
        "sigma": 1
    }
]


# ----------------------------------------
# 4. Generate dependent variable
# ----------------------------------------

Y = np.zeros(N)
regime_id = np.zeros(N, dtype=int)

for j, regime in enumerate(regimes, start=1):

    start = regime["start"]
    end = regime["end"]

    alpha = regime["alpha"]
    beta = regime["beta"]
    sigma = regime["sigma"]

    errors = rng.normal(0, sigma, end - start)

    Y[start:end] = (
        alpha
        + beta * X[start:end]
        + errors
    )

    regime_id[start:end] = j


# ----------------------------------------
# 5. Create dataset
# ----------------------------------------

df = pd.DataFrame({
    "time": np.arange(1, N + 1),
    "X": X,
    "Y": Y,
    "regime": regime_id
})

file_path = DATA_DIR / "multiple_breaks.csv"
df.to_csv(file_path, index=False)

print("Dataset generated successfully!")
print(df.head())

print("\nDataset information:")
print(df.describe())

print("\nActual breakpoints: 100, 200")


# ----------------------------------------
# 6. Plot the generated time series
# ----------------------------------------

plt.figure(figsize=(13, 5))

plt.plot(
    df["time"],
    df["Y"],
    label="Observed Y",
    linewidth=1
)

plt.axvline(
    100,
    color="red",
    linestyle="--",
    label="Break 1"
)

plt.axvline(
    200,
    color="green",
    linestyle="--",
    label="Break 2"
)

plt.xlabel("Time")
plt.ylabel("Y")
plt.title("Simulated Time Series with Two Structural Breaks")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "simulated_multiple_breaks.png",
    dpi=300
)

plt.show()

