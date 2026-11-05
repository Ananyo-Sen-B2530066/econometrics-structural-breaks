
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from pathlib import Path
from statsmodels.stats.diagnostic import breaks_cusumolsresid


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "outputs" / "figures"


scenarios = [
    "S1_no_break",
    "S2_intercept_break",
    "S3_slope_break",
    "S4_multiple_breaks",
    "S5_variance_break"
]


for name in scenarios:

    df = pd.read_csv(
        DATA_DIR / f"{name}.csv"
    )

    X = sm.add_constant(df["X"])
    Y = df["Y"]

    model = sm.OLS(Y, X).fit()

    # CUSUM test based on OLS residuals
    statistic, p_value, critical_values = (
        breaks_cusumolsresid(
            model.resid,
            ddof=2
        )
    )

    print("\n", "=" * 45)
    print(name)

    print("CUSUM statistic:", statistic)
    print("P-value:", p_value)
    print("Critical values:", critical_values)

    print(
        "Parameter instability detected:",
        p_value < 0.05
    )

    # Recursive residual CUSUM visualization
    from statsmodels.stats.diagnostic import recursive_olsresiduals

    recursive_results = recursive_olsresiduals(
        model,
        alpha=0.95,
        skip=20
    )

    cusum_ci = recursive_results[6]

    lower = cusum_ci[0, :]
    upper = cusum_ci[1, :]

    plt.figure(figsize=(12, 5))

    plt.plot(cusum_ci, label="CUSUM")
    plt.plot(lower, "r--", label="Lower boundary")
    plt.plot(upper, "r--", label="Upper boundary")

    plt.title(f"CUSUM Stability Test: {name}")
    plt.xlabel("Observation")
    plt.ylabel("Cumulative recursive residuals")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / f"cusum_{name}.png",
        dpi=300
    )

    plt.close()

