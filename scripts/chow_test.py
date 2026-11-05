
import pandas as pd
import numpy as np

from pathlib import Path
from scipy.stats import f
import statsmodels.api as sm


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"


def chow_test(df, breakpoint):

    # Split observations
    before = df[df["time"] <= breakpoint]
    after = df[df["time"] > breakpoint]

    # Dependent and independent variables
    X_full = sm.add_constant(df["X"])
    X1 = sm.add_constant(before["X"])
    X2 = sm.add_constant(after["X"])

    # Restricted pooled regression
    pooled = sm.OLS(df["Y"], X_full).fit()

    # Unrestricted separate regressions
    model1 = sm.OLS(before["Y"], X1).fit()
    model2 = sm.OLS(after["Y"], X2).fit()

    SSR_R = np.sum(pooled.resid ** 2)

    SSR_U = (
        np.sum(model1.resid ** 2)
        + np.sum(model2.resid ** 2)
    )

    n1 = len(before)
    n2 = len(after)
    k = 2

    numerator = (SSR_R - SSR_U) / k

    denominator = SSR_U / (n1 + n2 - 2 * k)

    F_stat = numerator / denominator

    p_value = f.sf(
        F_stat,
        k,
        n1 + n2 - 2 * k
    )

    return {
        "breakpoint": breakpoint,
        "F_statistic": F_stat,
        "p_value": p_value,
        "reject_H0": p_value < 0.05
    }


# ==========================================
# RUN CHOW TEST
# ==========================================

scenarios = {
    "S1_no_break": [],
    "S2_intercept_break": [150],
    "S3_slope_break": [150],
    "S4_multiple_breaks": [100, 200],
    "S5_variance_break": [150]
}

for name, breakpoints in scenarios.items():

    df = pd.read_csv(
        DATA_DIR / f"{name}.csv"
    )

    print(f"\n{'=' * 45}")
    print(name)

    if not breakpoints:
        print("No known break: test not applicable.")
        continue

    for point in breakpoints:

        result = chow_test(df, point)

        print(f"\nBreakpoint: {point}")
        print(f"F-statistic: {result['F_statistic']:.4f}")
        print(f"P-value: {result['p_value']:.6f}")
        print(f"Reject H0: {result['reject_H0']}")

