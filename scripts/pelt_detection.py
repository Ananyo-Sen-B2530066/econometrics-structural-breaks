
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt

from pathlib import Path
import statsmodels.api as sm


# ==========================================
# 1. CONFIGURATION
# ==========================================

MIN_SEGMENT = 30
N_PENALTIES = 50

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "outputs" / "figures"
TABLE_DIR = BASE_DIR / "outputs" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. REGRESSION SEGMENT COST
# ==========================================

def regression_ssr(y, x, start, end):

    ys = y[start:end]
    xs = x[start:end]

    X = np.column_stack([
        np.ones(len(xs)),
        xs
    ])

    beta = np.linalg.lstsq(
        X, ys, rcond=None
    )[0]

    residuals = ys - X @ beta

    return np.sum(residuals ** 2)


def variance_ssr(z, start, end):

    segment = z[start:end]

    return np.sum(
        (segment - segment.mean()) ** 2
    )


# ==========================================
# 3. BIC CALCULATION
# ==========================================

def calculate_bic(ssr, n, m, mode):

    if ssr <= 0:
        return np.inf

    if mode == "regression":

        # Two regression coefficients
        # per regime, plus break locations.
        k = 2 * (m + 1) + m

    else:

        # One segment mean per regime,
        # one common variance and break locations.
        k = (m + 1) + 1 + m

    return (
        n * np.log(ssr / n)
        + k * np.log(n)
    )


# ==========================================
# 4. GENERATE PELT CANDIDATE SEGMENTATIONS
# ==========================================

def run_pelt(signal, penalties, model):

    candidate_results = {}

    for penalty in penalties:

        algorithm = rpt.Pelt(
            model=model,
            min_size=MIN_SEGMENT,
            jump=1
        )

        algorithm.fit(signal)

        # Final endpoint n is included.
        endpoints = algorithm.predict(
            pen=float(penalty)
        )

        breakpoints = endpoints[:-1]

        candidate_results[
            tuple(breakpoints)
        ] = penalty

    return candidate_results


# ==========================================
# 5. SELECT BEST SEGMENTATION USING BIC
# ==========================================

def select_segmentation(
    y,
    x,
    candidates,
    mode
):

    n = len(y)
    results = []

    for breakpoints in candidates:

        boundaries = (
            [0]
            + list(breakpoints)
            + [n]
        )

        total_ssr = 0

        for i in range(len(boundaries) - 1):

            start = boundaries[i]
            end = boundaries[i + 1]

            if mode == "regression":

                total_ssr += regression_ssr(
                    y, x, start, end
                )

            else:

                total_ssr += variance_ssr(
                    y, start, end
                )

        m = len(breakpoints)

        bic = calculate_bic(
            total_ssr,
            n,
            m,
            mode
        )

        results.append({
            "breaks": m,
            "SSR": total_ssr,
            "BIC": bic,
            "breakpoints": list(breakpoints)
        })

    results_df = pd.DataFrame(results)

    best = results_df.loc[
        results_df["BIC"].idxmin()
    ]

    return results_df, best


# ==========================================
# 6. ANALYSE REGRESSION BREAKS
# ==========================================

def analyse_regression_scenario(name):

    df = pd.read_csv(
        DATA_DIR / f"{name}.csv"
    )

    y = df["Y"].to_numpy()
    x = df["X"].to_numpy()

    # CostLinear expects:
    # first column = dependent variable
    # remaining columns = regressors

    signal = np.column_stack([
        y,
        np.ones(len(y)),
        x
    ])

    # Estimate pooled regression variance
    X = sm.add_constant(x)

    pooled = sm.OLS(y, X).fit()

    sigma2 = np.sum(pooled.resid ** 2) / (
        len(y) - X.shape[1]
    )

    # Explore a range of penalty values
    penalties = (
        sigma2
        * np.log(len(y))
        * np.geomspace(0.1, 100, N_PENALTIES)
    )

    candidates = run_pelt(
        signal,
        penalties,
        model="linear"
    )

    results, best = select_segmentation(
        y,
        x,
        candidates,
        mode="regression"
    )

    return df, results, best


# ==========================================
# 7. ANALYSE VARIANCE BREAK
# ==========================================

def analyse_variance_scenario(name):

    df = pd.read_csv(
        DATA_DIR / f"{name}.csv"
    )

    y = df["Y"].to_numpy()
    x = df["X"].to_numpy()

    # Fit one stable regression
    X = sm.add_constant(x)

    pooled = sm.OLS(y, X).fit()

    # Squared residuals
    z = pooled.resid ** 2

    # PELT on squared residuals
    signal = z.reshape(-1, 1)

    scale = np.var(z, ddof=1)

    penalties = (
        scale
        * np.log(len(z))
        * np.geomspace(0.1, 100, N_PENALTIES)
    )

    candidates = run_pelt(
        signal,
        penalties,
        model="l2"
    )

    results, best = select_segmentation(
        z,
        x,
        candidates,
        mode="variance"
    )

    return df, results, best, z


# ==========================================
# 8. VISUALIZATION
# ==========================================

def plot_results(
    df,
    best,
    name,
    mode,
    values=None
):

    if values is None:
        values = df["Y"].to_numpy()

    plt.figure(figsize=(13, 5))

    plt.plot(
        df["time"],
        values,
        label="Observed series",
        linewidth=1
    )

    for point in best["breakpoints"]:

        plt.axvline(
            point,
            color="red",
            linestyle="--"
        )

    plt.title(f"PELT Detection: {name}")

    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / f"pelt_{name}.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 9. RUN EXPERIMENTS
# ==========================================

regression_scenarios = [
    "S1_no_break",
    "S2_intercept_break",
    "S3_slope_break",
    "S4_multiple_breaks"
]

all_results = []

for name in regression_scenarios:

    df, results, best = analyse_regression_scenario(
        name
    )

    print("\n" + "=" * 55)
    print(name)

    print("\nCandidate model comparison:")
    print(results.to_string(index=False))

    print("\nSelected number of breaks:")
    print(int(best["breaks"]))

    print("\nEstimated breakpoints:")
    print(best["breakpoints"])

    plot_results(
        df,
        best,
        name,
        mode="regression"
    )

    results["scenario"] = name
    all_results.append(results)


# ==========================================
# 10. VARIANCE SCENARIO
# ==========================================

name = "S5_variance_break"

df, results, best, squared_residuals = (
    analyse_variance_scenario(name)
)

print("\n" + "=" * 55)
print(name)

print("\nVariance changepoint model comparison:")
print(results.to_string(index=False))

print("\nEstimated variance breakpoints:")
print(best["breakpoints"])

plot_results(
    df,
    best,
    name,
    mode="variance",
    values=squared_residuals
)

results["scenario"] = name
all_results.append(results)


# ==========================================
# 11. SAVE RESULTS
# ==========================================

final_results = pd.concat(
    all_results,
    ignore_index=True
)

final_results.to_csv(
    TABLE_DIR / "pelt_results.csv",
    index=False
)

print("\nPELT ANALYSIS COMPLETED")

