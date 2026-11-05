
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ==========================================
# 1. CONFIGURATION
# ==========================================

N_BOOTSTRAP = 999
TRIM = 0.15
SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "outputs" / "figures"
TABLE_DIR = BASE_DIR / "outputs" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CALCULATE F-STATISTIC
# ==========================================

def calculate_f_statistic(y, X, breakpoint):

    n = len(y)
    k = X.shape[1]

    X1 = X[:breakpoint]
    X2 = X[breakpoint:]

    y1 = y[:breakpoint]
    y2 = y[breakpoint:]

    # Restricted regression
    beta_r = np.linalg.lstsq(X, y, rcond=None)[0]

    residual_r = y - X @ beta_r

    SSR_R = np.sum(residual_r ** 2)

    # Unrestricted regressions
    beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
    beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]

    SSR_U = (
        np.sum((y1 - X1 @ beta1) ** 2)
        + np.sum((y2 - X2 @ beta2) ** 2)
    )

    numerator = (SSR_R - SSR_U) / k

    denominator = SSR_U / (n - 2 * k)

    if denominator <= 0:
        return np.nan

    return numerator / denominator


# ==========================================
# 3. SUPREMUM F TEST
# ==========================================

def sup_f_test(y, X, trim=0.15):

    n = len(y)

    lower = int(np.ceil(n * trim))
    upper = int(np.floor(n * (1 - trim)))

    candidate_points = np.arange(
        lower,
        upper + 1
    )

    statistics = []

    for point in candidate_points:

        value = calculate_f_statistic(
            y,
            X,
            point
        )

        statistics.append(value)

    statistics = np.asarray(statistics)

    best_index = np.nanargmax(statistics)

    best_break = candidate_points[best_index]

    maximum_f = statistics[best_index]

    return (
        maximum_f,
        best_break,
        candidate_points,
        statistics
    )


# ==========================================
# 4. BOOTSTRAP P-VALUE
# ==========================================

def bootstrap_test(
    y,
    X,
    observed_supf,
    repetitions=999,
    seed=42
):

    rng = np.random.default_rng(seed)

    # Estimate restricted model
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    fitted = X @ beta

    residuals = y - fitted

    # Center residuals
    residuals = residuals - residuals.mean()

    bootstrap_statistics = []

    for _ in range(repetitions):

        # Resample residuals under the null
        simulated_errors = rng.choice(
            residuals,
            size=len(y),
            replace=True
        )

        y_star = fitted + simulated_errors

        supf_star, _, _, _ = sup_f_test(
            y_star,
            X,
            trim=TRIM
        )

        bootstrap_statistics.append(supf_star)

    bootstrap_statistics = np.asarray(
        bootstrap_statistics
    )

    # Empirical bootstrap p-value
    p_value = (
        1 + np.sum(
            bootstrap_statistics >= observed_supf
        )
    ) / (repetitions + 1)

    return p_value, bootstrap_statistics


# ==========================================
# 5. ANALYSE ONE SCENARIO
# ==========================================

def analyse_scenario(name, seed):

    df = pd.read_csv(
        DATA_DIR / f"{name}.csv"
    )

    y = df["Y"].to_numpy()

    X = np.column_stack([
        np.ones(len(df)),
        df["X"].to_numpy()
    ])

    supf, estimated_break, points, profile = (
        sup_f_test(y, X, trim=TRIM)
    )

    p_value, bootstrap_stats = bootstrap_test(
        y,
        X,
        supf,
        repetitions=N_BOOTSTRAP,
        seed=seed
    )

    print("\n" + "=" * 50)
    print(name)

    print("Maximum F-statistic:", round(supf, 4))

    print(
        "Estimated break:",
        estimated_break
    )

    print(
        "Bootstrap p-value:",
        round(p_value, 6)
    )

    print(
        "Reject parameter stability:",
        p_value < 0.05
    )

    # Plot F-statistic profile
    plt.figure(figsize=(12, 5))

    plt.plot(
        points,
        profile,
        label="F-statistic profile"
    )

    plt.axvline(
        estimated_break,
        color="red",
        linestyle="--",
        label=f"Estimated break: {estimated_break}"
    )

    plt.xlabel("Candidate breakpoint")
    plt.ylabel("F-statistic")

    plt.title(
        f"Quandt-Andrews Test: {name}"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / f"quandt_andrews_{name}.png",
        dpi=300
    )

    plt.close()

    return {
        "scenario": name,
        "sup_f": supf,
        "estimated_break": estimated_break,
        "bootstrap_p_value": p_value,
        "reject_H0": p_value < 0.05
    }


# ==========================================
# 6. RUN EXPERIMENTS
# ==========================================

scenarios = [
    "S1_no_break",
    "S2_intercept_break",
    "S3_slope_break",
    "S4_multiple_breaks"
]

results = []

for i, scenario in enumerate(scenarios):

    result = analyse_scenario(
        scenario,
        SEED + i
    )

    results.append(result)


# ==========================================
# 7. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    TABLE_DIR / "quandt_andrews_results.csv",
    index=False
)

print("\nFINAL RESULTS")
print(results_df.to_string(index=False))

