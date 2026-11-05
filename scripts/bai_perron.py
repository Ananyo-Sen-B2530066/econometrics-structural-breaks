
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ==========================================
# 1. CONFIGURATION
# ==========================================

MAX_BREAKS = 4
MIN_SEGMENT = 30

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "outputs" / "figures"
TABLE_DIR = BASE_DIR / "outputs" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. SEGMENT SSR USING OLS
# ==========================================

def segment_ssr(y, x, start, end):

    # Segment uses Python indexing [start:end]
    ys = y[start:end]
    xs = x[start:end]

    X = np.column_stack([
        np.ones(len(xs)),
        xs
    ])

    beta = np.linalg.lstsq(
        X,
        ys,
        rcond=None
    )[0]

    residuals = ys - X @ beta

    return np.sum(residuals ** 2)


# ==========================================
# 3. DYNAMIC PROGRAMMING
# ==========================================

def optimal_segmentation(
    y,
    x,
    max_breaks=4,
    min_segment=30
):

    n = len(y)

    max_segments = max_breaks + 1

    # Precompute SSR for all valid segments
    cost = np.full((n + 1, n + 1), np.inf)

    for start in range(n):

        for end in range(
            start + min_segment,
            n + 1
        ):

            cost[start, end] = segment_ssr(
                y,
                x,
                start,
                end
            )

    # DP table:
    # dp[k, t] = minimum SSR for k segments
    # covering the first t observations

    dp = np.full(
        (max_segments + 1, n + 1),
        np.inf
    )

    previous = np.full(
        (max_segments + 1, n + 1),
        -1,
        dtype=int
    )

    dp[0, 0] = 0

    for k in range(1, max_segments + 1):

        for t in range(
            k * min_segment,
            n + 1
        ):

            for s in range(
                (k - 1) * min_segment,
                t - min_segment + 1
            ):

                if not np.isfinite(dp[k - 1, s]):
                    continue

                candidate = (
                    dp[k - 1, s]
                    + cost[s, t]
                )

                if candidate < dp[k, t]:

                    dp[k, t] = candidate
                    previous[k, t] = s

    # Recover optimal breakpoints
    solutions = {}

    for k in range(1, max_segments + 1):

        if not np.isfinite(dp[k, n]):
            continue

        t = n
        boundaries = []

        for j in range(k, 0, -1):

            s = previous[j, t]

            if s < 0:
                break

            boundaries.append(s)
            t = s

        boundaries.reverse()

        # Remove starting boundary (0)
        breakpoints = boundaries[1:]

        solutions[k - 1] = {
            "SSR": dp[k, n],
            "breakpoints": breakpoints
        }

    return solutions


# ==========================================
# 4. SELECT MODEL USING BIC
# ==========================================

def calculate_bic(ssr, n, number_of_segments):

    # Two coefficients per segment:
    # intercept and slope.
    #
    # One residual variance parameter is shared.
    # The number of breakpoints is also penalized.

    k = (
        2 * number_of_segments
        + 1
        + number_of_segments - 1
    )

    return (
        n * np.log(ssr / n)
        + k * np.log(n)
    )


def select_best_model(solutions, n):

    results = []

    for breaks, result in solutions.items():

        segments = breaks + 1

        bic = calculate_bic(
            result["SSR"],
            n,
            segments
        )

        results.append({
            "breaks": breaks,
            "SSR": result["SSR"],
            "BIC": bic,
            "breakpoints": result["breakpoints"]
        })

    results_df = pd.DataFrame(results)

    best_index = results_df["BIC"].idxmin()

    best_model = results_df.loc[best_index]

    return results_df, best_model


# ==========================================
# 5. ANALYSE SCENARIO
# ==========================================

def analyse_scenario(name):

    df = pd.read_csv(
        DATA_DIR / f"{name}.csv"
    )

    y = df["Y"].to_numpy()
    x = df["X"].to_numpy()

    solutions = optimal_segmentation(
        y,
        x,
        max_breaks=MAX_BREAKS,
        min_segment=MIN_SEGMENT
    )

    results_df, best = select_best_model(
        solutions,
        len(y)
    )

    print("\n" + "=" * 55)
    print(name)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    print("\nSelected number of breaks:")
    print(int(best["breaks"]))

    print("\nEstimated breakpoints:")
    print(best["breakpoints"])

    # ======================================
    # Plot estimated breakpoints
    # ======================================

    plt.figure(figsize=(13, 5))

    plt.plot(
        df["time"],
        y,
        linewidth=1,
        label="Observed Y"
    )

    for point in best["breakpoints"]:

        plt.axvline(
            point,
            color="red",
            linestyle="--"
        )

    plt.title(
        f"Multiple Structural Breaks: {name}"
    )

    plt.xlabel("Time")
    plt.ylabel("Y")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / f"bai_perron_{name}.png",
        dpi=300
    )

    plt.close()

    return results_df.assign(scenario=name)


# ==========================================
# 6. RUN ALL SCENARIOS
# ==========================================

scenarios = [
    "S1_no_break",
    "S2_intercept_break",
    "S3_slope_break",
    "S4_multiple_breaks",
    "S5_variance_break"
]

all_results = []

for scenario in scenarios:

    result = analyse_scenario(scenario)

    all_results.append(result)


# ==========================================
# 7. SAVE RESULTS
# ==========================================

final_results = pd.concat(
    all_results,
    ignore_index=True
)

final_results.to_csv(
    TABLE_DIR / "bai_perron_results.csv",
    index=False
)

print("\nAll scenarios analysed successfully.")

