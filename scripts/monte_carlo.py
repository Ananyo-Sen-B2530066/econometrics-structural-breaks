import time
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import f


# ==================================================
# 1. CONFIGURATION
# ==================================================

REPS = 1000
N = 300
ALPHA = 0.05
MIN_SEGMENT = 30
MAX_BREAKS = 4
SEED = 2026

BASE = Path(__file__).resolve().parents[1]

RESULT_DIR = BASE / "results"
TABLE_DIR = BASE / "outputs" / "tables"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. DATA GENERATING PROCESS
# ==================================================

def generate_data(scenario, rng, n=N):

    X = rng.normal(10, 2, n)

    if scenario == "S1_no_break":
        breaks = []
        alphas = [2]
        betas = [0.5]
        sigmas = [1]

    elif scenario == "S2_intercept_break":
        breaks = [n // 2]
        alphas = [2, 5]
        betas = [0.5, 0.5]
        sigmas = [1, 1]

    elif scenario == "S3_slope_break":
        breaks = [n // 2]
        alphas = [2, 2]
        betas = [0.5, 1.2]
        sigmas = [1, 1]

    elif scenario == "S4_multiple_breaks":
        breaks = [n // 3, 2 * n // 3]
        alphas = [2, 5, 3]
        betas = [0.5, 1.2, 0.3]
        sigmas = [1, 1, 1]

    elif scenario == "S5_variance_break":
        breaks = [n // 2]
        alphas = [2, 2]
        betas = [0.5, 0.5]
        sigmas = [0.5, 2]

    else:
        raise ValueError("Unknown scenario")

    boundaries = [0] + breaks + [n]

    Y = np.zeros(n)

    for j in range(len(boundaries) - 1):

        start, end = boundaries[j], boundaries[j + 1]

        errors = rng.normal(
            0,
            sigmas[j],
            end - start
        )

        Y[start:end] = (
            alphas[j]
            + betas[j] * X[start:end]
            + errors
        )

    return X, Y, breaks


# ==================================================
# 3. REGRESSION SSR
# ==================================================

def regression_ssr(y, x):

    X = np.column_stack([
        np.ones(len(x)),
        x
    ])

    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    residuals = y - X @ beta

    return np.sum(residuals ** 2), residuals


# ==================================================
# 4. CHOW TEST
# ==================================================

def chow_test(y, x, point):

    n = len(y)
    k = 2

    ssr_r, _ = regression_ssr(y, x)

    ssr_1, _ = regression_ssr(
        y[:point], x[:point]
    )

    ssr_2, _ = regression_ssr(
        y[point:], x[point:]
    )

    ssr_u = ssr_1 + ssr_2

    numerator = max(0, ssr_r - ssr_u) / k

    denominator = ssr_u / (n - 2 * k)

    if denominator <= 0:
        return np.nan

    statistic = numerator / denominator

    return f.sf(
        statistic,
        k,
        n - 2 * k
    )


# # ==================================================
# # 5. CUSUM TEST
# # ==================================================

# def cusum_test(y, x):

#     X = np.column_stack([
#         np.ones(len(x)),
#         x
#     ])

#     beta = np.linalg.lstsq(X, y, rcond=None)[0]

#     residuals = y - X @ beta

#     statistic, p_value, critical = (
#         breaks_cusumolsresid(
#             residuals,
#             ddof=2
#         )
#     )

#     return p_value


# # ==================================================
# # 6. QUANDT-ANDREWS SUP-F
# # ==================================================

# def sup_f_statistic(y, x, trim=0.15):

#     n = len(y)
#     k = 2

#     ssr_r, _ = regression_ssr(y, x)

#     lower = int(np.ceil(n * trim))
#     upper = int(np.floor(n * (1 - trim)))

#     maximum = 0

#     for point in range(lower, upper + 1):

#         ssr_1, _ = regression_ssr(
#             y[:point], x[:point]
#         )

#         ssr_2, _ = regression_ssr(
#             y[point:], x[point:]
#         )

#         ssr_u = ssr_1 + ssr_2

#         denominator = ssr_u / (n - 2 * k)

#         if denominator <= 0:
#             continue

#         statistic = (
#             max(0, ssr_r - ssr_u) / k
#         ) / denominator

#         maximum = max(maximum, statistic)

#     return maximum


# ==================================================
# 7. BAI-PERRON-STYLE DYNAMIC PROGRAMMING
# ==================================================

def bai_perron_dp(
    y,
    x,
    max_breaks=MAX_BREAKS,
    min_segment=MIN_SEGMENT
):

    n = len(y)

    # Prefix sums for fast segment SSR calculations
    px = np.r_[0, np.cumsum(x)]
    py = np.r_[0, np.cumsum(y)]
    pxx = np.r_[0, np.cumsum(x * x)]
    pxy = np.r_[0, np.cumsum(x * y)]
    pyy = np.r_[0, np.cumsum(y * y)]

    cost = np.full((n + 1, n + 1), np.inf)

    for end in range(min_segment, n + 1):

        starts = np.arange(
            0,
            end - min_segment + 1
        )

        count = end - starts

        sx = px[end] - px[starts]
        sy = py[end] - py[starts]
        sxx = pxx[end] - pxx[starts]
        sxy = pxy[end] - pxy[starts]
        syy = pyy[end] - pyy[starts]

        determinant = count * sxx - sx * sx

        numerator = (
            sxx * sy * sy
            - 2 * sx * sy * sxy
            + count * sxy * sxy
        )

        ssr = np.where(
            determinant > 1e-12,
            syy - numerator / np.maximum(
                determinant, 1e-12
            ),
            np.inf
        )

        cost[starts, end] = np.maximum(ssr, 0)

    # Dynamic programming
    segments_max = max_breaks + 1

    dp = np.full(
        (segments_max + 1, n + 1),
        np.inf
    )

    previous = np.full(
        (segments_max + 1, n + 1),
        -1,
        dtype=int
    )

    dp[0, 0] = 0

    for k in range(1, segments_max + 1):

        for end in range(
            k * min_segment,
            n + 1
        ):

            starts = np.arange(
                (k - 1) * min_segment,
                end - min_segment + 1
            )

            values = (
                dp[k - 1, starts]
                + cost[starts, end]
            )

            best = np.argmin(values)

            dp[k, end] = values[best]
            previous[k, end] = starts[best]

    # Select model using BIC
    candidates = []

    for k in range(1, segments_max + 1):

        if not np.isfinite(dp[k, n]):
            continue

        m = k - 1

        # Regression coefficients + break locations
        parameters = 2 * k + m + 1

        bic = (
            n * np.log(max(dp[k, n], 1e-12) / n)
            + parameters * np.log(n)
        )

        # Recover breakpoints
        end = n
        points = []

        for j in range(k, 0, -1):

            start = previous[j, end]

            if start < 0:
                break

            points.append(start)
            end = start

        points.reverse()

        points = points[1:]

        candidates.append({
            "breaks": m,
            "bic": bic,
            "points": points
        })

    return min(candidates, key=lambda z: z["bic"])


# # ==================================================
# # 8. PELT
# # ==================================================

# def pelt_detection(y, x, mode="regression"):

#     n = len(y)

#     if mode == "regression":

#         signal = np.column_stack([
#             y,
#             np.ones(n),
#             x
#         ])

#         ssr, _ = regression_ssr(y, x)

#         variance = ssr / (n - 2)

#         penalty = 3 * variance * np.log(n)

#         model = "linear"

#     else:

#         _, residuals = regression_ssr(y, x)

#         signal = (residuals ** 2).reshape(-1, 1)

#         variance = np.var(signal)

#         penalty = 3 * variance * np.log(n)

#         model = "l2"

#     algorithm = rpt.Pelt(
#         model=model,
#         min_size=MIN_SEGMENT,
#         jump=1
#     )

#     algorithm.fit(signal)

#     endpoints = algorithm.predict(
#         pen=float(penalty)
#     )

#     return endpoints[:-1]


# ==================================================
# 9. BREAKPOINT ERROR
# ==================================================

def breakpoint_metrics(estimated, actual):

    correct_count = len(estimated) == len(actual)

    if not correct_count:
        return correct_count, np.nan

    if len(actual) == 0:
        return True, 0.0

    error = np.mean([
        abs(a - b)
        for a, b in zip(estimated, actual)
    ])

    return correct_count, error


# # ==================================================
# # 10. NULL CALIBRATION FOR QUANDT-ANDREWS
# # ==================================================

# def calibrate_supf(
#     repetitions=999,
#     seed=50000
# ):

#     rng = np.random.default_rng(seed)

#     statistics = []

#     for _ in range(repetitions):

#         x = rng.normal(10, 2, N)
#         errors = rng.normal(0, 1, N)

#         y = 2 + 0.5 * x + errors

#         statistics.append(
#             sup_f_statistic(y, x)
#         )

#     return np.quantile(statistics, 0.95)


# ==================================================
# 10. MONTE CARLO EXPERIMENT
# ==================================================

def run_monte_carlo():

    scenarios = [
        "S1_no_break",
        "S2_intercept_break",
        "S3_slope_break",
        "S4_multiple_breaks",
        "S5_variance_break"
    ]

    records = []

    for scenario in scenarios:

        print("\nRunning:", scenario)

        for repetition in range(REPS):

            rng = np.random.default_rng(
                SEED + repetition
                + scenarios.index(scenario) * 100000
            )

            x, y, actual = generate_data(
                scenario, rng
            )

            # ----------------------------------
            # CHOW TEST
            # ----------------------------------

            if scenario == "S5_variance_break":

                # Classical Chow test is not
                # appropriate for this variance-break DGP.
                chow_rejection_rate = np.nan
                chow_joint_rejection = np.nan

            else:

                # For S1, use midpoint as a placebo date.
                chow_points = actual if actual else [N // 2]

                chow_pvalues = [
                    chow_test(y, x, point)
                    for point in chow_points
                ]

                chow_rejected = [
                    p < ALPHA
                    for p in chow_pvalues
                    if np.isfinite(p)
                ]

                chow_rejection_rate = (
                    np.mean(chow_rejected)
                    if chow_rejected else np.nan
                )

                chow_joint_rejection = (
                    all(chow_rejected)
                    if chow_rejected else np.nan
                )

            # ----------------------------------
            # BAI-PERRON-STYLE METHOD
            # ----------------------------------

            # S5 has no regression coefficient break.
            # Its variance change must not be counted
            # as a correctly detected regression break.

            actual_regression_breaks = (
                [] if scenario == "S5_variance_break"
                else actual
            )

            start = time.perf_counter()

            bai = bai_perron_dp(
                y,
                x,
                max_breaks=MAX_BREAKS,
                min_segment=MIN_SEGMENT
            )

            bai_time = time.perf_counter() - start

            bai_correct, bai_error = breakpoint_metrics(
                bai["points"],
                actual_regression_breaks
            )

            # ----------------------------------
            # STORE RESULTS
            # ----------------------------------

            records.append({

                "scenario": scenario,
                "repetition": repetition + 1,

                "chow_rejection_rate": chow_rejection_rate,
                "chow_joint_rejection": chow_joint_rejection,

                "bai_break_count": bai["breaks"],
                "bai_correct_count": bai_correct,
                "bai_mae": bai_error,
                "bai_time": bai_time

            })

        print(
            f"Completed {REPS} repetitions for {scenario}"
        )

    raw = pd.DataFrame(records)

    raw.to_csv(
        RESULT_DIR / "monte_carlo_raw.csv",
        index=False
    )

    return raw

# ==================================================
# WILSON CONFIDENCE INTERVAL
# ==================================================

def wilson_interval(successes, total, z=1.96):

    if total == 0:
        return np.nan, np.nan

    p = successes / total

    denominator = 1 + (z ** 2 / total)

    centre = (
        p + z ** 2 / (2 * total)
    ) / denominator

    margin = (
        z
        * np.sqrt(
            (p * (1 - p) / total)
            + (z ** 2 / (4 * total ** 2))
        )
        / denominator
    )

    return (
        max(0, centre - margin),
        min(1, centre + margin)
    )

# ==================================================
# 11. PERFORMANCE SUMMARY
# ==================================================

def summarize_results(raw):

    summaries = []

    for scenario, group in raw.groupby("scenario"):

        summaries.append({

            "scenario": scenario,

            "Chow_rejection_rate":
                group["chow_rejection_rate"].mean(),

            "Chow_joint_rejection_rate":
                group["chow_joint_rejection"].mean(),

            "Bai_correct_count_rate":
                group["bai_correct_count"].mean(),

            "Bai_breakpoint_MAE":
                group["bai_mae"].mean(),

            "Bai_average_time":
                group["bai_time"].mean()

        })

    summary = pd.DataFrame(summaries)

    # ----------------------------------
    # WILSON CONFIDENCE INTERVALS
    # ----------------------------------

    rate_mapping = {
    "Chow_rejection_rate": "chow_rejection_rate",
    "Chow_joint_rejection_rate": "chow_joint_rejection",
    "Bai_correct_count_rate": "bai_correct_count"
}

    for summary_column, raw_column in rate_mapping.items():

        lower_values = []
        upper_values = []

        for scenario in summary["scenario"]:

            valid = raw.loc[
                raw["scenario"] == scenario,
                raw_column
            ].dropna()

            successes = int(valid.sum())
            total = len(valid)

            lower, upper = wilson_interval(
                successes,
                total
            )

            lower_values.append(lower)
            upper_values.append(upper)

        summary[summary_column + "_CI_Lower"] = lower_values
        summary[summary_column + "_CI_Upper"] = upper_values

    summary.to_csv(
        TABLE_DIR / "monte_carlo_summary.csv",
        index=False
    )

    print("\nMONTE CARLO SUMMARY")

    print(
        summary.to_string(index=False)
    )

    return summary


# ==================================================
# 13. MAIN
# ==================================================

if __name__ == "__main__":

    raw = run_monte_carlo()

    summary = summarize_results(raw)

    print("\nMONTE CARLO EXPERIMENT COMPLETED")

