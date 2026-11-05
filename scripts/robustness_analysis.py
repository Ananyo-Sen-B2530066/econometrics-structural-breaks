
import numpy as np
import pandas as pd
import time

from pathlib import Path

from monte_carlo import (
    bai_perron_dp,
    breakpoint_metrics
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

REPS = 200
SEED = 7000

BASE = Path(__file__).resolve().parents[1]

RESULT_DIR = BASE / "results"
TABLE_DIR = BASE / "outputs" / "tables"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CUSTOM DGP
# ==========================================

def generate_custom_data(
    n,
    sigma,
    magnitude,
    location,
    rng
):

    X = rng.normal(10, 2, n)

    breakpoint = int(n * location)

    errors = rng.normal(0, sigma, n)

    Y = np.zeros(n)

    Y[:breakpoint] = (
        2 + 0.5 * X[:breakpoint]
        + errors[:breakpoint]
    )

    Y[breakpoint:] = (
        (2 + magnitude)
        + 0.5 * X[breakpoint:]
        + errors[breakpoint:]
    )

    return X, Y, breakpoint


# ==========================================
# 3. EXPERIMENT CONFIGURATIONS
# ==========================================

experiments = []

# R1: Sample size
for n in [100, 300, 500]:

    experiments.append({
        "experiment": "Sample_Size",
        "n": n,
        "sigma": 1,
        "magnitude": 3,
        "location": 0.5
    })


# R2: Error variance
for sigma in [0.5, 1, 2, 3]:

    experiments.append({
        "experiment": "Error_Variance",
        "n": 300,
        "sigma": sigma,
        "magnitude": 3,
        "location": 0.5
    })


# R3: Break magnitude
for magnitude in [0.5, 1, 2, 4]:

    experiments.append({
        "experiment": "Break_Magnitude",
        "n": 300,
        "sigma": 1,
        "magnitude": magnitude,
        "location": 0.5
    })


# R4: Break location
for location in [0.25, 0.50, 0.75]:

    experiments.append({
        "experiment": "Break_Location",
        "n": 300,
        "sigma": 1,
        "magnitude": 3,
        "location": location
    })


# ==========================================
# 4. RUN ONE EXPERIMENT
# ==========================================

def run_experiment(config, experiment_id):

    records = []

    for repetition in range(REPS):

        rng = np.random.default_rng(
            SEED
            + experiment_id * 10000
            + repetition
        )

        X, Y, actual = generate_custom_data(
            n=config["n"],
            sigma=config["sigma"],
            magnitude=config["magnitude"],
            location=config["location"],
            rng=rng
        )

        # ------------------------------
        # BAI-PERRON-STYLE DYNAMIC PROGRAMMING
        # ------------------------------

        min_segment = max(
            15,
            int(config["n"] * 0.10)
        )

        start = time.perf_counter()

        bai = bai_perron_dp(
            Y,
            X,
            max_breaks=4,
            min_segment=min_segment
        )

        bai_time = time.perf_counter() - start

        bai_correct, bai_error = breakpoint_metrics(
            bai["points"],
            [actual]
        )

    #     # ------------------------------
    #     # PELT
    #     # ------------------------------

    #     start = time.perf_counter()

    #     pelt_points = pelt_detection(
    #         Y,
    #         X,
    #         mode="regression"
    #     )

    #     pelt_time = time.perf_counter() - start

    #     pelt_correct, pelt_error = breakpoint_metrics(
    #         pelt_points,
    #         [actual]
    #     )

        records.append({

            "experiment": config["experiment"],

            "n": config["n"],
            "sigma": config["sigma"],
            "magnitude": config["magnitude"],
            "location": config["location"],

            "repetition": repetition + 1,

            "actual_break": actual,

            "bai_estimated_breaks": bai["points"],
            "bai_correct": bai_correct,
            "bai_error": bai_error,
            "bai_time": bai_time,

    #         "pelt_estimated_breaks": list(pelt_points),
    #         "pelt_correct": pelt_correct,
    #         "pelt_error": pelt_error,
    #         "pelt_time": pelt_time

        })

    return records


# ==========================================
# 5. RUN ALL EXPERIMENTS
# ==========================================

all_records = []

for i, config in enumerate(experiments):

    print("\n" + "=" * 55)

    print("Experiment:", config["experiment"])
    print("Configuration:", config)

    records = run_experiment(config, i)

    all_records.extend(records)

    print(
        f"Completed {REPS} repetitions."
    )


# ==========================================
# 6. SAVE RAW RESULTS
# ==========================================

raw = pd.DataFrame(all_records)

raw.to_csv(
    RESULT_DIR / "robustness_raw.csv",
    index=False
)


# ==========================================
# 7. SUMMARIZE RESULTS
# ==========================================

summary = (
    raw.groupby([
        "experiment",
        "n",
        "sigma",
        "magnitude",
        "location"
    ])
    .agg(

        Bai_Correct_Count_Rate=(
            "bai_correct",
            "mean"
        ),

        Bai_Conditional_MAE=(
            "bai_error",
            "mean"
        ),

        Bai_Average_Time=(
            "bai_time",
            "mean"
        )

    )
    .reset_index()
)


# ==========================================
# 8. SAVE SUMMARY
# ==========================================

summary.to_csv(
    TABLE_DIR / "robustness_summary.csv",
    index=False
)

print("\nROBUSTNESS SUMMARY")

print(
    summary.to_string(index=False)
)

print("\nROBUSTNESS ANALYSIS COMPLETED")

