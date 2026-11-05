
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

N = 300
SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "outputs" / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. COMMON DATA GENERATOR
# ==========================================

def generate_scenario(
    name,
    breakpoints,
    alphas,
    betas,
    sigmas,
    seed
):

    rng = np.random.default_rng(seed)

    # Independent variable
    X = rng.normal(10, 2, N)

    # Initialize arrays
    Y = np.zeros(N)
    regime = np.zeros(N, dtype=int)

    boundaries = [0] + breakpoints + [N]

    # Generate observations regime by regime
    for j in range(len(boundaries) - 1):

        start = boundaries[j]
        end = boundaries[j + 1]

        alpha = alphas[j]
        beta = betas[j]
        sigma = sigmas[j]

        errors = rng.normal(
            0,
            sigma,
            end - start
        )

        Y[start:end] = (
            alpha
            + beta * X[start:end]
            + errors
        )

        regime[start:end] = j + 1

    df = pd.DataFrame({
        "time": np.arange(1, N + 1),
        "X": X,
        "Y": Y,
        "regime": regime
    })

    # Save dataset
    file_path = DATA_DIR / f"{name}.csv"
    df.to_csv(file_path, index=False)

    # Plot observations
    plt.figure(figsize=(12, 5))

    plt.plot(
        df["time"],
        df["Y"],
        linewidth=1,
        label="Observed Y"
    )

    for i, point in enumerate(breakpoints):
        plt.axvline(
            point,
            linestyle="--",
            label=f"Break at {point}"
        )

    plt.title(name.replace("_", " ").title())
    plt.xlabel("Time")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / f"{name}.png",
        dpi=300
    )

    plt.close()

    print(f"{name} generated: {file_path}")

    return df


# ==========================================
# 3. SCENARIO 1: NO STRUCTURAL BREAK
# ==========================================

S1 = generate_scenario(
    name="S1_no_break",
    breakpoints=[],
    alphas=[2],
    betas=[0.5],
    sigmas=[1],
    seed=SEED
)


# ==========================================
# 4. SCENARIO 2: INTERCEPT BREAK
# ==========================================

S2 = generate_scenario(
    name="S2_intercept_break",
    breakpoints=[150],
    alphas=[2, 5],
    betas=[0.5, 0.5],
    sigmas=[1, 1],
    seed=SEED + 1
)


# ==========================================
# 5. SCENARIO 3: SLOPE BREAK
# ==========================================

S3 = generate_scenario(
    name="S3_slope_break",
    breakpoints=[150],
    alphas=[2, 2],
    betas=[0.5, 1.2],
    sigmas=[1, 1],
    seed=SEED + 2
)


# ==========================================
# 6. SCENARIO 4: TWO STRUCTURAL BREAKS
# ==========================================

S4 = generate_scenario(
    name="S4_multiple_breaks",
    breakpoints=[100, 200],
    alphas=[2, 5, 3],
    betas=[0.5, 1.2, 0.3],
    sigmas=[1, 1, 1],
    seed=SEED + 3
)


# ==========================================
# 7. SCENARIO 5: VARIANCE BREAK
# ==========================================

S5 = generate_scenario(
    name="S5_variance_break",
    breakpoints=[150],
    alphas=[2, 2],
    betas=[0.5, 0.5],
    sigmas=[0.5, 2],
    seed=SEED + 4
)


# ==========================================
# 8. SUMMARY
# ==========================================

print("\nALL SIMULATION SCENARIOS GENERATED")

for name, df in [
    ("S1", S1),
    ("S2", S2),
    ("S3", S3),
    ("S4", S4),
    ("S5", S5)
]:
    print(f"\n{name}: {len(df)} observations")

    print(
        df.groupby("regime")["Y"]
        .agg(["mean", "std"])
    )

