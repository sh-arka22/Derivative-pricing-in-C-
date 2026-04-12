#!/usr/bin/env python3
"""
QuantPricer — Bates Model Visualization

Reads CSV data from bates_pricer and produces 4 publication-quality plots:
  1. Jump intensity sweep (price vs lambda with confidence bands)
  2. Call prices across strikes (BS vs Heston vs Merton vs Bates)
  3. Implied volatility smile (BS vs Heston vs Merton vs Bates)
  4. Jump size impact on smile (mu_j variation)

Usage:
    ./bates_pricer                  # generate CSVs
    python3 tools/plot_bates.py     # render plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Style — match project dark theme
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

OUTPUT_DIR = "showcase_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette — project standard
C1 = "#00d4ff"   # cyan
C2 = "#ff6b6b"   # red
C3 = "#51cf66"   # green
C4 = "#ffd43b"   # yellow
C5 = "#cc5de8"   # purple
C6 = "#ff922b"   # orange

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"    saved {path}")


# ============================================================================
# 1. Jump Intensity Sweep — Price vs Lambda
# ============================================================================
def plot_lambda_sweep():
    df = pd.read_csv("bates_lambda_sweep.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: price with confidence band
    ax1.fill_between(df["Lambda"], df["ConfLo"], df["ConfHi"],
                     alpha=0.2, color=C1, label="95% CI")
    ax1.plot(df["Lambda"], df["BatesPrice"], "o-", color=C1,
             markersize=6, label="Bates price")
    ax1.axhline(df["HestonPrice"].iloc[0], color=C2, linestyle="--",
                alpha=0.7, label="Heston (no jumps)")
    ax1.set_xlabel(r"Jump intensity $\lambda$ (jumps/year)")
    ax1.set_ylabel("Call Price ($)")
    ax1.set_title(r"Bates Call Price vs Jump Intensity")
    ax1.legend()

    # Right: price premium over Heston
    premium = df["BatesPrice"] - df["HestonPrice"]
    ax2.bar(df["Lambda"], premium, width=0.3, color=C5, alpha=0.8,
            edgecolor="white", linewidth=0.5)
    ax2.set_xlabel(r"Jump intensity $\lambda$ (jumps/year)")
    ax2.set_ylabel("Price Premium over Heston ($)")
    ax2.set_title("Jump Risk Premium")

    fig.suptitle(
        r"Bates Model — Effect of Jump Intensity  |  $\mu_J=-0.05,\ \sigma_J=0.1$",
        fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, "12_bates_lambda_sweep")


# ============================================================================
# 2 & 3. Model Smile Comparison — Prices + Implied Vols
# ============================================================================
def plot_smile_comparison():
    df = pd.read_csv("bates_smile.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Call prices
    ax1.plot(df["Strike"], df["BSPrice"], "--", color="white", alpha=0.5,
             linewidth=1.5, label="Black-Scholes")
    ax1.plot(df["Strike"], df["HestonPrice"], "o-", color=C1,
             markersize=4, label=r"Heston ($\rho=-0.7$)")
    ax1.plot(df["Strike"], df["MertonPrice"], "s-", color=C2,
             markersize=4, label=r"Merton ($\lambda=1$)")
    ax1.plot(df["Strike"], df["BatesPrice"], "D-", color=C3,
             markersize=4, label="Bates (Heston+jumps)")
    ax1.axvline(100, color="white", alpha=0.2, linestyle="--")
    ax1.set_xlabel("Strike")
    ax1.set_ylabel("Call Price ($)")
    ax1.set_title("Call Prices Across Models")
    ax1.legend()

    # Right: Implied volatility smile
    ax2.axhline(df["BSIV"].iloc[0] * 100, color="white", alpha=0.3,
                linestyle="--", label="BS flat vol = 20%")
    ax2.plot(df["Strike"], df["HestonIV"] * 100, "o-", color=C1,
             markersize=4, label="Heston IV")
    ax2.plot(df["Strike"], df["MertonIV"] * 100, "s-", color=C2,
             markersize=4, label="Merton IV")
    ax2.plot(df["Strike"], df["BatesIV"] * 100, "D-", color=C3,
             markersize=4, label="Bates IV")
    ax2.axvline(100, color="white", alpha=0.2, linestyle="--")
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Implied Volatility (%)")
    ax2.set_title("Implied Volatility Smile by Model")
    ax2.legend()

    fig.suptitle(
        "Model Comparison — BS vs Heston vs Merton vs Bates  |  S=100, T=1y",
        fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, "13_bates_smile_comparison")


# ============================================================================
# 4. Jump Size Impact on Smile Shape
# ============================================================================
def plot_jump_size_impact():
    df = pd.read_csv("bates_jump_size.csv")

    fig, ax = plt.subplots(figsize=(12, 7))

    mu_labels = [
        (r"$\mu_J = -0.15$", C2),
        (r"$\mu_J = -0.10$", C6),
        (r"$\mu_J = -0.05$", C4),
        (r"$\mu_J = 0.00$", "white"),
        (r"$\mu_J = +0.05$", C3),
        (r"$\mu_J = +0.10$", C1),
    ]

    for i, (label, color) in enumerate(mu_labels):
        col = df.columns[i + 1]  # skip Strike column
        ax.plot(df["Strike"], df[col] * 100, "o-", color=color,
                markersize=4, label=label, alpha=0.9)

    ax.axhline(20, color="white", alpha=0.2, linestyle=":",
               label="BS flat vol")
    ax.axvline(100, color="white", alpha=0.2, linestyle="--")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(r"Bates IV Smile — Effect of Mean Jump Size $\mu_J$")
    ax.legend(ncol=2)

    fig.suptitle(
        r"Jump Direction Controls Skew  |  $\lambda=1,\ \sigma_J=0.1,\ \rho=-0.7$",
        fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, "14_bates_jump_size_impact")


# ============================================================================
def main():
    print("=" * 60)
    print("  QuantPricer — Bates Model Plots")
    print("=" * 60)

    plots = [
        ("Lambda Sweep",       "bates_lambda_sweep.csv", plot_lambda_sweep),
        ("Smile Comparison",   "bates_smile.csv",        plot_smile_comparison),
        ("Jump Size Impact",   "bates_jump_size.csv",    plot_jump_size_impact),
    ]

    for name, csv_file, plot_fn in plots:
        if os.path.exists(csv_file):
            print(f"  {name}...")
            plot_fn()
        else:
            print(f"  {name} — SKIPPED (no {csv_file})")

    print(f"\n  Plots saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
