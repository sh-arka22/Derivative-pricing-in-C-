#!/usr/bin/env python3
"""
QuantPricer — Showcase Renderer
Generates all publication-quality plots for the GitHub README.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.titlesize": 14,
    "axes.labelsize": 12, "legend.fontsize": 10, "lines.linewidth": 2,
    "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight",
})

OUT = "showcase_plots"
os.makedirs(OUT, exist_ok=True)

C1, C2, C3, C4, C5 = "#00d4ff", "#ff6b6b", "#51cf66", "#ffd43b", "#cc5de8"

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print(f"    {name}.png")

# ============================================================================
# 1. Vol Surface — 3 rho variations
# ============================================================================
def plot_vol_surfaces():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': '3d'})
    rhos = [-0.9, -0.5, 0.0]
    labels = [r'$\rho = -0.9$ (strong skew)', r'$\rho = -0.5$ (moderate)', r'$\rho = 0.0$ (symmetric)']

    for ax, rho, label in zip(axes, rhos, labels):
        tag = str(int(rho * 10))
        df = pd.read_csv(f"showcase_data/vol_surface_rho_{tag}.csv")
        strikes = sorted(df['Strike'].unique())
        mats = sorted(df['Maturity'].unique())
        K, T = np.meshgrid(strikes, mats)
        IV = np.full_like(K, np.nan, dtype=float)
        for i, t in enumerate(mats):
            for j, k in enumerate(strikes):
                row = df[(df['Maturity']==t) & (df['Strike']==k)]
                if not row.empty:
                    IV[i,j] = row['ImpliedVol'].values[0]
        ax.plot_surface(K, T, IV*100, cmap=cm.plasma, alpha=0.9, edgecolor='white', linewidth=0.15)
        ax.set_xlabel('Strike'); ax.set_ylabel('T (years)'); ax.set_zlabel('IV (%)')
        ax.set_title(label, fontsize=12)
        ax.view_init(elev=25, azim=-45)

    fig.suptitle('Heston Implied Volatility Surface — Effect of Correlation', fontsize=16, y=1.02)
    plt.tight_layout()
    save(fig, "01_vol_surface_3rho")

    # 2D smile slices for rho=-0.7 (most realistic)
    df = pd.read_csv("showcase_data/vol_surface_rho_-5.csv")
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [C1, C2, C3, C4, C5, '#ff922b']
    for i, t in enumerate(sorted(df['Maturity'].unique())):
        sub = df[df['Maturity']==t].sort_values('Strike')
        ax.plot(sub['Strike'], sub['ImpliedVol']*100, 'o-', color=colors[i%len(colors)],
                label=f'T={t:.2f}y', markersize=4)
    ax.axvline(100, color='white', alpha=0.3, ls='--', label='ATM')
    ax.set_xlabel('Strike'); ax.set_ylabel('IV (%)'); ax.legend()
    ax.set_title(r'Heston Vol Smile by Maturity ($\rho=-0.5$)')
    save(fig, "02_vol_smile")

# ============================================================================
# 2. Greeks Dashboard
# ============================================================================
def plot_greeks():
    df = pd.read_csv("showcase_data/greeks.csv")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Black-Scholes Greeks Dashboard  |  K=100, r=5%, T=1y, σ=20%', fontsize=16, y=0.98)

    plots = [
        (0,0, 'Option Price', [('CallPrice',C1,'Call'),('PutPrice',C2,'Put')]),
        (0,1, r'Delta ($\Delta$)', [('DeltaCall',C1,'Call'),('DeltaPut',C2,'Put')]),
        (0,2, r'Gamma ($\Gamma$)', [('Gamma',C3,None)]),
        (1,0, r'Vega ($\nu$)', [('Vega',C4,None)]),
        (1,1, r'Theta ($\Theta$)', [('ThetaCall',C1,'Call'),('ThetaPut',C2,'Put')]),
    ]
    for r,c,title,series in plots:
        ax = axes[r][c]
        for col,color,label in series:
            ax.plot(df['Spot'], df[col], color=color, label=label)
            if len(series)==1:
                ax.fill_between(df['Spot'], 0, df[col], alpha=0.1, color=color)
        ax.set_title(title); ax.set_xlabel('Spot')
        if any(s[2] for s in series): ax.legend()

    # Payoff profiles in last panel
    ax = axes[1][2]
    S = df['Spot']
    ax.plot(S, np.maximum(S-100, 0), ':', color='white', alpha=0.4, label='Call payoff')
    ax.plot(S, df['CallPrice'], color=C1, label='Call value')
    ax.fill_between(S, 0, df['CallPrice'], alpha=0.1, color=C1)
    ax.set_title('Call: Value vs Payoff'); ax.set_xlabel('Spot'); ax.legend()
    plt.tight_layout()
    save(fig, "03_greeks_dashboard")

# ============================================================================
# 3. MC Convergence — 3 vol levels
# ============================================================================
def plot_mc_convergence():
    df = pd.read_csv("showcase_data/mc_convergence.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = {0.15: C3, 0.25: C1, 0.40: C2}

    for sigma in df['Sigma'].unique():
        sub = df[df['Sigma']==sigma]
        color = colors.get(sigma, 'white')
        ax1.fill_between(sub['NumPaths'], sub['ConfLo'], sub['ConfHi'], alpha=0.15, color=color)
        ax1.plot(sub['NumPaths'], sub['MCPrice'], 'o-', color=color, markersize=2,
                 label=f'σ={sigma*100:.0f}%')
        ax1.axhline(sub['Analytic'].iloc[0], color=color, ls='--', alpha=0.5)
    ax1.set_xscale('log'); ax1.set_xlabel('Paths'); ax1.set_ylabel('Price ($)')
    ax1.set_title('MC Price Convergence (3 vol levels)'); ax1.legend()

    for sigma in df['Sigma'].unique():
        sub = df[df['Sigma']==sigma]
        color = colors.get(sigma, 'white')
        ax2.plot(sub['NumPaths'], sub['StdError'], 'o-', color=color, markersize=2,
                 label=f'σ={sigma*100:.0f}%')
    N = df['NumPaths'].unique().astype(float)
    ax2.plot(N, 0.8/np.sqrt(N), '--', color='white', alpha=0.4, label=r'$O(1/\sqrt{N})$')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('Paths'); ax2.set_ylabel('Std Error'); ax2.set_title('Convergence Rate (log-log)')
    ax2.legend()
    fig.suptitle('Monte Carlo Convergence — Antithetic Variates', fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, "04_mc_convergence")

# ============================================================================
# 4. Model Comparison
# ============================================================================
def plot_model_comparison():
    df = pd.read_csv("showcase_data/model_comparison.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.plot(df['Strike'], df['BSPrice'], '--', color='white', alpha=0.5, label='Black-Scholes')
    ax1.plot(df['Strike'], df['HestonPrice'], 'o-', color=C1, ms=4, label=r'Heston ($\rho$=-0.7)')
    ax1.plot(df['Strike'], df['MertonPrice'], 's-', color=C2, ms=4, label='Merton (jumps)')
    ax1.set_xlabel('Strike'); ax1.set_ylabel('Price ($)'); ax1.set_title('Call Prices'); ax1.legend()

    ax2.axhline(20, color='white', alpha=0.3, ls='--', label='BS flat vol')
    hiv = df['HestonIV'].replace(0, np.nan)*100
    miv = df['MertonIV'].replace(0, np.nan)*100
    ax2.plot(df['Strike'], hiv, 'o-', color=C1, ms=4, label='Heston IV')
    ax2.plot(df['Strike'], miv, 's-', color=C2, ms=4, label='Merton IV')
    ax2.set_xlabel('Strike'); ax2.set_ylabel('IV (%)'); ax2.set_title('Implied Vol Smile'); ax2.legend()

    fig.suptitle('Model Comparison — BS vs Heston vs Merton', fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, "05_model_comparison")

# ============================================================================
# 5. Barrier Options — 3 vol levels
# ============================================================================
def plot_barrier():
    df = pd.read_csv("showcase_data/barrier.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = {0.15: C3, 0.25: C1, 0.35: C2}

    for sigma in df['Sigma'].unique():
        sub = df[df['Sigma']==sigma]
        color = colors.get(sigma, 'white')
        ax1.plot(sub['Barrier'], sub['DOCall'], color=color, label=f'DO σ={sigma*100:.0f}%')
        ax1.plot(sub['Barrier'], sub['DICall'], '--', color=color, alpha=0.6)
    ax1.axhline(df['Vanilla'].iloc[0], color='white', alpha=0.3, ls='--')
    ax1.set_xlabel('Barrier'); ax1.set_ylabel('Price ($)')
    ax1.set_title('Down-and-Out/In Call (solid/dashed)'); ax1.legend()

    for sigma in df['Sigma'].unique():
        sub = df[df['Sigma']==sigma]
        color = colors.get(sigma, 'white')
        ax2.plot(sub['Barrier'], sub['KnockPct'], color=color, label=f'σ={sigma*100:.0f}%')
    ax2.set_xlabel('Barrier'); ax2.set_ylabel('Knock-Out Prob (%)'); ax2.set_title('Knock Probability')
    ax2.legend()

    fig.suptitle('Barrier Option Analysis — 3 Volatility Levels', fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, "06_barrier_analysis")

# ============================================================================
# 6. Correlation Trade
# ============================================================================
def plot_correlation():
    df = pd.read_csv("showcase_data/correlation.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for N in df['N'].unique():
        sub = df[df['N']==N]
        style = '-' if N==2 else '--'
        ax1.plot(sub['Rho'], sub['BestOfCall'], f'o{style}', color=C1, ms=3,
                 label=f'Best-of (N={N})', alpha=0.8 if N==2 else 0.5)
        ax1.plot(sub['Rho'], sub['BasketCall'], f'^{style}', color=C3, ms=3,
                 label=f'Basket (N={N})', alpha=0.8 if N==2 else 0.5)
        ax1.plot(sub['Rho'], sub['WorstOfCall'], f's{style}', color=C2, ms=3,
                 label=f'Worst-of (N={N})', alpha=0.8 if N==2 else 0.5)
    ax1.set_xlabel('Correlation (ρ)'); ax1.set_ylabel('Price ($)')
    ax1.set_title('Correlation Trade: 2 vs 5 Assets'); ax1.legend(fontsize=8, ncol=2)

    bv = pd.read_csv("showcase_data/basket_vol.csv")
    colors = [C1, C2, C3, C4, C5]
    labels = ['ρ=0', 'ρ=0.3', 'ρ=0.5', 'ρ=0.8', 'ρ=1']
    for i, (col, label) in enumerate(zip(bv.columns[1:], labels)):
        ax2.plot(bv['N'], bv[col]*100, 'o-', color=colors[i], label=label, ms=4)
        ax2.axhline(20*np.sqrt(max(float(label.split('=')[1]),0)), color=colors[i], alpha=0.2, ls=':')
    ax2.set_xlabel('N assets'); ax2.set_ylabel('Basket Vol (%)')
    ax2.set_title(r'Diversification: $\sigma_{basket} \to \sigma\sqrt{\rho}$ as $N\to\infty$')
    ax2.legend(); ax2.set_ylim(bottom=0)

    fig.suptitle('Multi-Asset Options & Diversification', fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, "07_correlation_trade")

# ============================================================================
# 7. American vs European
# ============================================================================
def plot_american():
    df = pd.read_csv("showcase_data/american.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = {0.02: C3, 0.05: C1, 0.10: C2}

    for r in df['Rate'].unique():
        sub = df[df['Rate']==r]
        color = colors.get(r, 'white')
        ax1.plot(sub['Spot'], sub['AmePut'], color=color, label=f'American r={r*100:.0f}%')
        ax1.plot(sub['Spot'], sub['EurPut'], '--', color=color, alpha=0.5)
    ax1.plot(df['Spot'].unique(), np.maximum(100-df['Spot'].unique(), 0), ':', color='white', alpha=0.3)
    ax1.set_xlabel('Spot'); ax1.set_ylabel('Put Value ($)')
    ax1.set_title('American (solid) vs European (dashed) Put'); ax1.legend()

    for r in df['Rate'].unique():
        sub = df[df['Rate']==r]
        color = colors.get(r, 'white')
        ax2.plot(sub['Spot'], sub['EEPremium'], color=color, label=f'r={r*100:.0f}%')
        ax2.fill_between(sub['Spot'], 0, sub['EEPremium'], alpha=0.08, color=color)
    ax2.set_xlabel('Spot'); ax2.set_ylabel('Premium ($)')
    ax2.set_title('Early Exercise Premium (higher r = more premium)'); ax2.legend()

    fig.suptitle('American Option Analysis — 3 Interest Rate Levels', fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, "08_american_options")

# ============================================================================
# 8. FDM Solution
# ============================================================================
def plot_fdm():
    df = pd.read_csv("showcase_data/fdm.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.plot(df['Spot'], df['AnalyticCall'], '--', color='white', alpha=0.5, lw=1.5, label='BS Analytic')
    ax1.plot(df['Spot'], df['FDMCall'], color=C1, label='FDM Crank-Nicolson')
    ax1.plot(df['Spot'], np.maximum(df['Spot']-100, 0), ':', color=C2, alpha=0.4, label='Intrinsic')
    ax1.set_xlim(50,200); ax1.set_xlabel('Spot'); ax1.set_ylabel('Value ($)')
    ax1.set_title('European Call'); ax1.legend()

    ax2.plot(df['Spot'], df['AnalyticPut'], '--', color='white', alpha=0.5, lw=1.5, label='BS Analytic')
    ax2.plot(df['Spot'], df['FDMPut'], color=C2, label='FDM Crank-Nicolson')
    ax2.set_xlim(50,200); ax2.set_xlabel('Spot'); ax2.set_ylabel('Value ($)')
    ax2.set_title('European Put'); ax2.legend()

    fig.suptitle('Finite Difference Method — Crank-Nicolson (400x2000 grid)', fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, "09_fdm_solution")

# ============================================================================
# 9. Yield Curves + Rate Models
# ============================================================================
def plot_yield_curves():
    yc = pd.read_csv("showcase_data/yield_curves.csv")
    rm = pd.read_csv("showcase_data/rate_models.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.plot(yc['Tenor'], yc['Normal']*100, color=C1, label='Normal (upward)')
    ax1.plot(yc['Tenor'], yc['Flat']*100, color=C4, label='Flat')
    ax1.plot(yc['Tenor'], yc['Inverted']*100, color=C2, label='Inverted (recession)')
    ax1.fill_between(yc['Tenor'], yc['Normal']*100, yc['Inverted']*100, alpha=0.05, color='white')
    ax1.set_xlabel('Maturity (years)'); ax1.set_ylabel('Zero Rate (%)')
    ax1.set_title('Nelson-Siegel Yield Curve Shapes'); ax1.legend()

    ax2.plot(rm['Tenor'], rm['VasicekRate']*100, color=C1, label='Vasicek')
    ax2.plot(rm['Tenor'], rm['CIRRate']*100, color=C2, label='CIR')
    ax2.set_xlabel('Maturity (years)'); ax2.set_ylabel('Zero Rate (%)')
    ax2.set_title('Rate Model Yield Curves (r₀=3%, θ=6%)'); ax2.legend()

    fig.suptitle('Fixed Income — Yield Curves & Rate Models', fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, "10_yield_curves")

# ============================================================================
# 10. Payoff Profiles
# ============================================================================
def plot_payoffs():
    df = pd.read_csv("showcase_data/payoffs.csv")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (pay, val, c, title) in zip(axes, [
        ('CallPayoff','CallValue',C1,'European Call'),
        ('PutPayoff','PutValue',C2,'European Put'),
        ('Straddle','StraddleValue',C3,'Straddle (Call+Put)')
    ]):
        ax.plot(df['Spot'], df[pay], ':', color='white', alpha=0.4, label='Payoff at T')
        ax.plot(df['Spot'], df[val], color=c, label='Value today')
        ax.fill_between(df['Spot'], 0, df[val], alpha=0.1, color=c)
        ax.axvline(100, color='white', alpha=0.2, ls='--')
        ax.set_title(title); ax.set_xlabel('Spot'); ax.set_ylabel('Value ($)'); ax.legend()

    fig.suptitle('Option Payoff Profiles — K=100, σ=20%, T=1y', fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, "11_payoff_profiles")

# ============================================================================
def main():
    print("=" * 60)
    print("  QuantPricer — Rendering Showcase Plots")
    print("=" * 60)
    for fn in [plot_vol_surfaces, plot_greeks, plot_mc_convergence,
               plot_model_comparison, plot_barrier, plot_correlation,
               plot_american, plot_fdm, plot_yield_curves, plot_payoffs]:
        fn()
    print(f"\n  All plots saved to {OUT}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
