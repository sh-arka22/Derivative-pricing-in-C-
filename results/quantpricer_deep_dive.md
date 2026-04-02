# QUANTPRICER — Full Module Deep Dive

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    quantpricer C++ Library                │
│                    17 headers · 142 tests                │
├──────────┬──────────┬──────────┬──────────┬──────────────┤
│  PayOff  │  Option  │  RNG     │  Matrix  │              │
│  (6 types)│ (structs)│(Box-Muller)│(Thomas, │              │
│  functor │          │Cholesky  │Cholesky,LU)             │
├──────────┴──────────┴──────────┴──────────┤              │
│          PRICING ENGINES                   │   RISK &     │
│  ┌─────────┬─────────┬──────────┐         │   ANALYTICS  │
│  │   MC    │  FDM    │Binomial  │         │  ┌──────────┐│
│  │European │θ-scheme │CRR Tree  │         │  │VaR/CVaR  ││
│  │Asian   │Thomas   │American  │         │  │Sharpe    ││
│  │Heston  │solve    │Greeks    │         │  │Sortino   ││
│  │Merton  │         │          │         │  │Drawdown  ││
│  │Barrier │         │          │         │  └──────────┘│
│  │LSM Amer│         │          │         │              │
│  │Multi-  │         │          │         │  ┌──────────┐│
│  │Asset   │         │          │         │  │OrderBook ││
│  └─────────┴─────────┴──────────┘         │  │Matching  ││
├───────────────────────────────────────────┤  └──────────┘│
│  CALIBRATION: Implied Vol (Newton/Bisect) │              │
│  CURVES: Nelson-Siegel, Bootstrap         │ Fixed Income │
│  RATES: Vasicek, CIR, Hull-White          │ Duration/DV01│
└──────────────────────────────────────────────────────────┘
```

---

## 1. PAYOFF HIERARCHY — `include/payoff/payoff.h`

**Purpose:** Define option terminal payoffs as polymorphic function objects that any pricing engine can consume without knowing the specific payoff type.

**Math:**
```
Call:           max(S - K, 0)
Put:            max(K - S, 0)
Digital Call:   1{S > K}
Digital Put:    1{S < K}
Double Digital: 1{D ≤ S ≤ U}
Power Option:   max(S^α - K, 0)
```

**ASCII Payoff Diagrams:**
```
Call Payoff                    Put Payoff
  P/L                           P/L
   |          /                   |\
   |         /                    | \
   |        /                     |  \
   |-------*---------> S          *---\--------> S
           K                      K

Digital Call                  Double Digital
  P/L                           P/L
   |       ________              |    ______
   |      |                      |   |      |
   |      |                      |   |      |
   |______|________-> S          |___|______|___-> S
          K                      D         U
```

**Your Code's Design:**
```
              PayOff (abstract)
          ┌────┼────┬─────┬──────┐
       PayOffCall  Put  Digital  Power
          │              │
   operator()       operator()
   max(S-K,0)       1{S>K}
```

- **Functor pattern** via `operator()(double spot)` — payoffs are callable objects (`payoff.h:23`)
- **Prototype pattern** via `clone()` — enables polymorphic copying with `unique_ptr` (`payoff.h:24`)
- **Validation** — strikes checked non-negative in constructors (`payoff.h:31`)

**Interview Killshot:** "We use the functor pattern so payoffs are first-class objects — they can be stored, copied, and passed to any pricing engine without the engine knowing which payoff type it's pricing. The `clone()` method implements the Prototype pattern because `unique_ptr<PayOff>` can't be copied directly — you need virtual dispatch to get the right derived type."

**Common Interview Questions:**
1. **"Why virtual destructor?"** — Without it, deleting a derived object through a base pointer leaks resources (undefined behavior).
2. **"Why `explicit` on single-arg constructors?"** — Prevents implicit conversion from `double` to `PayOffCall`.
3. **"Why `clone()` instead of just copying?"** — `unique_ptr<PayOff>` erases the concrete type; `clone()` uses virtual dispatch to copy the right derived class.

---

## 2. BLACK-SCHOLES ANALYTICS — `include/greeks/black_scholes.h`

**Purpose:** Closed-form pricing and risk sensitivities for European options. The benchmark every other method is validated against.

**Full Derivation:**
```
Under risk-neutral measure Q, stock follows GBM:
  dS = r·S·dt + σ·S·dW

By Itô's lemma on ln(S):
  d(ln S) = (r - σ²/2)·dt + σ·dW

Integrating:
  ln(S_T/S_0) ~ N((r - σ²/2)·T, σ²·T)

Therefore:
  S_T = S_0 · exp((r - σ²/2)·T + σ·√T·Z),  Z ~ N(0,1)

Call price = e^{-rT} · E^Q[max(S_T - K, 0)]
           = S·N(d₁) - K·e^{-rT}·N(d₂)

where:
  d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
  d₂ = d₁ - σ·√T
```

**Your Code** implements this at `black_scholes.h:28-48`:
- `d1()` and `d2()` as inline functions
- `N(x) = 0.5 * erfc(-x * M_SQRT1_2)` — uses the complementary error function for the CDF
- `n(x)` — standard normal PDF, used for Greeks

**Worked Example:**
```
S=100, K=100, r=5%, T=1, σ=20%

d₁ = [ln(1) + (0.05 + 0.02)·1] / (0.2·1) = 0.07/0.2 = 0.35
d₂ = 0.35 - 0.20 = 0.15

N(0.35) = 0.6368,  N(0.15) = 0.5596

Call = 100·0.6368 - 100·e^{-0.05}·0.5596
     = 63.68 - 53.24 = 10.45
```

**Greeks (from your code, `black_scholes.h:63-104`):**
```
Greek     Formula                     Your Code
-----     -------                     ---------
Delta     N(d₁)                       delta_call() at line 64
Gamma     n(d₁)/(S·σ·√T)             gamma() at line 73
Vega      S·n(d₁)·√T                 vega() at line 78
Theta     -S·n(d₁)·σ/(2√T) - rKe^{-rT}N(d₂)    theta_call() at line 83
Rho       K·T·e^{-rT}·N(d₂)         rho_call() at line 98

  Delta
  1.0 |                    --------
      |                 /
  0.5 |              /           <-- ATM: delta ≈ 0.5
      |           /
  0.0 |----------
      +---------------------------------> S
                  K
```

**Interview Killshot:** "N(d₁) is the probability of expiring ITM under the *stock-price measure* (where S is the numeraire), while N(d₂) is that probability under the *risk-neutral* (money-market) measure. Delta = N(d₁) is NOT the probability of expiring ITM — that's N(d₂). This distinction matters for hedging vs. probability estimation."

**Common Interview Questions:**
1. **"Derive BS from scratch"** — Itô on ln(S), integrate, risk-neutral expectation.
2. **"Prove put-call parity"** — C - P = S - Ke^{-rT}. One replicating portfolio for both.
3. **"Why is Gamma the same for calls and puts?"** — Differentiate parity twice: d²C/dS² = d²P/dS².
4. **"What's the connection between Theta and Gamma?"** — BS PDE: Θ + ½σ²S²Γ + rSΔ - rV = 0. For delta-hedged portfolio: Θ ≈ -½σ²S²Γ.

---

## 3. MONTE CARLO ENGINE — `include/mc/monte_carlo.h`

**Purpose:** Price any payoff (European, Asian, exotic) by simulating risk-neutral paths. The universal pricing method.

**Math:**
```
Risk-neutral pricing:
  V₀ = e^{-rT} · E^Q[Payoff(S_T)]

MC estimator:
  V̂ = e^{-rT} · (1/N) · Σᵢ Payoff(S_T^(i))

Standard error: SE = std(payoffs) / √N
95% CI: [V̂ - 1.96·SE, V̂ + 1.96·SE]

Convergence rate: O(1/√N) — need 4× paths for 2× accuracy!
```

**Your Code** (`monte_carlo.h:41-78`):
- **Antithetic variates** built in — uses Z and -Z pairs to halve variance
- `n_pairs = num_paths / 2` — each pair contributes one averaged sample
- Returns `MCResult` with price, SE, and 95% CI

**Antithetic Variates (your implementation at lines 56-71):**
```
For each pair:
  Path:       Z  → S_T = S₀·exp(drift + σ√T·Z)   → payoff₁
  Antithetic: -Z → S_T = S₀·exp(drift - σ√T·Z)  → payoff₂
  avg = (payoff₁ + payoff₂) / 2

Why it works: Cov(payoff(Z), payoff(-Z)) < 0
  → Var(avg) = [Var(X) + Cov(X,X')]/2 < Var(X)/2
```

**Path-Dependent: Asian Options (`monte_carlo.h:110-172`):**
```
Arithmetic Asian: A = (1/n)·Σ S(tᵢ),  payoff = max(A - K, 0)
Geometric Asian:  A_geo = exp((1/n)·Σ ln(S(tᵢ)))

Your code generates full GBM paths via generate_gbm_path(),
then averages prices (excluding S(0) — line 125).
```

**Convergence:**
```
  Error
   |  \
   |   \  No variance reduction
   |    \
   |     \___
   |      \   Antithetic
   |       \___________
   +-------------------------> N (paths)
```

**Interview Killshot:** "MC convergence is dimension-independent — O(1/√N) whether you have 1 asset or 100. PDE methods suffer the curse of dimensionality (grid points grow as O(M^d)). This is why MC dominates for multi-asset and path-dependent products."

---

## 4. RANDOM NUMBER GENERATION — `include/rng/rng.h`

**Purpose:** Generate uniform and normal random variates for MC simulation, with support for correlated samples via Cholesky.

**Your Design:**
```
RandomNumberGenerator (abstract)
    ├── LinearCongruentialGenerator   (educational — glibc params)
    └── MersenneTwisterRNG            (production — std::mt19937_64)
              │
    StatisticalDistribution (abstract)
    └── StandardNormalDistribution    (CDF via erfc, inv_cdf via Beasley-Springer-Moro)
```

**Box-Muller Transform** (`rng.h:122-135`):
```
Given U₁, U₂ ~ Uniform(0,1):
  R = √(-2·ln(U₁))
  Θ = 2π·U₂
  Z₁ = R·cos(Θ),  Z₂ = R·sin(Θ)
Then Z₁, Z₂ ~ iid N(0,1)

Your code: normals[i] = r*cos(theta), normals[i+1] = r*sin(theta)
Guard: u1 = max(u1, 1e-15) to avoid log(0)
```

**Cholesky for Correlated Normals** (`rng.h:139-145`):
```
For 2D (Heston):
  z₁ = Z[0]
  z₂ = ρ·Z[0] + √(1-ρ²)·Z[1]    ← this is exactly your line 143

This gives Corr(z₁, z₂) = ρ by construction.
```

**Interview Killshot:** "In production, you'd use Sobol sequences (quasi-random) instead of Mersenne Twister (pseudo-random). Sobol fills the space more uniformly, giving O(1/N) convergence instead of O(1/√N). But Sobol requires careful dimension handling and doesn't work well with antithetic variates."

---

## 5. GREEKS ENGINE — `include/greeks/greeks_engine.h`

**Purpose:** Compute price sensitivities via three independent methods: analytic, finite difference, and MC pathwise.

**Three Methods Compared:**
```
Method          Speed      Accuracy    Generality       Your Code
-----------     -----      --------    ----------       ---------
Analytic        O(1)       Exact       BS only          analytic_greeks_call() :29
Finite Diff     O(k·MC)    O(h²)      Any model        FDGreeks class :59
MC Pathwise     O(MC)      O(1/√N)    Smooth payoffs   mc_delta_pathwise() :131
```

**FDGreeks Design** — Uses `std::function<double(S,K,r,T,σ)>` as the pricing callback. This means it works with *any* pricing model:
```cpp
FDGreeks fd_greeks(bs::call_price);         // BS
FDGreeks fd_greeks(my_heston_pricer);       // Heston
FDGreeks fd_greeks(my_local_vol_pricer);    // Local vol
```

**Central Differences** (`greeks_engine.h:67-73`):
```
Delta = [V(S+h) - V(S-h)] / (2h)           O(h²) accuracy
Gamma = [V(S+h) - 2V(S) + V(S-h)] / h²

h = S * h_pct (proportional bump, default 1%)
```

**MC Pathwise / IPA** (`greeks_engine.h:131-149`):
```
Delta = e^{-rT} · E[1{S_T > K} · S_T/S₀]

Key: same paths, no re-simulation.
Fails for discontinuous payoffs (digitals)!
```

**Interview Killshot:** "The pathwise method gives you Greeks 'for free' — same paths, no re-simulation. But it fails for discontinuous payoffs like digitals. The likelihood ratio method works universally but has higher variance. In practice, desks use pathwise for vanilla Greeks and likelihood ratio or Malliavin calculus for exotics with discontinuous payoffs."

---

## 6. IMPLIED VOLATILITY — `include/vol/implied_vol.h`

**Purpose:** Invert the BS formula to extract the market's implied volatility from observed option prices.

**The Inverse Problem:**
```
Given: C_mkt (observed option price)
Find:  σ_imp such that BS(S, K, r, T, σ_imp) = C_mkt
```

**Method 1: Bisection** (`implied_vol.h:33-98`):
```
Bracket [σ_lo=0.001, σ_hi=5.0], halve each iteration.
Rate: 1 bit of accuracy per iteration.
After 20 iterations: accuracy ~ 5/(2²⁰) ≈ 5×10⁻⁶

Your code validates the bracket: f(lo)·f(hi) < 0
```

**Method 2: Newton-Raphson** (`implied_vol.h:107-135`):
```
σ_{n+1} = σ_n - [C(σ_n) - C_mkt] / Vega(σ_n)

Quadratic convergence: doubles correct digits each step.
Your safety guard: when |Vega| < 1e-14, falls back to bisection (line 123).
Also: σ = max(σ, 1e-6) to stay positive (line 131).
```

**Vol Surface Builder** (`implied_vol.h:149-169`):
```
Computes σ_imp(K, T) across a grid of strikes × maturities
Uses Newton with bisection fallback per grid point.
```

**Interview Killshot:** "Implied vol is not a real volatility — it's a quoting convention. The BS model is wrong (constant vol, no jumps), but we use it as a coordinate transformation: instead of quoting prices in dollars, we quote in 'BS implied vol units'. It's like quoting bond prices as yields — nobody believes rates are constant, but yield is a useful metric."

---

## 7. FINITE DIFFERENCE METHOD — `include/fdm/fdm.h`

**Purpose:** Solve the Black-Scholes PDE numerically on a grid. Gives the *entire price surface* V(S) in one solve.

**The PDE:**
```
∂V/∂t + ½σ²S²·∂²V/∂S² + rS·∂V/∂S - rV = 0
```

**Your Design** (`fdm.h:27-59`):
```
PDECoefficients (abstract)
   └── BlackScholesPDE
         diffusion(S)  = ½σ²S²
         convection(S) = rS
         reaction(S)   = -r
```

**Grid and Time-Stepping** (`fdm.h:80-183`):
```
  S_max
   |  ___________________________
   |  |  |  |  |  |  |  |  |  |  Terminal payoff at T
   |  |  |  |  |  |  |  |  |  |
   |  |  |  |  |  |  |  |  |  |  ← Grid points V(Sᵢ, tⱼ)
   |  |  |  |  |  |  |  |  |  |
   0  ___________________________
      t=0                    t=T
      ←←← march backwards ←←←←←
```

**Theta-Scheme** (`fdm.h:137-148`):
```
(I - θΔtA)·V^n = (I + (1-θ)ΔtA)·V^{n+1}

θ = 0.0  → EXPLICIT  (fast, conditionally stable)
θ = 0.5  → CRANK-NICOLSON (O(Δt²+ΔS²), unconditionally stable)
θ = 1.0  → FULLY IMPLICIT (O(Δt), unconditionally stable)

Tridiagonal system solved by Thomas algorithm: O(N) per step.
Total: O(N·M).
```

**Your code auto-detects call vs put** from the payoff at S=0 (`fdm.h:96-99`) for boundary conditions.

**Interview Killshot:** "FDM beats MC for 1D and 2D because you get the ENTIRE price surface V(S) in one solve — all strikes simultaneously. MC gives one price per run. But beyond d=3 assets, the curse of dimensionality kills FDM and MC wins."

---

## 8. BINOMIAL TREE — `include/tree/binomial_tree.h`

**Purpose:** CRR lattice for European and American options with tree-based Greeks.

**CRR Parameters** (`binomial_tree.h:53-57`):
```
dt = T/N
u  = exp(σ·√dt)          Up factor
d  = 1/u                  Down factor (recombining!)
p  = (e^{r·dt} - d)/(u-d)  Risk-neutral probability

Tree structure (N=3):
                     S·u³
                   /
             S·u²
           /       \
      S·u            S·u·d ← recombining
     /    \        /
    S      S·d
     \    /    \
      S·d       S·d²
           \   /
            S·d²
               \
                S·d³
```

**O(N) Space Optimization** (`binomial_tree.h:62-98`):
Your code uses a single vector of N+1 values, updated in-place during backward induction — no O(N²) tree storage.

**American Early Exercise** (`binomial_tree.h:81-85`):
```
V[j] = max(continuation, payoff(S_ij))
```

**Greeks from the Tree** (`binomial_tree.h:100-117`):
```
Delta = (V₁₁ - V₁₀) / (S·u - S·d)           from step 1
Gamma = (Δ_up - Δ_down) / (½(S·u² - S·d²))   from step 2
Theta = (V₂₁ - V₀) / (2·dt)                   from steps 0 and 2
```

**Interview Killshot:** "The binomial tree converges to BS as N→∞. The CRR parameters match the first two moments of GBM, and by CLT the sum of N binomial steps → normal distribution. Convergence rate is O(1/N) with oscillation — Richardson extrapolation (V* = 2V(2N) - V(N)) removes the leading error."

**Key Questions:**
1. **"Can you exercise an American call early (no dividends)?"** — NEVER. C ≥ S - Ke^{-rT} > S - K = intrinsic. The call is always worth more alive.
2. **"When do you exercise an American put early?"** — When S ≪ K and time value < intrinsic. Deep ITM put: you'd rather have Ke^{-rT} now.

---

## 9. HESTON STOCHASTIC VOLATILITY — `include/mc/monte_carlo.h:257-315`

**Purpose:** Model the volatility smile/skew by making variance itself stochastic.

**The SDE System:**
```
dS = r·S·dt + √v·S·dW_S              Stock price
dv = κ(θ-v)·dt + ξ·√v·dW_v           Variance (CIR process)
Corr(dW_S, dW_v) = ρ                  Leverage effect

Feller condition: 2κθ > ξ² ensures v > 0 a.s.
```

**Full Truncation Scheme** (your code at lines 284-300):
```
v_pos = max(v, 0)                          ← prevents √(negative)
S *= exp((r - v_pos/2)·dt + √v_pos·√dt·Z₁)   ← log-Euler for S
v += κ·(θ - v_pos)·dt + ξ·√v_pos·√dt·Z₂      ← Euler for v

Key: raw v is carried forward (can go negative), but all coefficients use v_pos.
This is Lord-Koekkoek-Van Dijk (2010) — best bias/RMSE tradeoff.
```

**Why Heston Produces a Smile:**
```
ρ < 0 (leverage): S drops → dW_S < 0 → dW_v tends positive → vol spikes
  → OTM puts expensive → LEFT SKEW

ξ > 0 (vol-of-vol): variance is random → fat tails
  → both OTM puts AND calls expensive → SMILE

  Implied Vol
   |  \         /
   |   \       /     ← ρ < 0, ξ > 0: skewed smile
   |    \_____/
   +-------------------> K/S
       0.8  1.0  1.2

Special case: ξ = 0 → Heston collapses to BS! (your code comment at line 251)
```

**Interview Killshot:** "The leverage effect (ρ < 0) is why equity vol surfaces have negative skew — when stocks crash, vol spikes. This was dramatically visible in 1987 and 2008. Before '87, the smile was nearly flat."

---

## 10. MERTON JUMP-DIFFUSION — `include/mc/monte_carlo.h:181-230`

**Purpose:** Model sudden large moves (crashes/spikes) that GBM can't capture.

**The SDE:**
```
dS/S = (r - λk)·dt + σ·dW + J·dN

N ~ Poisson(λ),  J ~ LN(μ_j, σ_j²)
k = E[e^J - 1] = exp(μ_j + σ_j²/2) - 1   (jump compensator)
```

**Your Code** (`monte_carlo.h:192-217`):
- Computes the jump compensator k for drift correction (line 192)
- Uses `std::poisson_distribution` for jump counts (line 198)
- Sums multiple jumps per step (inner loop, line 214)

**Interview Killshot:** "Merton explains the SHORT-TERM vol smile but not the long-term one. At short maturities, jump risk dominates → steep smile. At long maturities, CLT averages out jumps → smile flattens. Heston explains long-term smile. In practice, desks use Bates model = Heston + jumps for both."

---

## 11. BARRIER OPTIONS — `include/barrier/barrier.h`

**Purpose:** Options whose payoff depends on whether the underlying crosses a barrier during its life. Cheaper than vanilla → popular in structured products.

**In-Out Parity:**
```
V_in(S,K,H) + V_out(S,K,H) = V_vanilla(S,K)   [rebate = 0]

Your code verifies this at barrier.h:317-337 — in_out_parity_error().
```

**8 Barrier Types** — Your code implements all via Haug building blocks A-F (`barrier.h:168-226`) with a dispatch table at lines 285-309:
```
Type               | K ≥ H             | K < H
-------------------|--------------------|-------------------
Down-and-In  Call  | C + E              | A - B + D + E
Down-and-Out Call  | A - C + F          | B - D + F
Up-and-In    Call  | A + E              | B - C + D + E
... (8 total)
```

**BGK Continuity Correction** (`barrier.h:442-466`):
```
Problem: discrete monitoring overprices knock-out (misses barrier crossings)
Fix: H_adj = H · exp(∓ β·σ·√dt),  β ≈ 0.5826

Your code: BGK_BETA = 0.5826, with barrier_price_bgk() at line 461.
```

**Interview Killshot:** "Barrier options have discontinuous Greeks at the barrier — Delta can flip sign, Gamma explodes. A trader short a down-and-out call near the barrier faces: small move → entire position wiped out. This is why structured desks use 'soft barriers' (averaged) or cash rebates."

---

## 12. AMERICAN OPTIONS — LSM — `include/mc/american_mc.h`

**Purpose:** Price early-exercise options via simulation using regression-based continuation value estimates.

**Longstaff-Schwartz Algorithm** (`american_mc.h:164-263`):
```
1. Generate N paths forward (GBM)             → paths[i][j]
2. At T: cashflow = payoff(S_T)
3. Walk BACKWARDS from T-1 to 1:
   a. Find ITM paths (payoff > 0)
   b. Regress discounted future cashflows on S/S₀ polynomial basis
      → β = (X'X)⁻¹ X'Y  (normal equations at lines 54-131)
   c. If intrinsic > continuation: EXERCISE
4. Price = mean of discounted cashflows

  Payoff
   |  x  x        Exercise region
   |     x   x    (intrinsic > continuation)
   |  --------    Continuation curve (regression fit)
   |      x  x
   | x   x        Hold region
   +-------------------> S(t)
```

**Your Choices:**
- Polynomial basis: simple monomials {1, S/S₀, (S/S₀)², ...} with normalization for stability (`american_mc.h:212`)
- Degree default: 3 (cubic regression)
- Normal equations solved via Gaussian elimination with partial pivoting (`american_mc.h:94-131`)

**Interview Killshot:** "LSM gives a LOWER BOUND because the exercise policy is suboptimal (estimated continuation). For an UPPER BOUND, use Andersen-Broadie duality. The true price is bracketed between the two."

---

## 13. MULTI-ASSET & BASKET — `include/multi_asset/multi_asset.h`

**Purpose:** Price options on multiple correlated assets — baskets, best-of, worst-of, exchange options.

**N-Dimensional Cholesky** (`multi_asset.h:216-232`):
```
R = L·L'  (decompose correlation matrix)
W = L·Z   (correlate independent normals)

Your code at line 227: W[i] += L(i,j)*Z[j] for j ≤ i (lower triangular)
```

**9 Payoff Types** (`multi_asset.h:149-197`):
BasketCall, BasketPut, BestOfCall, WorstOfCall, BestOfPut, WorstOfPut, Rainbow, SpreadCall, Exchange

**Correlation Effects:**
```
                     ρ↑              ρ↓
  Basket call:       MORE expensive   LESS expensive
  Best-of call:      LESS expensive   MORE expensive

  → This is the "CORRELATION TRADE"
```

**Margrabe's Formula** (`multi_asset.h:345-366`):
```
V = S₁·N(d₁) - S₂·N(d₂)    ← NO discounting!
σ_spread = √(σ₁² + σ₂² - 2ρσ₁σ₂)

Key: does NOT depend on r (S₂ is the numeraire).
```

**Interview Killshot:** "MC is the ONLY practical method for d > 3 assets. PDE grid points grow as O(N^d), but MC scales linearly in dimension. This is why every exotic desk runs MC for structured products."

---

## 14. RISK MANAGEMENT — `include/risk/risk.h`

**Purpose:** Portfolio risk measurement (VaR, CVaR, Sharpe, Sortino, drawdown) and risk decomposition.

**VaR vs CVaR:**
```
  Probability
   |           /\
   |          /  \
   |         /    \
   |   ////|        \
   |  /////|         \___
   +--|----|-----------------> P&L
     CVaR  VaR            (loss is negative)

VaR₉₉%  = "maximum loss at 99% confidence"
CVaR₉₉% = "average loss in worst 1%"
Always: CVaR ≥ VaR
```

**Your Code Implements 3 Approaches:**
1. **Parametric** (`risk.h:182-195`): VaR = -μ + z_α·σ, CVaR = -μ + σ·φ(z_α)/(1-α)
2. **Historical/MC** (`risk.h:240-261`): Sort P&L, take percentile
3. **MC Portfolio** (`risk.h:278-310`): Simulate correlated returns via Cholesky

**VaR Subadditivity Demo** (`risk.h:500-531`) — Two binary bets proving VaR(A+B) > VaR(A) + VaR(B):
```
P(at least one loss) = 1 - 0.96² = 7.84% > 5%
→ VaR(A+B) > 0 = VaR(A) + VaR(B)
→ VaR penalises diversification!
```

**Risk Decomposition** (`risk.h:328-363`):
```
Marginal VaR:   MVaR_i = z_α·(Σw)_i / σ_p
Component VaR:  CVaR_i = w_i · MVaR_i
Property: Σ CVaR_i = Total VaR  (Euler decomposition)
```

**Interview Killshot:** "VaR answers 'how bad?' CVaR answers 'if bad, HOW bad?' Basel III switched from VaR to Expected Shortfall because VaR violates subadditivity — banks could game it by concentrating tail risk. CVaR is coherent and can't be gamed this way."

---

## 15. FIXED INCOME — `include/fixed_income/fixed_income.h`

**Purpose:** Yield curve construction, bond pricing, duration, convexity, and DV01.

**Four Views of the Yield Curve:**
```
  Rate
   |        Forward
   |       /
   |      /     Spot (zero)
   |     / ___/
   |    //
   |   //       Par
   |  //  ___/
   | //__/
   +-------------------------> Maturity
```

**Nelson-Siegel Model** (`fixed_income.h:233-244`):
```
r(t) = β₀ + β₁·g₁(t/τ) + β₂·g₂(t/τ)

g₁(x) = (1-e^{-x})/x        slope factor
g₂(x) = g₁(x) - e^{-x}      curvature factor

β₀ = level, β₁ = slope, β₂ = curvature
→ First 3 PCA components (~99% of yield curve variation)
```

**Bootstrapping** (`fixed_income.h:322-356`): Extracts zero rates from deposits and par swaps:
```
Deposit: D(t) = 1/(1 + r·t)
Swap:    D(tₙ) = (1 - s·Σ D(tᵢ)) / (1 + s)
```

**Duration & Convexity** (`fixed_income.h:444-474`):
```
D_mac = Σ tᵢ·PV(CFᵢ) / P          Macaulay duration
D_mod = D_mac (continuous compounding)
DV01  = D_mod · P · 0.0001         Dollar per basis point
C     = Σ tᵢ²·PV(CFᵢ) / P          Convexity

dP/P ≈ -D_mod·dy + ½·C·dy²
```

**Interview Killshot:** "Duration of a zero-coupon bond equals its maturity T — this is the upper bound. A perpetuity (infinite maturity) has finite duration: D = (1+y)/y. At 5% yield, that's only 21 years."

---

## 16. INTEREST RATE MODELS — `include/rates/rate_models.h`

**Purpose:** Short rate models (Vasicek, CIR, Hull-White) with analytic bond pricing and MC simulation.

**Three Models:**
```
Model       SDE                              r<0?  Fits curve?
Vasicek     dr = κ(θ-r)dt + σdW             YES   NO
CIR         dr = κ(θ-r)dt + σ√r·dW          NO*   NO
Hull-White  dr = [θ(t)-ar]dt + σdW           YES   YES

*CIR: r≥0 if Feller condition 2κθ > σ² holds
```

**Vasicek Analytic Bond** (`rate_models.h:173-196`):
```
P(0,T) = A(T)·exp(-B(T)·r₀)
B(T)   = (1-e^{-κT})/κ
ln(A)  = (B-T)·(κ²θ-σ²/2)/κ² - σ²B²/(4κ)
```

**Simulation Schemes:**
- Vasicek: **exact Gaussian transition** (not Euler!) at `rate_models.h:321-349`
- CIR: **Milstein** with full truncation at `rate_models.h:352-384`
- Hull-White: Euler with time-dependent `θ(t)` at `rate_models.h:387-416`

**Bond Option** (`rate_models.h:479-503`) — Jamshidian's formula using Black's formula on the bond forward.

**Interview Killshot:** "Hull-White's key advantage: θ(t) is calibrated to EXACTLY match today's bond prices. Vasicek/CIR can't. For relative-value trading, you MUST match the market curve first (no-arbitrage), then find mispricings."

---

## 17. ORDER BOOK & MATCHING ENGINE — `include/orderbook/orderbook.h`

**Purpose:** Full limit order book with price-time priority matching — exchange-grade market microstructure.

**Data Structure Design:**
```
Layer 1: Order — {id, side, price, qty, timestamp}
Layer 2: PriceLevel — std::list<Order> (FIFO queue)
Layer 3: Book — std::map<price, PriceLevel>

  ASKS (sorted ascending — lowest first)
  ┌──────────────────────────────────────────┐
  │ 100.05: [Order A (100)] → [Order B (200)]│
  │ 100.03: [Order C (500)]                  │ ← Best Ask
  ├──────── SPREAD: 0.02 ────────────────────┤
  │ 100.01: [Order D (300)] → [Order E (150)]│ ← Best Bid
  │  99.98: [Order F (1000)]                 │
  └──────────────────────────────────────────┘
  BIDS (sorted descending — highest first)

Complexity:
  Add:    O(log P) — sorted map insert
  Cancel: O(1) — unordered_map lookup + list erase
  Match:  O(1) per fill — front of FIFO queue
  Best:   O(1) — map.begin()
```

**Matching Logic** (`orderbook.h:478-558`):
```
Incoming BUY:
  1. Walk asks (lowest first)
  2. If buy_price ≥ ask_price → match at ask_price (passive improvement)
  3. Fill FIFO at each level
  4. Remaining qty → rest in bid book
```

**Order Cancellation** (`orderbook.h:280-299`):
```
OrderLocation: { side, price, list<Order>::iterator }
stored in unordered_map<OrderId, OrderLocation>
→ O(1) cancel: hash lookup + list.erase()
```

**Metrics:** spread, mid, imbalance, VWAP, L2 depth — all at `orderbook.h:306-413`.

**Interview Killshot:** "In production, you'd NEVER use floating-point prices — integer ticks (price × 10000). Also, std::map has poor cache locality — real exchanges use arrays indexed by tick offset from a reference price, giving O(1) everything."

---

## 18. MATRIX & LINEAR ALGEBRA — `include/matrix/matrix.h`

**Purpose:** Template matrix class with the numerical solvers that power FDM and correlated MC.

**Key Algorithms:**
```
Thomas Algorithm (O(N)):     solve_thomas() at matrix.h:199
  → Critical for Crank-Nicolson FDM (tridiagonal system each time step)

Cholesky (O(N³/3)):         cholesky() at matrix.h:224
  → Correlation matrix decomposition for multi-asset MC

LU with Partial Pivoting:   solve_lu() at matrix.h:146
  → General square systems, used in LSM regression
```

**QMatrix Design:** Template class with `vector<vector<T>>` storage, operator overloading for `+`, `-`, `*`, and `()` element access. Supports matrix-matrix multiply, matrix-vector multiply, transpose, and Frobenius norm.

---

## TOP 20 RAPID-FIRE INTERVIEW ANSWERS

| # | Question | Answer |
|---|----------|--------|
| 1 | Derive BS | Itô on ln(S), integrate, risk-neutral expectation → S·N(d₁) - Ke^{-rT}·N(d₂) |
| 2 | d₁ vs d₂? | d₁ = ITM prob under stock measure; d₂ = under money-market measure |
| 3 | Early exercise American call? | NEVER (no dividends). C ≥ S - Ke^{-rT} > intrinsic |
| 4 | MC convergence? | O(1/√N), dimension-independent, 4× paths for 2× accuracy |
| 5 | Antithetic variates? | Use Z and -Z; negative covariance reduces variance |
| 6 | Feller condition? | 2κθ > ξ²; ensures variance stays strictly positive in CIR/Heston |
| 7 | Why Heston smile? | ρ<0 gives skew (leverage), ξ>0 gives fat tails (smile) |
| 8 | In-out parity? | V_in + V_out = V_vanilla (paths either hit barrier or don't) |
| 9 | Longstaff-Schwartz? | Backward MC; regress continuation on polynomial basis of S |
| 10 | VaR not coherent? | Violates subadditivity: VaR(A+B) can > VaR(A) + VaR(B) |
| 11 | Thomas algorithm? | O(N) tridiagonal solver; forward sweep + back substitution |
| 12 | CN vs explicit FDM? | CN: unconditionally stable, O(Δt²). Explicit: conditionally stable |
| 13 | Duration of zero-coupon? | Exactly T (its maturity) |
| 14 | Margrabe? | Exchange option = S₁N(d₁)-S₂N(d₂), no discounting, no r dependence |
| 15 | Cholesky in MC? | Decompose R=LL'; multiply independent normals by L → correlated |
| 16 | Price-time vs pro-rata? | FIFO rewards speed; pro-rata rewards capital |
| 17 | BGK correction? | H_adj = H·exp(±β·σ·√dt), β≈0.5826 |
| 18 | Nelson-Siegel factors? | β₀=level, β₁=slope, β₂=curvature ≈ PCA components |
| 19 | Hull-White over Vasicek? | θ(t) calibrated to match today's curve exactly |
| 20 | MC Greeks methods? | Pathwise: fast, smooth payoffs. Likelihood ratio: universal. FD: always works, 2-3× cost |
