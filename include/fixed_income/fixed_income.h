#ifndef QUANTPRICER_FIXED_INCOME_H
#define QUANTPRICER_FIXED_INCOME_H

// ============================================================================
// Fixed Income — Yield Curves & Bond Pricing — Day 13
//
// ============================================================================
// MATHEMATICS — The Discount Factor Framework
// ============================================================================
//
// FUNDAMENTAL PRINCIPLE OF FIXED INCOME:
//   The price of any fixed-income instrument is the present value of
//   its future cash flows, discounted using the appropriate discount factors.
//
//   PV = sum_i CF_i * D(t_i)
//
//   where D(t) = discount factor = price today of $1 received at time t.
//
// ============================================================================
// YIELD CURVE REPRESENTATIONS — four equivalent views:
//
// 1. DISCOUNT FACTORS D(t):
//    D(0) = 1,  D(t) decreasing in t (money later is worth less)
//    D(t) = exp(-r(t) * t)  under continuous compounding
//
// 2. ZERO (SPOT) RATES r(t):
//    The continuously-compounded rate for maturity t:
//    r(t) = -ln(D(t)) / t
//    This is the yield on a zero-coupon bond maturing at t.
//
// 3. FORWARD RATES f(t1, t2):
//    The rate agreed today for borrowing from t1 to t2:
//    f(t1,t2) = -[ln(D(t2)) - ln(D(t1))] / (t2 - t1)
//    No-arbitrage: D(t2) = D(t1) * exp(-f(t1,t2) * (t2-t1))
//
// 4. INSTANTANEOUS FORWARD RATE f(t):
//    f(t) = lim_{dt->0} f(t, t+dt) = -d/dt ln(D(t))
//    Relationship: D(t) = exp(-integral_0^t f(s) ds)
//    This is the fundamental building block in term structure models.
//
// ============================================================================
// NELSON-SIEGEL MODEL (1987):
//
//   r(t) = beta0 + beta1 * [(1 - exp(-t/tau)) / (t/tau)]
//                 + beta2 * [(1 - exp(-t/tau)) / (t/tau) - exp(-t/tau)]
//
//   Parameters:
//     beta0 = long-term level (r(infinity) = beta0)
//     beta1 = short-long spread (r(0) = beta0 + beta1)
//     beta2 = curvature (hump/trough)
//     tau   = decay speed (controls where the hump peaks)
//
//   Interview insight: the three components are interpretable:
//     - beta0: level factor (parallel shift)
//     - beta1: slope factor (steepening/flattening)
//     - beta2: curvature factor (twist)
//   These correspond to the first three principal components of yield curve
//   movements, which explain ~99% of all yield curve variation.
//
// ============================================================================
// SVENSSON EXTENSION (1994):
//
//   Adds a second curvature term for better fit at short maturities:
//   r(t) = NS(t) + beta3 * [(1 - exp(-t/tau2)) / (t/tau2) - exp(-t/tau2)]
//
// ============================================================================
// BOND PRICING:
//
//   For a bond with coupon rate c, face value F, and N coupon payments:
//
//   Clean Price = sum_{i=1}^{N} c*F/freq * D(t_i) + F * D(t_N)
//   Dirty Price = Clean Price + Accrued Interest
//   Accrued = c * F / freq * (days since last coupon / days in coupon period)
//
// ============================================================================
// DURATION & CONVEXITY — Interest Rate Risk Measures:
//
// MACAULAY DURATION (weighted average time to cash flows):
//   D_mac = [sum_i t_i * CF_i * D(t_i)] / P
//   Interpretation: "effective maturity" of the bond
//
// MODIFIED DURATION (price sensitivity to yield):
//   D_mod = D_mac / (1 + y/freq)
//   Approximation: dP/P ≈ -D_mod * dy
//   "A bond with D_mod=5 loses ~5% if yields rise 1%"
//
// DV01 (Dollar Value of a Basis Point):
//   DV01 = -dP/dy * 0.0001 ≈ D_mod * P * 0.0001
//   "Dollar loss per $1 notional for a 1bp yield increase"
//
// CONVEXITY (curvature of price-yield relationship):
//   C = [sum_i t_i^2 * CF_i * D(t_i)] / P    (simplified)
//   Better approximation: dP/P ≈ -D_mod * dy + 0.5 * C * dy^2
//   Convexity is GOOD: positive convexity means the bond gains more from
//   falling yields than it loses from rising yields.
//
// KEY INTERVIEW INSIGHTS:
//
// 1. "Why is the yield curve usually upward sloping?"
//    Expectations hypothesis: market expects rates to rise.
//    Liquidity preference: investors demand premium for longer maturities.
//    In practice: mix of both + term premium.
//
// 2. "What does an inverted yield curve signal?"
//    Historically the best recession predictor (10y-2y spread < 0).
//    Explanation: market expects rate CUTS (recession).
//
// 3. "Duration of a zero-coupon bond?"
//    D_mac = T (the maturity itself). This is the upper bound.
//    A coupon bond always has D_mac < T because coupons pull the
//    weighted average time forward.
//
// 4. "Why do traders care about DV01?"
//    It converts yield moves to dollar P&L: P&L ≈ -DV01 * delta_bp.
//    A portfolio with DV01 = $50,000 loses $50K per bp rise.
//    Hedging = matching DV01 (and convexity for large moves).
//
// 5. "Duration-convexity approximation accuracy?"
//    Duration alone: linear, good for dy < 50bp.
//    Duration+convexity: quadratic, good for dy < 200bp.
//    Beyond that: reprice from the curve.
// ============================================================================

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace fixed_income {

// ============================================================================
// Yield Curve — Core Data Structure
// ============================================================================

class YieldCurve {
public:
    // Construct from zero rates at given tenors
    // tenors = {0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30} (years)
    // zeros  = {0.03, 0.032, 0.035, ...} (continuously compounded)
    YieldCurve(const std::vector<double>& tenors,
               const std::vector<double>& zero_rates)
        : tenors_(tenors), zeros_(zero_rates)
    {
        if (tenors_.size() != zeros_.size() || tenors_.empty())
            throw std::invalid_argument("YieldCurve: size mismatch or empty");
    }

    // Default: flat curve at given rate
    explicit YieldCurve(double flat_rate = 0.05, double max_tenor = 30.0)
        : tenors_({0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30}),
          zeros_(11, flat_rate) {}

    /// Interpolate zero rate at arbitrary maturity t (linear interpolation)
    double zero_rate(double t) const {
        if (t <= 0.0) return zeros_.front();
        if (t <= tenors_.front()) return zeros_.front();
        if (t >= tenors_.back()) return zeros_.back();

        // Find bracketing tenors
        auto it = std::lower_bound(tenors_.begin(), tenors_.end(), t);
        size_t i = static_cast<size_t>(it - tenors_.begin());
        if (i == 0) return zeros_[0];

        // Linear interpolation on zero rates
        double t0 = tenors_[i-1], t1 = tenors_[i];
        double r0 = zeros_[i-1],  r1 = zeros_[i];
        double frac = (t - t0) / (t1 - t0);
        return r0 + frac * (r1 - r0);
    }

    /// Discount factor: D(t) = exp(-r(t) * t)
    double discount(double t) const {
        return std::exp(-zero_rate(t) * t);
    }

    /// Forward rate between t1 and t2:
    /// f(t1,t2) = -[ln(D(t2)) - ln(D(t1))] / (t2 - t1)
    ///          = [r(t2)*t2 - r(t1)*t1] / (t2 - t1)
    double forward_rate(double t1, double t2) const {
        if (t2 <= t1) throw std::invalid_argument("forward_rate: t2 must be > t1");
        if (t1 <= 0.0) return zero_rate(t2);
        return (zero_rate(t2) * t2 - zero_rate(t1) * t1) / (t2 - t1);
    }

    /// Instantaneous forward rate at t (numerical approximation)
    /// f(t) = -d/dt ln(D(t)) = r(t) + t * r'(t)
    double inst_forward(double t, double dt = 0.001) const {
        if (t < dt) return zero_rate(dt);
        return forward_rate(t - dt/2, t + dt/2);
    }

    // Accessors
    const std::vector<double>& tenors() const { return tenors_; }
    const std::vector<double>& zeros() const { return zeros_; }
    size_t size() const { return tenors_.size(); }

private:
    std::vector<double> tenors_;
    std::vector<double> zeros_;
};

// ============================================================================
// Nelson-Siegel Model
// ============================================================================
//
// r(t) = b0 + b1 * g1(t/tau) + b2 * g2(t/tau)
//
// where:
//   g1(x) = (1 - exp(-x)) / x        — slope loading
//   g2(x) = (1 - exp(-x)) / x - exp(-x)  — curvature loading
//
// Properties:
//   r(0)  = b0 + b1            (short end)
//   r(inf) = b0                (long end)
//   Hump at t ~ tau when b2 != 0
// ============================================================================

struct NelsonSiegelParams {
    double beta0;   // Long-term level
    double beta1;   // Short-long spread
    double beta2;   // Curvature
    double tau;     // Decay factor

    NelsonSiegelParams()
        : beta0(0.05), beta1(-0.02), beta2(0.01), tau(2.0) {}

    NelsonSiegelParams(double b0, double b1, double b2, double t)
        : beta0(b0), beta1(b1), beta2(b2), tau(t) {}
};

/// Nelson-Siegel zero rate at maturity t
inline double nelson_siegel_rate(double t, const NelsonSiegelParams& p) {
    if (t <= 1e-10) return p.beta0 + p.beta1;  // limit as t->0

    double x = t / p.tau;
    double ex = std::exp(-x);
    // g1 = (1 - exp(-x)) / x
    double g1 = (1.0 - ex) / x;
    // g2 = g1 - exp(-x)
    double g2 = g1 - ex;

    return p.beta0 + p.beta1 * g1 + p.beta2 * g2;
}

/// Build a YieldCurve from Nelson-Siegel parameters
inline YieldCurve build_nelson_siegel_curve(
    const NelsonSiegelParams& p,
    const std::vector<double>& tenors = {0.25,0.5,1,2,3,5,7,10,15,20,30})
{
    std::vector<double> rates(tenors.size());
    for (size_t i = 0; i < tenors.size(); ++i)
        rates[i] = nelson_siegel_rate(tenors[i], p);
    return YieldCurve(tenors, rates);
}

// ============================================================================
// Svensson Extension
// ============================================================================

struct SvenssonParams {
    double beta0, beta1, beta2, beta3;
    double tau1, tau2;

    SvenssonParams()
        : beta0(0.05), beta1(-0.02), beta2(0.01), beta3(0.005),
          tau1(2.0), tau2(5.0) {}

    SvenssonParams(double b0, double b1, double b2, double b3, double t1, double t2)
        : beta0(b0), beta1(b1), beta2(b2), beta3(b3), tau1(t1), tau2(t2) {}
};

inline double svensson_rate(double t, const SvenssonParams& p) {
    if (t <= 1e-10) return p.beta0 + p.beta1;

    double x1 = t / p.tau1;
    double ex1 = std::exp(-x1);
    double g1 = (1.0 - ex1) / x1;

    double x2 = t / p.tau2;
    double ex2 = std::exp(-x2);

    return p.beta0
         + p.beta1 * g1
         + p.beta2 * (g1 - ex1)
         + p.beta3 * ((1.0 - ex2) / x2 - ex2);
}

inline YieldCurve build_svensson_curve(
    const SvenssonParams& p,
    const std::vector<double>& tenors = {0.25,0.5,1,2,3,5,7,10,15,20,30})
{
    std::vector<double> rates(tenors.size());
    for (size_t i = 0; i < tenors.size(); ++i)
        rates[i] = svensson_rate(tenors[i], p);
    return YieldCurve(tenors, rates);
}

// ============================================================================
// Yield Curve Bootstrapping
// ============================================================================
//
// Given market instruments with known prices, extract zero rates.
//
// Step 1: Short end — deposit rates give zero rates directly:
//   D(t) = 1 / (1 + r_dep * t)  →  r_zero = -ln(D(t)) / t
//
// Step 2: Longer tenors — par swap rates. A par swap with rate s_n has PV=0:
//   sum_{i=1}^{n} s_n * D(t_i) + D(t_n) = 1
//   Solve for D(t_n) given previously bootstrapped D(t_1)...D(t_{n-1}):
//   D(t_n) = (1 - s_n * sum_{i=1}^{n-1} D(t_i)) / (1 + s_n)
//
// This is the industry standard method for building curves.
// ============================================================================

struct BootstrapInstrument {
    double maturity;    // in years
    double rate;        // deposit rate or par swap rate
    bool is_deposit;    // true = deposit, false = swap (annual frequency)
};

inline YieldCurve bootstrap(const std::vector<BootstrapInstrument>& instruments) {
    // Sort by maturity
    auto sorted = instruments;
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.maturity < b.maturity; });

    std::vector<double> tenors, zeros;

    // Cumulative sum of D(t_i) for swap bootstrapping
    double disc_sum = 0.0;

    for (auto& inst : sorted) {
        double D;
        if (inst.is_deposit) {
            // Deposit: D(t) = 1 / (1 + r * t)  (simple compounding)
            D = 1.0 / (1.0 + inst.rate * inst.maturity);
        } else {
            // Par swap (annual frequency):
            // 1 = s * sum_{i=1}^{n} D(t_i) + D(t_n)
            // 1 = s * (disc_sum + D(t_n)) + D(t_n)
            // 1 = s * disc_sum + D(t_n) * (1 + s)
            // D(t_n) = (1 - s * disc_sum) / (1 + s)
            D = (1.0 - inst.rate * disc_sum) / (1.0 + inst.rate);
        }

        // Convert to continuous zero rate: r = -ln(D) / t
        double r = -std::log(D) / inst.maturity;

        tenors.push_back(inst.maturity);
        zeros.push_back(r);
        disc_sum += D;
    }

    return YieldCurve(tenors, zeros);
}

// ============================================================================
// Bond Pricing
// ============================================================================

struct Bond {
    double face;        // Face value (typically 100)
    double coupon;      // Annual coupon rate (e.g. 0.05 = 5%)
    double maturity;    // Years to maturity
    int freq;           // Coupon frequency (1=annual, 2=semi-annual)

    Bond() : face(100.0), coupon(0.05), maturity(10.0), freq(2) {}
    Bond(double f, double c, double m, int frq)
        : face(f), coupon(c), maturity(m), freq(frq) {}

    /// Generate cash flow schedule: {time, amount}
    std::vector<std::pair<double, double>> cashflows() const {
        std::vector<std::pair<double, double>> cfs;
        double dt = 1.0 / freq;
        double coupon_amt = face * coupon / freq;

        for (double t = dt; t <= maturity + 1e-10; t += dt) {
            double cf = coupon_amt;
            if (std::abs(t - maturity) < 1e-10)
                cf += face;  // Add principal at maturity
            cfs.push_back({std::min(t, maturity), cf});
        }
        return cfs;
    }
};

/// Price a bond from a yield curve (using discount factors)
inline double bond_price(const Bond& bond, const YieldCurve& curve) {
    double price = 0.0;
    for (auto& [t, cf] : bond.cashflows())
        price += cf * curve.discount(t);
    return price;
}

/// Price a bond at a flat yield (for yield-based calculations)
/// Uses the yield y with the bond's compounding frequency
inline double bond_price_at_yield(const Bond& bond, double y) {
    double price = 0.0;
    for (auto& [t, cf] : bond.cashflows()) {
        // Discount at flat yield (continuous compounding for consistency)
        price += cf * std::exp(-y * t);
    }
    return price;
}

/// Yield to maturity — solve for y such that price = bond_price_at_yield(y)
/// Uses Newton-Raphson with numerical derivative
inline double yield_to_maturity(const Bond& bond, double market_price,
                                 double y_init = 0.05, double tol = 1e-10,
                                 int max_iter = 100)
{
    double y = y_init;
    for (int iter = 0; iter < max_iter; ++iter) {
        double p = bond_price_at_yield(bond, y);
        double diff = p - market_price;
        if (std::abs(diff) < tol) return y;

        // Numerical derivative: dp/dy via central difference
        double dy = 1e-6;
        double dp = (bond_price_at_yield(bond, y + dy)
                   - bond_price_at_yield(bond, y - dy)) / (2.0 * dy);
        if (std::abs(dp) < 1e-15) break;

        y -= diff / dp;
    }
    return y;
}

// ============================================================================
// Duration, Convexity, DV01
// ============================================================================

struct BondRiskMetrics {
    double macaulay_duration;    // Weighted average time to cash flows
    double modified_duration;    // Price sensitivity: dP/P / dy
    double convexity;            // Curvature of price-yield relationship
    double dv01;                 // Dollar value of 1bp yield change
    double price;
    double ytm;
};

/// Compute all bond risk metrics at a given yield
inline BondRiskMetrics bond_risk(const Bond& bond, double y) {
    auto cfs = bond.cashflows();
    double price = 0.0;
    double dur_sum = 0.0;   // sum(t_i * CF_i * D(t_i))
    double conv_sum = 0.0;  // sum(t_i^2 * CF_i * D(t_i))

    for (auto& [t, cf] : cfs) {
        double pv = cf * std::exp(-y * t);
        price += pv;
        // Macaulay duration: weighted time
        dur_sum += t * pv;
        // Convexity: weighted time-squared
        conv_sum += t * t * pv;
    }

    // Macaulay Duration = sum(t * PV(CF)) / P
    double mac_dur = dur_sum / price;

    // Modified Duration = Macaulay / (1 + y/freq)
    // For continuous compounding: D_mod = D_mac
    // (since d/dy exp(-yt) = -t*exp(-yt), and D_mod = -1/P * dP/dy = D_mac)
    double mod_dur = mac_dur;  // continuous compounding

    // Convexity = sum(t^2 * PV(CF)) / P
    double convexity = conv_sum / price;

    // DV01 = |dP/dy| * 0.0001 ≈ D_mod * P * 0.0001
    double dv01 = mod_dur * price * 0.0001;

    return {mac_dur, mod_dur, convexity, dv01, price, y};
}

/// Duration-convexity approximation for price change
/// dP/P ≈ -D_mod * dy + 0.5 * C * dy^2
inline double price_change_approx(const BondRiskMetrics& m, double dy) {
    return m.price * (-m.modified_duration * dy
                      + 0.5 * m.convexity * dy * dy);
}

// ============================================================================
// Par Rate — the coupon rate that makes price = face value
// ============================================================================

/// Par rate for a given maturity and yield curve
/// par = (1 - D(T)) / sum_{i} D(t_i) * (1/freq)
inline double par_rate(const YieldCurve& curve, double maturity, int freq = 2) {
    double dt = 1.0 / freq;
    double disc_sum = 0.0;
    for (double t = dt; t <= maturity + 1e-10; t += dt)
        disc_sum += curve.discount(std::min(t, maturity)) * dt;

    double D_T = curve.discount(maturity);
    return (1.0 - D_T) / disc_sum;
}

// ============================================================================
// Curve Shift Scenarios
// ============================================================================

/// Parallel shift: add delta to all zero rates
inline YieldCurve parallel_shift(const YieldCurve& curve, double delta) {
    std::vector<double> new_zeros(curve.zeros().size());
    for (size_t i = 0; i < new_zeros.size(); ++i)
        new_zeros[i] = curve.zeros()[i] + delta;
    return YieldCurve(curve.tenors(), new_zeros);
}

/// Steepener: add -delta at short end, +delta at long end
inline YieldCurve steepener(const YieldCurve& curve, double delta) {
    std::vector<double> new_zeros(curve.zeros().size());
    double t_max = curve.tenors().back();
    for (size_t i = 0; i < new_zeros.size(); ++i) {
        double w = curve.tenors()[i] / t_max;  // 0 at short end, 1 at long end
        new_zeros[i] = curve.zeros()[i] + delta * (2.0 * w - 1.0);
    }
    return YieldCurve(curve.tenors(), new_zeros);
}

} // namespace fixed_income

#endif // QUANTPRICER_FIXED_INCOME_H
