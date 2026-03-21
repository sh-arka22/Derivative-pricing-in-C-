// ============================================================================
// QuantPricer Library — Compilation Unit
// Most functionality is header-only (templates + inline functions).
// This file provides any non-inline implementations and library metadata.
// ============================================================================

#include "payoff/payoff.h"
#include "option/option.h"
#include "matrix/matrix.h"
#include "rng/rng.h"
#include "mc/monte_carlo.h"
#include "greeks/black_scholes.h"
#include "greeks/greeks_engine.h"
#include "vol/implied_vol.h"
#include "fdm/fdm.h"
#include "barrier/barrier.h"
#include "multi_asset/multi_asset.h"
#include "risk/risk.h"
#include "fixed_income/fixed_income.h"
#include "rates/rate_models.h"
#include "orderbook/orderbook.h"

namespace quantpricer {

const char* version() { return "1.0.0"; }
const char* description() {
    return "QuantPricer: Multi-Model Derivatives Pricing Engine";
}

} // namespace quantpricer
