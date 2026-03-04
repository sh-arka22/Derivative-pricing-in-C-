# QuantPricer — Multi-Model Derivatives Pricing Engine

A production-grade C++17 derivatives pricing library implementing Monte Carlo, Finite Difference, and analytical methods for European, exotic, jump-diffusion, and stochastic volatility models. Built from first principles with clean OOP architecture, custom numerical solvers, and comprehensive test coverage.

## Why This Project Matters for HFT/Quant Roles

| Signal | What It Demonstrates |
|---|---|
| **C++17 mastery** | Templates, RAII, move semantics, smart pointers, `constexpr`, STL algorithms |
| **Numerical methods** | Monte Carlo, FDM (explicit/implicit/Crank-Nicolson), root-finding, matrix decompositions |
| **Financial mathematics** | Black-Scholes, Greeks, Heston, Merton jump-diffusion, implied volatility surfaces |
| **Software architecture** | Inheritance hierarchies, polymorphism, design patterns (Strategy, Prototype, Template Method) |
| **Testing discipline** | 43 automated tests validating pricing accuracy, convergence, and numerical stability |
| **Performance awareness** | Variance reduction, antithetic variates, solver convergence analysis, timing benchmarks |

## Architecture

```
quantpricer/
├── include/
│   ├── payoff/payoff.h          # PayOff hierarchy (abstract base, call, put, digital, power)
│   ├── option/option.h          # Option, Heston, Merton, FDM parameter structs
│   ├── matrix/matrix.h          # Template matrix + LU, Thomas, Cholesky solvers
│   ├── rng/rng.h                # RNG hierarchy (LCG, MT19937) + statistical distributions
│   ├── mc/monte_carlo.h         # MC engine: European, Asian, jump-diffusion, Heston
│   ├── greeks/black_scholes.h   # Analytic BS pricing + all Greeks
│   ├── greeks/greeks_engine.h   # FD bump-and-reprice + MC pathwise Greeks
│   ├── vol/implied_vol.h        # Implied vol: bisection, Newton-Raphson, vol surface
│   └── fdm/fdm.h                # PDE solver: explicit, implicit, Crank-Nicolson
├── src/quantpricer.cpp          # Library compilation unit
├── examples/
│   ├── main_demo.cpp            # Full showcase of all modules
│   ├── mc_european.cpp          # European MC convergence study
│   ├── mc_exotic.cpp            # Asian options + jump-diffusion
│   ├── fdm_solver.cpp           # FDM pricing with CSV output
│   ├── greeks_engine.cpp        # 3-method Greeks comparison
│   ├── vol_surface.cpp          # Implied volatility surface from Heston
│   ├── heston_pricer.cpp        # Heston sensitivity analysis
│   └── jump_diffusion.cpp       # Jump intensity parameter study
├── tests/test_runner.cpp        # 43 automated validation tests
├── CMakeLists.txt               # CMake build configuration
└── Makefile                     # Simple make-based build
```

## Chapter-to-Code Mapping

Every module directly implements concepts from **"C++ For Quantitative Finance"** by Michael Halls-Moore (QuantStart):

| Chapter | Topic | Files | Key Concepts Implemented |
|---|---|---|---|
| **3** | First QF C++ Program | `payoff.h`, `option.h` | VanillaOption class, OOP, constructors, const correctness, pass-by-ref |
| **4** | PayOff Hierarchies & Inheritance | `payoff.h` | Abstract base class, pure virtual `operator()`, virtual destructors, polymorphic cloning |
| **5** | Generic Programming & Templates | `matrix.h` | `QMatrix<T>` template class, default template parameters, template specialization |
| **6** | Standard Template Library | Throughout | `std::vector`, `std::function`, iterators, `<algorithm>`, `<numeric>` |
| **7** | Function Objects | `greeks_engine.h`, `implied_vol.h` | Functor pattern (`operator()`), `std::function`, lambda callbacks |
| **8** | Matrix Classes | `matrix.h` | Custom matrix with STL storage, operator overloading (+, -, *, transpose), Frobenius norm |
| **9** | Numerical Linear Algebra | `matrix.h` | LU decomposition (partial pivoting), Thomas tridiagonal algorithm, Cholesky decomposition |
| **10** | European MC | `monte_carlo.h`, `black_scholes.h` | Risk-neutral GBM simulation, antithetic variates, analytic BS benchmark |
| **11** | Greeks | `greeks_engine.h` | Analytic formulae (Δ,Γ,ν,Θ,ρ), FD bump-and-reprice, MC pathwise (IPA) |
| **12** | Asian/Path-Dependent MC | `monte_carlo.h` | Path generation, arithmetic/geometric averaging, OOP option design |
| **13** | Implied Volatility | `implied_vol.h` | Interval bisection, Newton-Raphson with Vega, volatility surface construction |
| **14** | Random Number Generation | `rng.h` | RNG class hierarchy, LCG implementation, Mersenne Twister, Box-Muller, inverse CDF |
| **15** | Jump-Diffusion Models | `monte_carlo.h`, `option.h` | Merton model, Poisson jumps, jump compensator, log-normal jump sizes |
| **16** | Stochastic Volatility | `monte_carlo.h`, `option.h` | Heston model, Euler discretisation, correlated Brownians via Cholesky, Feller condition |
| **17** | Finite Difference Methods | `fdm.h` | BS PDE discretisation, explicit/implicit/Crank-Nicolson theta-scheme, Thomas solver integration, grid convergence |

## Build & Run

### Requirements
- C++17 compiler (GCC 8+, Clang 7+, MSVC 2019+)
- No external dependencies — header-only design with STL only

### Using Make
```bash
make release          # Build everything with -O3
make run-demo         # Build and run full demo
make run-tests        # Build and run test suite
make clean            # Clean build artifacts
```

### Using CMake
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./quantpricer_demo    # Full demo
./test_runner          # Test suite
```

### Manual Compilation
```bash
g++ -std=c++17 -O3 -I include src/quantpricer.cpp examples/main_demo.cpp -o demo -lm
./demo
```

## Key Results

### Pricing Accuracy
| Model | MC Price | Analytic/Ref | Std Error | Paths |
|---|---|---|---|---|
| European Call (GBM) | 10.4413 | 10.4506 | 0.0146 | 500,000 |
| Arithmetic Asian | 5.7750 | — | 0.0179 | 200,000 |
| Heston (ρ=-0.7) | 10.4080 | 10.4506 (BS) | 0.0274 | 200,000 |
| FDM Crank-Nicolson | 10.4544 | 10.4506 | — | 200×1000 grid |

### Greeks Validation (3 independent methods)
| Greek | Analytic | Finite Diff | MC Pathwise |
|---|---|---|---|
| Delta | 0.6368 | 0.6367 | 0.6367 |
| Gamma | 0.0188 | 0.0188 | — |
| Vega | 37.524 | 37.524 | 37.482 |

### Implied Volatility Recovery
| Method | σ Recovered | Iterations | Time |
|---|---|---|---|
| Bisection | 0.25000000 | 33 | 2.7 μs |
| Newton-Raphson | 0.25000000 | 2 | 0.2 μs |

### FDM Grid Convergence (Crank-Nicolson)
| N_space | N_time | Error vs Analytic |
|---|---|---|
| 50 | 250 | 0.063 |
| 200 | 1000 | 0.004 |
| 800 | 4000 | 0.0002 |

## Design Decisions

1. **Header-heavy design**: Most code is in headers for template support and inlining — matches how quant libraries like Eigen and QuantLib are structured.

2. **Smart pointers for ownership**: `std::unique_ptr<PayOff>` with `clone()` for polymorphic copying — no raw `new`/`delete` anywhere.

3. **Template matrix class**: `QMatrix<T>` works with any numeric type — enables potential future extension to `std::complex<double>` for Fourier pricing.

4. **Theta-scheme FDM**: Unified solver handles explicit (θ=0), Crank-Nicolson (θ=0.5), and implicit (θ=1) via a single parameter — cleaner than separate solver classes.

5. **Three independent Greeks methods**: Validates correctness through cross-comparison — if analytic, FD, and MC agree, the implementation is almost certainly correct.

## Suggested 2-3 Week Development Timeline

| Week | Focus | Deliverables |
|---|---|---|
| **Week 1** | Core infrastructure | PayOff hierarchy, Matrix class, RNG, BS analytics, basic MC |
| **Week 2** | Advanced models | Asian MC, Heston, Merton jump-diffusion, FDM solver, Greeks engine |
| **Week 3** | Polish & extensions | Implied vol surface, test suite, convergence analysis, README, profiling |

## License

MIT
