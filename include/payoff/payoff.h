#ifndef QUANTPRICER_PAYOFF_H
#define QUANTPRICER_PAYOFF_H

// ============================================================================
// PayOff Hierarchy — Book Chapters 3-4
// Ch 3: OOP, classes, constructors, selectors/modifiers, pass-by-ref-to-const
// Ch 4: Inheritance, abstract base classes, pure virtual methods, virtual
//        destructors, operator() overloading (functor pattern)
// ============================================================================

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

/// Abstract base class for all option payoff functions.
/// Uses the functor pattern (operator()) so payoffs can be called like functions.
/// This is the core of the inheritance hierarchy from Ch 4.3.
class PayOff {
public:
    PayOff() = default;
    virtual ~PayOff() = default;                           // Ch 4.9: Virtual destructor
    virtual double operator()(double spot) const = 0;      // Pure virtual — Ch 4.4
    virtual std::unique_ptr<PayOff> clone() const = 0;     // Prototype pattern for polymorphic copy
};

/// European Call: max(S - K, 0) — Ch 4.5, 4.7
class PayOffCall : public PayOff {
public:
    explicit PayOffCall(double strike) : K_(strike) {
        if (K_ < 0.0) throw std::invalid_argument("Strike must be non-negative");
    }
    double operator()(double spot) const override {
        return std::max(spot - K_, 0.0);
    }
    std::unique_ptr<PayOff> clone() const override {
        return std::make_unique<PayOffCall>(*this);
    }
    double strike() const { return K_; }
private:
    double K_;
};

/// European Put: max(K - S, 0)
class PayOffPut : public PayOff {
public:
    explicit PayOffPut(double strike) : K_(strike) {
        if (K_ < 0.0) throw std::invalid_argument("Strike must be non-negative");
    }
    double operator()(double spot) const override {
        return std::max(K_ - spot, 0.0);
    }
    std::unique_ptr<PayOff> clone() const override {
        return std::make_unique<PayOffPut>(*this);
    }
    double strike() const { return K_; }
private:
    double K_;
};

/// Digital Call: 1 if S > K, else 0 — Ch 4.5 extension
class PayOffDigitalCall : public PayOff {
public:
    explicit PayOffDigitalCall(double strike) : K_(strike) {}
    double operator()(double spot) const override {
        return (spot > K_) ? 1.0 : 0.0;
    }
    std::unique_ptr<PayOff> clone() const override {
        return std::make_unique<PayOffDigitalCall>(*this);
    }
private:
    double K_;
};

/// Digital Put: 1 if S < K, else 0
class PayOffDigitalPut : public PayOff {
public:
    explicit PayOffDigitalPut(double strike) : K_(strike) {}
    double operator()(double spot) const override {
        return (spot < K_) ? 1.0 : 0.0;
    }
    std::unique_ptr<PayOff> clone() const override {
        return std::make_unique<PayOffDigitalPut>(*this);
    }
private:
    double K_;
};

/// Double Digital: 1 if D <= S <= U, else 0 — Ch 4.6, 4.8
class PayOffDoubleDigital : public PayOff {
public:
    PayOffDoubleDigital(double lower, double upper)
        : D_(lower), U_(upper) {
        if (D_ >= U_) throw std::invalid_argument("Lower barrier must be < upper barrier");
    }
    double operator()(double spot) const override {
        return (spot >= D_ && spot <= U_) ? 1.0 : 0.0;
    }
    std::unique_ptr<PayOff> clone() const override {
        return std::make_unique<PayOffDoubleDigital>(*this);
    }
private:
    double D_;
    double U_;
};

/// Power Option: max(S^alpha - K, 0) — extension beyond the book
class PayOffPower : public PayOff {
public:
    PayOffPower(double strike, double alpha) : K_(strike), alpha_(alpha) {}
    double operator()(double spot) const override {
        return std::max(std::pow(spot, alpha_) - K_, 0.0);
    }
    std::unique_ptr<PayOff> clone() const override {
        return std::make_unique<PayOffPower>(*this);
    }
private:
    double K_;
    double alpha_;
};

#endif // QUANTPRICER_PAYOFF_H
