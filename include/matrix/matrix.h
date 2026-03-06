#ifndef QUANTPRICER_MATRIX_H
#define QUANTPRICER_MATRIX_H

// ============================================================================
// Matrix Template Class — Book Chapters 5, 8, 9
// Ch 5:  Generic programming, template classes, default types
// Ch 8:  Custom matrix class with STL storage, operator overloading,
//        mathematical operations (add, multiply, transpose)
// Ch 9:  LU decomposition, Thomas algorithm, Cholesky decomposition,
//        QR decomposition — all for solving linear systems in FDM
// ============================================================================

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

/// Template matrix class using STL vector<vector<T>> storage — Ch 5.6, Ch 8.1
template <typename T = double>
class QMatrix {
public:
    // === Constructors — Ch 5.7, Ch 8.1.5 ===
    QMatrix() : rows_(0), cols_(0) {}

    QMatrix(size_t rows, size_t cols, const T& init = T())
        : rows_(rows), cols_(cols), data_(rows, std::vector<T>(cols, init)) {}

    // Copy constructor — Ch 3.3
    QMatrix(const QMatrix& rhs) = default;
    QMatrix& operator=(const QMatrix& rhs) = default;
    QMatrix(QMatrix&&) noexcept = default;
    QMatrix& operator=(QMatrix&&) noexcept = default;

    // === Element Access — Ch 8.1.3 ===
    T& operator()(size_t r, size_t c) { return data_[r][c]; }
    const T& operator()(size_t r, size_t c) const { return data_[r][c]; }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    // === Arithmetic Operators — Ch 8.1.6 ===
    QMatrix operator+(const QMatrix& rhs) const {
        check_dims(rhs, "addition");
        QMatrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                result(i, j) = data_[i][j] + rhs(i, j);
        return result;
    }

    QMatrix operator-(const QMatrix& rhs) const {
        check_dims(rhs, "subtraction");
        QMatrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                result(i, j) = data_[i][j] - rhs(i, j);
        return result;
    }

    /// Matrix-matrix multiplication — Ch 8.2.7
    QMatrix operator*(const QMatrix& rhs) const {
        if (cols_ != rhs.rows())
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        QMatrix result(rows_, rhs.cols(), T());
        for (size_t i = 0; i < rows_; ++i)
            for (size_t k = 0; k < cols_; ++k)
                for (size_t j = 0; j < rhs.cols(); ++j)
                    result(i, j) += data_[i][k] * rhs(k, j);
        return result;
    }

    /// Scalar multiplication — Ch 8.2.5
    QMatrix operator*(const T& scalar) const {
        QMatrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                result(i, j) = data_[i][j] * scalar;
        return result;
    }

    /// Matrix-vector multiply — Ch 8.2.7
    std::vector<T> operator*(const std::vector<T>& vec) const {
        if (cols_ != vec.size())
            throw std::invalid_argument("Matrix-vector dimension mismatch");
        std::vector<T> result(rows_, T());
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                result[i] += data_[i][j] * vec[j];
        return result;
    }

    /// Transpose — Ch 8.2.6
    QMatrix transpose() const {
        QMatrix result(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                result(j, i) = data_[i][j];
        return result;
    }

    /// Frobenius norm — Ch 8.2.9
    T frobenius_norm() const {
        T sum = T();
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                sum += data_[i][j] * data_[i][j];
        return std::sqrt(sum);
    }

    /// Identity matrix factory
    static QMatrix identity(size_t n) {
        QMatrix I(n, n, T());
        for (size_t i = 0; i < n; ++i) I(i, i) = T(1);
        return I;
    }

    void print(int precision = 6) const {
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j)
                std::cout << std::setw(precision + 4) << std::setprecision(precision)
                          << data_[i][j] << " ";
            std::cout << "\n";
        }
    }

private:
    size_t rows_, cols_;
    std::vector<std::vector<T>> data_;

    void check_dims(const QMatrix& rhs, const char* op) const {
        if (rows_ != rhs.rows() || cols_ != rhs.cols())
            throw std::invalid_argument(std::string("Dimension mismatch for ") + op);
    }
};

// ============================================================================
// Numerical Linear Algebra Solvers — Chapter 9
// ============================================================================

/// LU Decomposition with partial pivoting — Ch 9.2
/// Solves Ax = b by decomposing A = LU, then forward/back substitution
template <typename T = double>
std::vector<T> solve_lu(QMatrix<T> A, std::vector<T> b) {
    size_t n = A.rows();
    if (n != A.cols() || n != b.size())
        throw std::invalid_argument("LU: dimension mismatch");

    std::vector<size_t> pivot(n);
    std::iota(pivot.begin(), pivot.end(), 0);

    // Forward elimination with partial pivoting — Ch 9.2.1
    for (size_t k = 0; k < n - 1; ++k) {
        // Find pivot
        size_t max_row = k;
        T max_val = std::abs(A(k, k));
        for (size_t i = k + 1; i < n; ++i) {
            if (std::abs(A(i, k)) > max_val) {
                max_val = std::abs(A(i, k));
                max_row = i;
            }
        }
        if (max_val < 1e-14)
            throw std::runtime_error("LU: singular or near-singular matrix");

        if (max_row != k) {
            std::swap(pivot[k], pivot[max_row]);
            for (size_t j = 0; j < n; ++j)
                std::swap(A(k, j), A(max_row, j));
            std::swap(b[k], b[max_row]);
        }

        for (size_t i = k + 1; i < n; ++i) {
            T factor = A(i, k) / A(k, k);
            A(i, k) = factor;  // Store L below diagonal
            for (size_t j = k + 1; j < n; ++j)
                A(i, j) -= factor * A(k, j);
            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    std::vector<T> x(n);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        x[i] = b[i];
        for (size_t j = i + 1; j < n; ++j)
            x[i] -= A(i, j) * x[j];
        x[i] /= A(i, i);
    }
    return x;
}

/// Thomas Algorithm for tridiagonal systems — Ch 9.3
/// Solves [a_i, b_i, c_i] * x = d where a=sub, b=main, c=super diagonal
/// Critical for FDM implicit/Crank-Nicolson schemes (Ch 17)
template <typename T = double>
std::vector<T> solve_thomas(std::vector<T> a, std::vector<T> b,
                            std::vector<T> c, std::vector<T> d) {
    size_t n = b.size();
    if (a.size() != n - 1 || c.size() != n - 1 || d.size() != n)
        throw std::invalid_argument("Thomas: dimension mismatch");

    // Forward sweep — Ch 9.3.1
    for (size_t i = 1; i < n; ++i) {
        T w = a[i - 1] / b[i - 1];
        b[i] -= w * c[i - 1];
        d[i] -= w * d[i - 1];
    }

    // Back substitution
    std::vector<T> x(n);
    x[n - 1] = d[n - 1] / b[n - 1];
    for (int i = static_cast<int>(n) - 2; i >= 0; --i) {
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i];
    }
    return x;
}

/// Cholesky Decomposition — Ch 9.4
/// For SPD matrices: A = LL^T. Used in Ch 16 for correlated Brownian motion.
template <typename T = double>
QMatrix<T> cholesky(const QMatrix<T>& A) {
    size_t n = A.rows();
    if (n != A.cols())
        throw std::invalid_argument("Cholesky: matrix must be square");

    QMatrix<T> L(n, n, T());

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            T sum = T();
            for (size_t k = 0; k < j; ++k)
                sum += L(i, k) * L(j, k);

            if (i == j) {
                T diag = A(i, i) - sum;
                if (diag <= T())
                    throw std::runtime_error("Cholesky: matrix is not positive definite");
                L(i, j) = std::sqrt(diag);
            } else {
                L(i, j) = (A(i, j) - sum) / L(j, j);
            }
        }
    }
    return L;
}

#endif // QUANTPRICER_MATRIX_H
