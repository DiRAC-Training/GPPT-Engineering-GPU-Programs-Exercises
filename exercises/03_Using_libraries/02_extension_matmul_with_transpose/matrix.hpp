#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <cstddef>
#include <cmath>
#include <random>

/**
 * @brief Simple matrix class wrapping contiguous storage for BLAS compatibility.
 *
 * Stores elements in row-major order in a single contiguous std::vector.
 * Provides 2D indexing via operator() and raw pointer access via data().
 */
class Matrix {
public:
    /**
     * @brief Constructs a matrix of given dimensions, initialized to zero.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     */
    Matrix(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

    /**
     * @brief Access element at (i, j) for reading and writing.
     *
     * @param i Row index.
     * @param j Column index.
     * @return Reference to the element.
     */
    double& operator()(size_t i, size_t j) {
        return data_[i * cols_ + j];
    }

    /**
     * @brief Access element at (i, j) for reading (const version).
     *
     * @param i Row index.
     * @param j Column index.
     * @return Const reference to the element.
     */
    const double& operator()(size_t i, size_t j) const {
        return data_[i * cols_ + j];
    }

    /**
     * @brief Returns pointer to underlying contiguous data.
     *
     * Use this when passing to BLAS functions that expect double*.
     *
     * @return Pointer to first element.
     */
    double* data() { return data_.data(); }

    /**
     * @brief Returns const pointer to underlying contiguous data.
     *
     * @return Const pointer to first element.
     */
    const double* data() const { return data_.data(); }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    /**
     * @brief Fills the matrix with random values in [0, 1).
     *
     * @param seed Random seed for reproducibility.
     */
    void randomize(unsigned int seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (auto& val : data_) {
            val = dist(gen);
        }
    }

    /**
     * @brief Sets all elements to zero.
     */
    void zero() {
        std::fill(data_.begin(), data_.end(), 0.0);
    }

    /**
     * @brief Compares two matrices for approximate equality.
     *
     * Uses a relative tolerance suitable for double precision arithmetic.
     * Two elements are considered equal if |a - b| <= tol * max(|a|, |b|, 1.0).
     *
     * @param other Matrix to compare against.
     * @param tol Relative tolerance (default 1e-10).
     * @return true if matrices have same dimensions and all elements match.
     */
    bool equals(const Matrix& other, double tol = 1e-10) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            return false;
        }
        for (size_t i = 0; i < data_.size(); i++) {
            double a = data_[i];
            double b = other.data_[i];
            double scale = std::fmax(std::fmax(std::fabs(a), std::fabs(b)), 1.0);
            if (std::fabs(a - b) > tol * scale) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Equality operator using default tolerance.
     */
    bool operator==(const Matrix& other) const {
        return equals(other);
    }

    bool operator!=(const Matrix& other) const {
        return !equals(other);
    }

private:
    size_t rows_;
    size_t cols_;
    std::vector<double> data_;
};

#endif // MATRIX_HPP
