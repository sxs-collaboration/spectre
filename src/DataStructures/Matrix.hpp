// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Matrix.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <limits>
#include <vector>

namespace PUP {
class er;
}  // namespace PUP

/*!
 * \ingroup DataStructuresGroup
 * \brief A dynamically sized matrix.
 *
 * \note The data layout is column-major (0-based)
 */
class Matrix {
 public:
  /// Create and set each element to value
  Matrix(size_t number_of_rows, size_t number_of_colums,
         double value = std::numeric_limits<double>::signaling_NaN()) noexcept;

  /// Create a matrix from a two-dimensional vector of rows x columns. Requires
  /// that all rows have the same size.
  explicit Matrix(const std::vector<std::vector<double>>& rows) noexcept;

  /// Default constructor needed for serialization
  Matrix() noexcept = default;

  double& operator()(size_t row, size_t column) noexcept {
    return data_[row + column * number_of_rows_];
  }
  const double& operator()(size_t row, size_t column) const noexcept {
    return data_[row + column * number_of_rows_];
  }

  double* data() noexcept { return data_.data(); }
  const double* data() const noexcept { return data_.data(); }

  decltype(auto) begin() { return data_.begin(); }
  decltype(auto) begin() const { return data_.begin(); }

  decltype(auto) end() { return data_.end(); }
  decltype(auto) end() const { return data_.end(); }

  size_t rows() const noexcept { return number_of_rows_; }
  size_t columns() const noexcept { return number_of_columns_; }

  /// Charm++ serialization
  void pup(PUP::er& p);  // NOLINT

 private:
  size_t number_of_rows_{0};
  size_t number_of_columns_{0};
  std::vector<double> data_;
};

/// Stream operator for Matrix
std::ostream& operator<<(std::ostream& os, const Matrix& m);

/// Equivalence operator for Matrix
bool operator==(const Matrix& lhs, const Matrix& rhs);

/// Inequivalence operator for Matrix
bool operator!=(const Matrix& lhs, const Matrix& rhs);
