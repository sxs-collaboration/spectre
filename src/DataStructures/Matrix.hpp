// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Matrix.

#pragma once

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
         double value = std::numeric_limits<double>::signaling_NaN());

  /// Default constructor needed for serialization
  Matrix() = default;

  double& operator()(size_t i, size_t j) noexcept {
    return data_[i + j * number_of_rows_];
  }
  const double& operator()(size_t i, size_t j) const noexcept {
    return data_[i + j * number_of_rows_];
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
