// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Matrix.hpp"

#include <algorithm>
#include <memory>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>

#include "ErrorHandling/Assert.hpp"

Matrix::Matrix(const size_t number_of_rows, const size_t number_of_colums,
               const double value) noexcept
    : number_of_rows_(number_of_rows),
      number_of_columns_(number_of_colums),
      data_(number_of_rows_ * number_of_columns_, value) {}

Matrix::Matrix(const std::vector<std::vector<double>>& rows) noexcept {
  number_of_rows_ = rows.size();
  if (number_of_rows_ > 0) {
    number_of_columns_ = rows[0].size();
    ASSERT(std::all_of(rows.begin(), rows.end(),
                       [this](const std::vector<double>& row) noexcept {
                         return row.size() == number_of_columns_;
                       }),
           "All rows of a Matrix must be of the same size.");
    // `Matrix` stores the data in column-major order, i.e. the `()` operator
    // assumes that `data_` is constructed out of concatenated columns, not
    // rows. Therefore we can't data_.insert() but need to transpose, using the
    // `()` operator for that.
    data_.resize(number_of_rows_ * number_of_columns_);
    for (size_t i = 0; i < number_of_rows_; i++) {
      for (size_t j = 0; j < number_of_columns_; j++) {
        operator()(i, j) = rows[i][j];
      }
    }
  }
}

void Matrix::pup(PUP::er& p) {  // NOLINT
  p | number_of_rows_;
  p | number_of_columns_;
  p | data_;
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
  for (size_t i = 0; i < m.rows(); ++i) {
    os << "{ ";
    for (size_t j = 0; j < m.columns(); ++j) {
      os << m(i, j) << " ";
    }
    os << "}\n";
  }
  return os;
}

bool operator==(const Matrix& lhs, const Matrix& rhs) {
  return lhs.columns() == rhs.columns() and lhs.rows() == rhs.rows() and
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool operator!=(const Matrix& lhs, const Matrix& rhs) {
  return not(lhs == rhs);
}
