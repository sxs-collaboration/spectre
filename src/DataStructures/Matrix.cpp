// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Matrix.hpp"

#include <ostream>
#include <pup.h>
#include <pup_stl.h>

Matrix::Matrix(const size_t number_of_rows, const size_t number_of_colums,
               const double value)
    : number_of_rows_(number_of_rows),
      number_of_columns_(number_of_colums),
      data_(number_of_rows_ * number_of_columns_, value) {}

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
