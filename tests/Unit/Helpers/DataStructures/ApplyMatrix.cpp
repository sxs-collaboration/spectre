// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/DataStructures/ApplyMatrix.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

DataVector apply_matrix(const Matrix& m, const DataVector& v) {
  ASSERT(m.columns() == v.size(), "Bad apply_matrix");
  DataVector result(m.rows(), 0.);
  for (size_t i = 0; i < m.rows(); ++i) {
    for (size_t j = 0; j < m.columns(); ++j) {
      result[i] += m(i, j) * v[j];
    }
  }
  return result;
}
