// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Identity.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <size_t Dim, typename DataType>
void test_identity(const DataType& used_for_size) {
  const auto identity_matrix{identity<Dim>(used_for_size)};

  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      if (i == j) {
        CHECK_ITERABLE_APPROX(identity_matrix.get(i, j),
                              make_with_value<DataType>(used_for_size, 1.0));
      } else {
        CHECK_ITERABLE_APPROX(identity_matrix.get(i, j),
                              make_with_value<DataType>(used_for_size, 0.0));
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Identity",
                  "[DataStructures][Unit]") {
  const double d(std::numeric_limits<double>::signaling_NaN());
  test_identity<1>(d);
  test_identity<2>(d);
  test_identity<3>(d);

  const DataVector dv(5);
  test_identity<1>(dv);
  test_identity<2>(dv);
  test_identity<3>(dv);
}
