// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "tests/Unit/Domain/DomainTestHelpers.hpp"

namespace {

template <size_t SpatialDim, typename DataType>
void test_euclidean_basis_vectors(const DataType& used_for_size) noexcept {
  for (const auto& direction : Direction<SpatialDim>::all_directions()) {
    auto expected =
        make_with_value<tnsr::i<DataType, SpatialDim>>(used_for_size, 0.0);
    expected.get(direction.axis()) =
        make_with_value<DataType>(used_for_size, direction.sign());

    CHECK_ITERABLE_APPROX((euclidean_basis_vector(direction, used_for_size)),
                          std::move(expected));
  }
}

}  //  namespace

SPECTRE_TEST_CASE("Unit.Domain.TestHelpers.BasisVector", "[Unit][Domain]") {
  const double d(std::numeric_limits<double>::signaling_NaN());
  test_euclidean_basis_vectors<1>(d);
  test_euclidean_basis_vectors<2>(d);
  test_euclidean_basis_vectors<3>(d);

  const DataVector dv(5);
  test_euclidean_basis_vectors<1>(dv);
  test_euclidean_basis_vectors<2>(dv);
  test_euclidean_basis_vectors<3>(dv);
}
