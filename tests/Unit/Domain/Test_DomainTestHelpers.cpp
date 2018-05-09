// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "tests/Unit/Domain/DomainTestHelpers.hpp"

namespace {

template <size_t SpatialDim, typename SpatialFrame>
void test_euclidean_basis_vectors(const DataVector& used_for_size) noexcept {
  for (const auto& direction : Direction<SpatialDim>::all_directions()) {
    auto expected =
        make_with_value<tnsr::i<DataVector, SpatialDim, SpatialFrame>>(
            used_for_size, 0.0);
    expected.get(direction.axis()) =
        make_with_value<DataVector>(used_for_size, direction.sign());

    CHECK_ITERABLE_APPROX((euclidean_basis_vector<SpatialDim, SpatialFrame>(
                              direction, used_for_size)),
                          std::move(expected));
  }
}

}  //  namespace

SPECTRE_TEST_CASE("Unit.Domain.TestHelpers.BasisVector", "[Unit][Domain]") {
  const DataVector dv(5);

  test_euclidean_basis_vectors<1, Frame::Inertial>(dv);
  test_euclidean_basis_vectors<2, Frame::Inertial>(dv);
  test_euclidean_basis_vectors<3, Frame::Inertial>(dv);

  test_euclidean_basis_vectors<1, Frame::Grid>(dv);
  test_euclidean_basis_vectors<2, Frame::Grid>(dv);
  test_euclidean_basis_vectors<3, Frame::Grid>(dv);
}
