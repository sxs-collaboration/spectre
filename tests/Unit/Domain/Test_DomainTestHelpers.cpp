// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace {

template <typename DataType, size_t SpatialDim>
tnsr::II<DataType, SpatialDim> random_inv_spatial_metric(
    const gsl::not_null<std::mt19937*> generator,
    const DataType& used_for_size) noexcept {
  std::uniform_real_distribution<> distribution(-0.05, 0.05);
  auto inv_spatial_metric =
      make_with_random_values<tnsr::II<DataType, SpatialDim>>(
          generator, make_not_null(&distribution), used_for_size);
  for (size_t d = 0; d < SpatialDim; ++d) {
    inv_spatial_metric.get(d, d) += 1.0;
  }
  return inv_spatial_metric;
}

template <size_t SpatialDim, typename DataType>
void test_euclidean_basis_vector(const DataType& used_for_size) noexcept {
  for (const auto& direction : Direction<SpatialDim>::all_directions()) {
    auto expected =
        make_with_value<tnsr::i<DataType, SpatialDim>>(used_for_size, 0.0);
    expected.get(direction.axis()) =
        make_with_value<DataType>(used_for_size, direction.sign());

    CHECK_ITERABLE_APPROX((euclidean_basis_vector(direction, used_for_size)),
                          std::move(expected));
  }
}

template <size_t SpatialDim, typename DataType>
void test_unit_basis_form(const DataType& used_for_size) noexcept {
  MAKE_GENERATOR(generator);
  const auto inv_spatial_metric =
      random_inv_spatial_metric<DataType, SpatialDim>(make_not_null(&generator),
                                                      used_for_size);
  for (const auto& direction : Direction<SpatialDim>::all_directions()) {
    const auto basis_form = unit_basis_form(direction, inv_spatial_metric);
    auto expected = euclidean_basis_vector(direction, used_for_size);
    const DataType norm = get(magnitude(expected, inv_spatial_metric));
    for (size_t d = 0; d < SpatialDim; ++d) {
      expected.get(d) /= norm;
    }
    CHECK_ITERABLE_APPROX(basis_form, expected);
    CHECK_ITERABLE_APPROX(get(magnitude(expected, inv_spatial_metric)),
                          make_with_value<DataType>(used_for_size, 1.0));
  }
}

}  //  namespace

SPECTRE_TEST_CASE("Unit.Domain.TestHelpers.BasisVector", "[Unit][Domain]") {
  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_euclidean_basis_vector, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_unit_basis_form, (1, 2, 3));
}
