// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"
#include "tests/Utilities/RandomUnitNormal.hpp"

namespace {

template <typename DataType, size_t Dim>
tnsr::ii<DataType, Dim> random_spatial_metric(
    const gsl::not_null<std::mt19937*> generator,
    const DataType& used_for_size) noexcept {
  std::uniform_real_distribution<> distribution(-0.05, 0.05);
  auto spatial_metric = make_with_random_values<tnsr::ii<DataType, Dim>>(
      generator, make_not_null(&distribution), used_for_size);
  for (size_t d = 0; d < Dim; ++d) {
    spatial_metric.get(d, d) += 1.0;
  }
  return spatial_metric;
}

template <size_t Dim, typename DataType>
void test_random_unit_normal(const gsl::not_null<std::mt19937*> generator,
                             const DataType& used_for_size) noexcept {
  const auto spatial_metric =
      random_spatial_metric<DataType, Dim>(generator, used_for_size);
  const auto unit_normal = random_unit_normal(generator, spatial_metric);
  const auto expected_magnitude =
      make_with_value<Scalar<DataType>>(used_for_size, 1.0);
  const auto magnitude_of_unit_normal = magnitude(unit_normal, spatial_metric);
  CHECK_ITERABLE_APPROX(expected_magnitude, magnitude_of_unit_normal);
}
}  // namespace

SPECTRE_TEST_CASE("Test.TestHelpers.RandomUnitNormal", "[Unit]") {
  MAKE_GENERATOR(generator);

  const double d = std::numeric_limits<double>::signaling_NaN();
  test_random_unit_normal<1>(&generator, d);
  test_random_unit_normal<2>(&generator, d);
  test_random_unit_normal<3>(&generator, d);

  const DataVector dv(5);
  test_random_unit_normal<1>(&generator, dv);
  test_random_unit_normal<2>(&generator, dv);
  test_random_unit_normal<3>(&generator, dv);
}
