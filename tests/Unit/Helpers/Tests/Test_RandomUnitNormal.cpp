// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/RandomUnitNormal.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

template <size_t Dim, typename DataType>
void test_random_unit_normal(const gsl::not_null<std::mt19937*> generator,
                             const DataType& used_for_size) {
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<Dim, DataType>(generator,
                                                            used_for_size);
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
