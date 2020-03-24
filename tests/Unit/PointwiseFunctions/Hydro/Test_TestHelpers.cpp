// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/TestHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim, typename DataType>
void test_spatial_velocity(const gsl::not_null<std::mt19937*> generator,
                           const DataType& used_for_size) noexcept {
  const auto metric =
      TestHelpers::gr::random_spatial_metric<Dim>(generator, used_for_size);
  const auto lorentz_factor =
      TestHelpers::hydro::random_lorentz_factor(generator, used_for_size);
  const auto spatial_velocity =
      TestHelpers::hydro::random_velocity(generator, lorentz_factor, metric);
  CHECK_ITERABLE_APPROX(
      get(dot_product(spatial_velocity, spatial_velocity, metric)),
      1.0 - 1.0 / square(get(lorentz_factor)));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.TestHelpers",
                  "[Unit][Hydro]") {
  MAKE_GENERATOR(generator);
  const double d = std::numeric_limits<double>::signaling_NaN();
  test_spatial_velocity<1>(&generator, d);
  test_spatial_velocity<2>(&generator, d);
  test_spatial_velocity<3>(&generator, d);
  const DataVector dv(5);
  test_spatial_velocity<1>(&generator, dv);
  test_spatial_velocity<2>(&generator, dv);
  test_spatial_velocity<3>(&generator, dv);
}
