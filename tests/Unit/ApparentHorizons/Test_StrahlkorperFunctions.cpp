// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "ApparentHorizons/StrahlkorperFunctions.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Helpers/ApparentHorizons/StrahlkorperTestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/YlmTestFunctions.hpp"

namespace Frame {
struct Inertial;
}  // namespace Frame

namespace {
void test_radius() {
  const double y11_amplitude = 1.0;
  const double radius = 2.0;
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const auto strahlkorper =
      create_strahlkorper_y11(y11_amplitude, radius, center);

  // Now construct a Y00 + Im(Y11) surface by hand.
  const auto& theta_points = strahlkorper.ylm_spherepack().theta_points();
  const auto& phi_points = strahlkorper.ylm_spherepack().phi_points();
  const auto n_pts = theta_points.size() * phi_points.size();
  YlmTestFunctions::Y11 y_11;
  DataVector expected_radius{n_pts};
  y_11.func(&expected_radius, 1, 0, theta_points, phi_points);
  for (size_t s = 0; s < n_pts; ++s) {
    expected_radius[s] *= y11_amplitude;
    expected_radius[s] += radius;
  }

  const DataVector strahlkorper_radius{
      get(StrahlkorperFunctions::radius(strahlkorper))};
  CHECK_ITERABLE_APPROX(strahlkorper_radius, expected_radius);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperFunctions",
                  "[ApparentHorizons][Unit]") {
  test_radius();
}
