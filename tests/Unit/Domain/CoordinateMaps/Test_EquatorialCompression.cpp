// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <pup.h>
#include <random>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/EquatorialCompression.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace domain {
namespace {
void test_suite() {  // Set up random number generator
  INFO("Suite");
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> aspect_ratio_dis(0.1, 10);

  const double aspect_ratio = aspect_ratio_dis(gen);
  CAPTURE_PRECISE(aspect_ratio);
  const CoordinateMaps::EquatorialCompression angular_compression_map(
      aspect_ratio);
  test_suite_for_map_on_unit_cube(angular_compression_map);
}

void test_radius() {
  INFO("Radius");
  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-10, 10);
  std::uniform_real_distribution<> aspect_ratio_dis(0.1, 10);
  const double x = real_dis(gen);
  CAPTURE_PRECISE(x);
  const double y = real_dis(gen);
  CAPTURE_PRECISE(y);
  const double z = real_dis(gen);
  CAPTURE_PRECISE(z);
  const double aspect_ratio = aspect_ratio_dis(gen);
  CAPTURE_PRECISE(aspect_ratio);
  const std::array<double, 3> input_point{{x, y, z}};
  const CoordinateMaps::EquatorialCompression angular_compression_map(
      aspect_ratio);
  const auto result_point = angular_compression_map(input_point);
  const auto& result_x = result_point[0];
  const auto& result_y = result_point[1];
  const auto& result_z = result_point[2];

  // The EquatorialCompression map should preserve radial lengths.
  CHECK(magnitude(result_point) == approx(magnitude(input_point)));

  // It is easy to mix up \alpha with one over \alpha,
  // so this comment serves as a reminder to future readers
  // as to which direction the \alpha should go in the
  // CHECK statement.
  //
  // The aspect ratio is the ratio of the horizontal length
  // to the vertical height. The parameter name `aspect_ratio` is
  // used because points with tan \theta = 1 get mapped to points
  // with tan \theta = \alpha, the value passed to `aspect_ratio`.
  //
  // The angular compression map is:
  //  tan \theta' = \alpha tan\theta.
  // Then, if the argument provided to the EquatorialCompression
  // map is, for example, 4, points on the undistorted sphere
  // that have an angle tan \theta = sqrt(x^2 + y^2)/z are
  // mapped to points that have an angle
  //  tan \theta' = 4*sqrt(x^2+y^2)/z
  //              = sqrt(x'^2 + y'^2)/z'.
  CHECK(aspect_ratio * sqrt(pow<2>(x) + pow<2>(y)) / z ==
        approx(sqrt(pow<2>(result_x) + pow<2>(result_y)) / result_z));
}

void test_is_identity() {
  INFO("Is identity");
  check_if_map_is_identity(CoordinateMaps::EquatorialCompression{1.0});
  CHECK(not CoordinateMaps::EquatorialCompression{0.9}.is_identity());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.EquatorialCompression",
                  "[Domain][Unit]") {
  test_suite();
  test_radius();
  test_is_identity();
}
// [[OutputRegex, The aspect_ratio must be greater than zero.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Domain.CoordinateMaps.EquatorialCompression.Assert2",
    "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_angular_compression = CoordinateMaps::EquatorialCompression(-0.2);
  static_cast<void>(failed_angular_compression);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
}  // namespace domain
