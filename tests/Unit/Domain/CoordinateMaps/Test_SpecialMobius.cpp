// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/optional/optional.hpp>
#include <cmath>
#include <memory>
#include <pup.h>
#include <random>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/SpecialMobius.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace domain {
namespace {
void test_suite() {  // Set up random number generator
  INFO("Suite");
  MAKE_GENERATOR(gen);
  // Note: Empirically we have found that the map is accurate
  // to 12 decimal places for mu = 0.96.
  // test_suite demands more accuracy than 12 decimal places
  // so we narrow the range of mu correspondingly.
  std::uniform_real_distribution<> mu_dis(-0.85, 0.85);

  const double mu = mu_dis(gen);
  CAPTURE_PRECISE(mu);
  const CoordinateMaps::SpecialMobius special_mobius_map(mu);
  test_suite_for_map_on_sphere(special_mobius_map);
}

void test_map() {
  INFO("Map");
  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> radius_dis(0, 1);
  std::uniform_real_distribution<> phi_dis(0, 2.0 * M_PI);
  std::uniform_real_distribution<> theta_dis(0, M_PI);
  // Note: Empirically we have found that the map is accurate
  // to 12 decimal places for mu = 0.96.
  std::uniform_real_distribution<> mu_dis(-0.90, 0.90);
  const double theta = theta_dis(gen);
  CAPTURE_PRECISE(theta);
  const double phi = phi_dis(gen);
  CAPTURE_PRECISE(phi);
  const double radius = radius_dis(gen);
  CAPTURE_PRECISE(radius);
  const double x = radius * sin(theta) * cos(phi);
  CAPTURE_PRECISE(x);
  const double y = radius * sin(theta) * sin(phi);
  CAPTURE_PRECISE(y);
  const double z = radius * cos(theta);
  CAPTURE_PRECISE(z);
  const double mu = mu_dis(gen);
  CAPTURE_PRECISE(mu);
  const std::array<double, 3> input_point{{x, y, z}};
  const CoordinateMaps::SpecialMobius special_mobius_map(mu);
  const auto result_point = special_mobius_map(input_point);
  const auto& result_y = result_point[1];
  const auto& result_z = result_point[2];

  // Points inside the unit ball should remain inside the unit ball
  // under the SpecialMobius map:
  CHECK(magnitude(result_point) < 1.0);
  // As a map obtained by rotation about the x-axis, angles in
  // the y-z plane must be preserved:
  CHECK(atan2(y, z) == approx(atan2(result_y, result_z)));
  // The value `mu` should correspond to the x-coordinate of
  // the preimage of the origin under this map. That is, the
  // point with the x-coordinate `mu` (and y=z=0) should be
  // mapped to the origin:
  CHECK_ITERABLE_APPROX(
      (std::array<double, 3>{{0.0, 0.0, 0.0}}),
      (special_mobius_map(std::array<double, 3>{{mu, 0.0, 0.0}})));
  const std::array<double, 3> plus_one{{1.0, 0.0, 0.0}};
  const std::array<double, 3> minus_one{{-1.0, 0.0, 0.0}};
  // The points (1,0,0) and (-1,0,0) are fixed points:
  CHECK_ITERABLE_APPROX((plus_one), (special_mobius_map(plus_one)));
  CHECK_ITERABLE_APPROX((minus_one), (special_mobius_map(minus_one)));

  // Point at which the map is singular.
  // Since |mu|<1, this point also has x outside [-1,1].
  const std::array<double, 3> bad_point{{-1.0 / mu, 0.0, 0.0}};
  CHECK_FALSE(static_cast<bool>(special_mobius_map.inverse(bad_point)));
}

void test_large_mu() {
  INFO("Large mu");
  const double mu = 0.95;
  CAPTURE_PRECISE(mu);
  // A point on the unit sphere with x=0, y!=z:
  const std::array<double, 3> input_point{{0.0, 0.6, 0.8}};
  const CoordinateMaps::SpecialMobius special_mobius_map(mu);
  const auto result_point = special_mobius_map(input_point);
  const auto expected_input_point =
      special_mobius_map.inverse(result_point).get();
  const auto& result_y = result_point[1];
  const auto& result_z = result_point[2];
  // The SpecialMobius map should map the unit sphere to itself:
  CHECK(magnitude(result_point) == approx(1.0));
  // As a map obtained by rotation about the x-axis, angles in
  // the y-z plane must be preserved:
  CHECK(atan2(0.6, 0.8) == approx(atan2(result_y, result_z)));
  // The map is singular for mu = 1.0, but we can guarantee
  // that the accuracy does not suffer even for values of mu
  // that are close to 1.0:
  CHECK(abs(expected_input_point[0] - 0.0) < 1.e-13);
  CHECK(abs(expected_input_point[1] - 0.6) < 1.e-13);
  CHECK(abs(expected_input_point[2] - 0.8) < 1.e-13);
}

void test_is_identity() {
  INFO("Is identity");
  check_if_map_is_identity(CoordinateMaps::SpecialMobius{0.0});
  CHECK(not CoordinateMaps::SpecialMobius{0.1}.is_identity());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.SpecialMobius",
                  "[Domain][Unit]") {
  test_suite();
  test_map();
  test_large_mu();
  test_is_identity();
}

// [[OutputRegex, The magnitude of mu must be less than 0.96.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Domain.CoordinateMaps.SpecialMobius.Assert1", "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_special_mobius = CoordinateMaps::SpecialMobius(-2.3);
  static_cast<void>(failed_special_mobius);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The magnitude of mu must be less than 0.96.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Domain.CoordinateMaps.SpecialMobius.Assert2", "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_special_mobius = CoordinateMaps::SpecialMobius(0.98);
  static_cast<void>(failed_special_mobius);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
}  // namespace domain
