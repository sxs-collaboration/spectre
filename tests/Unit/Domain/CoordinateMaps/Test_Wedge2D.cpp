// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/optional.hpp>
#include <cmath>
#include <random>

#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Domain/Direction.hpp"
#include "Domain/OrientationMap.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace domain {
namespace {
void test_wedge2d_all_orientations(const bool with_equiangular_map) {
  INFO("Wedge2d all orientations");
  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1, 1);
  std::uniform_real_distribution<> unit_dis(0, 1);
  std::uniform_real_distribution<> inner_dis(1, 3);
  std::uniform_real_distribution<> outer_dis(4, 7);

  // Check that points on the corners of the reference square map to the correct
  // corners of the wedge.
  const std::array<double, 2> lower_right_corner{{1.0, -1.0}};
  const std::array<double, 2> upper_right_corner{{1.0, 1.0}};
  CAPTURE_PRECISE(gsl::at(lower_right_corner, 0));
  CAPTURE_PRECISE(gsl::at(upper_right_corner, 0));
  CAPTURE_PRECISE(gsl::at(lower_right_corner, 1));
  CAPTURE_PRECISE(gsl::at(upper_right_corner, 1));

  const double random_inner_radius_upper_xi = inner_dis(gen);
  CAPTURE_PRECISE(random_inner_radius_upper_xi);
  const double random_inner_radius_upper_eta = inner_dis(gen);
  CAPTURE_PRECISE(random_inner_radius_upper_eta);
  const double random_inner_radius_lower_xi = inner_dis(gen);
  CAPTURE_PRECISE(random_inner_radius_lower_xi);
  const double random_inner_radius_lower_eta = inner_dis(gen);
  CAPTURE_PRECISE(random_inner_radius_lower_eta);
  const double random_outer_radius_upper_xi = outer_dis(gen);
  CAPTURE_PRECISE(random_outer_radius_upper_xi);
  const double random_outer_radius_upper_eta = outer_dis(gen);
  CAPTURE_PRECISE(random_outer_radius_upper_eta);
  const double random_outer_radius_lower_xi = outer_dis(gen);
  CAPTURE_PRECISE(random_outer_radius_lower_xi);
  const double random_outer_radius_lower_eta = outer_dis(gen);
  CAPTURE_PRECISE(random_outer_radius_lower_eta);

  const CoordinateMaps::Wedge2D map_upper_xi(
      random_inner_radius_upper_xi, random_outer_radius_upper_xi, 0.0, 1.0,
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
      with_equiangular_map);
  const CoordinateMaps::Wedge2D map_upper_eta(
      random_inner_radius_upper_eta, random_outer_radius_upper_eta, 0.0, 1.0,
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
      with_equiangular_map);
  const CoordinateMaps::Wedge2D map_lower_xi(
      random_inner_radius_lower_xi, random_outer_radius_lower_xi, 0.0, 1.0,
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
      with_equiangular_map);
  const CoordinateMaps::Wedge2D map_lower_eta(
      random_inner_radius_lower_eta, random_outer_radius_lower_eta, 0.0, 1.0,
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
      with_equiangular_map);
  CHECK(map_lower_eta != map_lower_xi);
  CHECK(map_upper_eta != map_lower_eta);
  CHECK(map_lower_eta != map_upper_xi);

  CHECK(map_upper_xi(lower_right_corner)[0] ==
        approx(random_outer_radius_upper_xi / sqrt(2.0)));
  CHECK(map_upper_eta(lower_right_corner)[1] ==
        approx(random_outer_radius_upper_eta / sqrt(2.0)));
  CHECK(map_lower_xi(upper_right_corner)[0] ==
        approx(-random_outer_radius_lower_xi / sqrt(2.0)));
  CHECK(map_lower_eta(upper_right_corner)[1] ==
        approx(-random_outer_radius_lower_eta / sqrt(2.0)));

  // Check that random points on the edges of the reference square map to the
  // correct edges of the wedge.
  const std::array<double, 2> random_right_edge{{1.0, real_dis(gen)}};
  const std::array<double, 2> random_left_edge{{-1.0, real_dis(gen)}};

  CHECK(magnitude(map_upper_xi(random_right_edge)) ==
        approx(random_outer_radius_upper_xi));
  CHECK(map_upper_xi(random_left_edge)[0] ==
        approx(random_inner_radius_upper_xi / sqrt(2.0)));
  CHECK(magnitude(map_upper_eta(random_right_edge)) ==
        approx(random_outer_radius_upper_eta));
  CHECK(map_upper_eta(random_left_edge)[1] ==
        approx(random_inner_radius_upper_eta / sqrt(2.0)));
  CHECK(magnitude(map_lower_xi(random_right_edge)) ==
        approx(random_outer_radius_lower_xi));
  CHECK(map_lower_xi(random_left_edge)[0] ==
        approx(-random_inner_radius_lower_xi / sqrt(2.0)));
  CHECK(magnitude(map_lower_eta(random_right_edge)) ==
        approx(random_outer_radius_lower_eta));
  CHECK(map_lower_eta(random_left_edge)[1] ==
        approx(-random_inner_radius_lower_eta / sqrt(2.0)));

  const double inner_radius = inner_dis(gen);
  CAPTURE_PRECISE(inner_radius);
  const double outer_radius = outer_dis(gen);
  CAPTURE_PRECISE(outer_radius);
  const double inner_circularity = unit_dis(gen);
  CAPTURE_PRECISE(inner_circularity);
  const double outer_circularity = unit_dis(gen);
  CAPTURE_PRECISE(outer_circularity);

  for (OrientationMapIterator<2> map_i{}; map_i; ++map_i) {
    test_suite_for_map_on_unit_cube(CoordinateMaps::Wedge2D{
        inner_radius, outer_radius, inner_circularity, outer_circularity,
        map_i(), with_equiangular_map});
  }
}

void test_wedge2d_fail() noexcept {
  INFO("Wedge2d fail");
  const auto map =
      CoordinateMaps::Wedge2D(0.2, 4.0, 0.0, 1.0, OrientationMap<2>{}, true);

  // Any point with x<=0 should fail the inverse map.
  const std::array<double, 2> test_mapped_point1{{0.0, 3.0}};
  const std::array<double, 2> test_mapped_point2{{0.0, -6.0}};
  const std::array<double, 2> test_mapped_point3{{-1.0, 3.0}};

  // This point is outside the mapped wedge.  So inverse should either
  // return the correct inverse (which happens to be computable for
  // this point) or it should return boost::none.
  const std::array<double, 2> test_mapped_point4{{100.0, -6.0}};

  CHECK_FALSE(static_cast<bool>(map.inverse(test_mapped_point1)));
  CHECK_FALSE(static_cast<bool>(map.inverse(test_mapped_point2)));
  CHECK_FALSE(static_cast<bool>(map.inverse(test_mapped_point3)));
  if (map.inverse(test_mapped_point4)) {
    CHECK_ITERABLE_APPROX(map(map.inverse(test_mapped_point4).get()),
                          test_mapped_point4);
  }
}

void test_equality() {
  INFO("Equality");
  const auto wedge2d =
      CoordinateMaps::Wedge2D(0.2, 4.0, 0.0, 1.0, OrientationMap<2>{}, true);
  const auto wedge2d_inner_radius_changed =
      CoordinateMaps::Wedge2D(0.3, 4.0, 0.0, 1.0, OrientationMap<2>{}, true);
  const auto wedge2d_outer_radius_changed =
      CoordinateMaps::Wedge2D(0.2, 4.2, 0.0, 1.0, OrientationMap<2>{}, true);
  const auto wedge2d_inner_circularity_changed =
      CoordinateMaps::Wedge2D(0.2, 4.0, 0.3, 1.0, OrientationMap<2>{}, true);
  const auto wedge2d_outer_circularity_changed =
      CoordinateMaps::Wedge2D(0.2, 4.0, 0.0, 0.9, OrientationMap<2>{}, true);
  const auto wedge2d_orientation_map_changed = CoordinateMaps::Wedge2D(
      0.2, 4.0, 0.0, 1.0,
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::upper_eta(), Direction<2>::upper_xi()}}},
      true);
  const auto wedge2d_use_equiangular_map_changed =
      CoordinateMaps::Wedge2D(0.2, 4.0, 0.0, 1.0, OrientationMap<2>{}, false);
  CHECK_FALSE(wedge2d == wedge2d_inner_radius_changed);
  CHECK_FALSE(wedge2d == wedge2d_outer_radius_changed);
  CHECK_FALSE(wedge2d == wedge2d_inner_circularity_changed);
  CHECK_FALSE(wedge2d == wedge2d_outer_circularity_changed);
  CHECK_FALSE(wedge2d == wedge2d_orientation_map_changed);
  CHECK_FALSE(wedge2d == wedge2d_use_equiangular_map_changed);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge2D.Map", "[Domain][Unit]") {
  test_wedge2d_fail();
  test_wedge2d_all_orientations(false);  // Equidistant
  test_wedge2d_all_orientations(true);   // Equiangular
  test_equality();
  CHECK(not CoordinateMaps::Wedge2D{}.is_identity());
}

// [[OutputRegex, The radius of the inner surface must be greater than zero.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge2D.RadiusInner",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_wedge2d =
      CoordinateMaps::Wedge2D(-0.2, 4.0, 0.0, 1.0, OrientationMap<2>{}, true);
  static_cast<void>(failed_wedge2d);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Circularity of the inner surface must be between 0 and 1]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Domain.CoordinateMaps.Wedge2D.CircularityInner", "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_wedge2d =
      CoordinateMaps::Wedge2D(0.2, 4.0, -0.2, 1.0, OrientationMap<2>{}, true);
  static_cast<void>(failed_wedge2d);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Circularity of the outer surface must be between 0 and 1]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Domain.CoordinateMaps.Wedge2D.CircularityOuter", "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_wedge2d =
      CoordinateMaps::Wedge2D(0.2, 4.0, 0.0, -0.2, OrientationMap<2>{}, true);
  static_cast<void>(failed_wedge2d);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The radius of the outer surface must be greater than the
// radius of the inner surface.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge2D.RadiusOuter",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_wedge2d =
      CoordinateMaps::Wedge2D(4.2, 4.0, 0.0, 1.0, OrientationMap<2>{}, true);
  static_cast<void>(failed_wedge2d);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, The arguments passed into the constructor for Wedge2D result
// in an object where the outer surface is pierced by the inner surface.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Domain.CoordinateMaps.Wedge2D.PiercedSurface", "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_wedge2d =
      CoordinateMaps::Wedge2D(3.0, 4.0, 1.0, 0.0, OrientationMap<2>{}, true);
  static_cast<void>(failed_wedge2d);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
}  // namespace domain
