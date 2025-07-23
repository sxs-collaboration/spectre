// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <random>

#include "Domain/CoordinateMaps/FlatOffsetSphericalWedge.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"

namespace domain {

namespace {
void test_flat_offset_wedge() {
  INFO("FlatOffsetSphericalWedge");

  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> unit_dis(0.0, 1.0);

  // The value of outer_radius sets the scale.  Don't need to choose
  // it to be random because we will scale all other variables in terms
  // of outer_radius.
  const double outer_radius = 10.0;
  CAPTURE(outer_radius);

  // The value of epsilon here must be the same as in the constructor
  // of FlatOffsetSphericalWedge, so that the sanity checks in the constructor
  // do not fail.  The sanity checks limit the values of the map
  // parameters so that the map doesn't get too close to singular.
  // With epsilon=0, the map might be singular for particularly
  // unlucky choices of the random numbers; with epsilon>0 the map is
  // always nonsingular (but might be close to singular depending on how
  // small epsilon is and what random numbers are chosen).
  const double epsilon = 0.1;

  // To pass sanity checks in the constructor, we must have
  // epsilon*outer_radius <= inner_radius <= (1-epsilon)*outer_radius
  const double inner_radius =
      outer_radius * (epsilon + (1.0 - 2.0 * epsilon) * unit_dis(gen));
  CAPTURE(inner_radius);

  // To pass sanity checks in the constructor, we must have
  // D > epsilon*R1 and D < (1-epsilon)^2*(R2^2-R1^2)
  // where D is lower_face_x_width, R1 is inner_radius, and R2 is outer_radius.
  // Also, D < (1-epsilon)*R1.
  // Note that for epsilon < 1, all these conditions are consistent.
  const double lower_face_x_width_min = epsilon * inner_radius;
  const double lower_face_x_width_max = std::min(
      (1.0 - epsilon) * sqrt(square(outer_radius) - square(inner_radius)),
      (1.0 - epsilon) * inner_radius);
  const double lower_face_x_width =
      lower_face_x_width_min +
      unit_dis(gen) * (lower_face_x_width_max - lower_face_x_width_min);
  CAPTURE(lower_face_x_width);

  const CoordinateMaps::FlatOffsetSphericalWedge map(
      lower_face_x_width, inner_radius, outer_radius);
  test_suite_for_map_on_unit_cube(map);

  // The following are tests that the inverse function correctly
  // returns an invalid std::optional when called for a point that is
  // outside the range of the map.

  // point with x < 0
  CHECK_FALSE(map.inverse({{-1.0, 0.0, 0.0}}));

  // point with x > lower_face_x_width
  CHECK_FALSE(map.inverse({{1.1 * lower_face_x_width, 0.0, 0.0}}));

  // point with z==0 but x is in range
  CHECK_FALSE(map.inverse({{0.5 * lower_face_x_width, 0.0, 0.0}}));

  // point outside of opening angle in y
  // At the x-midpoint of the Block,
  // the inner radius is sqrt(R1^2-D^2/4), the outer radius is sqrt(R2^2-D^2/4).
  // So here we place the point at 45+eps degrees midway between the inner
  // and outer radius.
  CHECK_FALSE(map.inverse(
      {{0.5 * lower_face_x_width,
        sqrt(2.1) * 0.5 *
            (sqrt(square(outer_radius) - square(lower_face_x_width)) +
             sqrt(square(inner_radius) - square(lower_face_x_width))),
        sqrt(2.0) * 0.5 *
            (sqrt(square(outer_radius) - square(lower_face_x_width)) +
             sqrt(square(inner_radius) - square(lower_face_x_width)))}}));

  // Too small z; last check in the inverse map.
  CHECK_FALSE(
      map.inverse({{0.5 * lower_face_x_width, 0.0, 0.5 * inner_radius}}));

  // Now test the sanity checks in the constructor.
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetSphericalWedge{0.0, inner_radius,
                                                  outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring(
          "Cannot have zero lower_face_x_width"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetSphericalWedge{-lower_face_x_width,
                                                  inner_radius, outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring(
          "Cannot have negative lower_face_x_width"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetSphericalWedge{lower_face_x_width, 0.0,
                                                  outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring("Cannot have zero inner_radius"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetSphericalWedge{lower_face_x_width,
                                                  -inner_radius, outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring("Cannot have negative inner_radius"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetSphericalWedge{lower_face_x_width,
                                                  inner_radius, 0.0});
      }(),
      Catch::Matchers::ContainsSubstring("Cannot have zero outer_radius"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetSphericalWedge{lower_face_x_width,
                                                  inner_radius, -outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring("Cannot have negative outer_radius"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetSphericalWedge{1.1 * inner_radius,
                                                  inner_radius, outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring(
          "Must have lower_face_x_width < inner_radius"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetSphericalWedge{
            lower_face_x_width, inner_radius,
            0.99 * sqrt(square(inner_radius) + square(lower_face_x_width))});
      }(),
      Catch::Matchers::ContainsSubstring("Must have (outer_radius)^2 >"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetSphericalWedge{0.98 * epsilon * outer_radius,
                                                  0.99 * epsilon * outer_radius,
                                                  outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring(
          "The map is not tested if inner_radius < epsilon*outer_radius"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetSphericalWedge{
            epsilon*outer_radius, (1.0 - 0.5 * epsilon) * outer_radius,
            outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring(
          "The map is not tested if inner_radius < epsilon*outer_radius"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetSphericalWedge{0.5 * epsilon * inner_radius,
                                                  inner_radius, outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring(
          "The map is not tested if lower_face_x_width < "
          "epsilon*inner_radius"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetSphericalWedge{
            (1.0 - 0.5 * epsilon) * 0.5 * outer_radius, 0.5 * outer_radius,
            outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring(
          "The map is not tested if lower_face_x_width < "
          "epsilon*inner_radius"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetSphericalWedge{
            (1.0 - 0.5 * epsilon) *
                sqrt(square(outer_radius) - square(0.9 * outer_radius)),
            0.9 * outer_radius, outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring(
          "The map is not tested if D^2 < (1-epsilon)^2"));
#endif
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.FlatOffsetSphericalWedge",
                  "[Domain][Unit]") {
  test_flat_offset_wedge();
  CHECK(not CoordinateMaps::FlatOffsetSphericalWedge{}.is_identity());
}
}  // namespace domain
