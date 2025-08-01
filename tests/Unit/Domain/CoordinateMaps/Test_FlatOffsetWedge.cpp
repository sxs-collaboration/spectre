// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <random>

#include "Domain/CoordinateMaps/FlatOffsetWedge.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"

namespace domain {

namespace {
void test_flat_offset_wedge() {
  INFO("FlatOffsetWedge");

  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> unit_dis(0.0, 1.0);

  // The value of outer_radius sets the scale.  Don't need to choose
  // it to be random because we will scale all other variables in terms
  // of outer_radius.
  const double outer_radius = 10.0;

  // The value of epsilon here must be the same as in the constructor
  // of FlatOffsetWedge, so that the sanity checks in the constructor
  // do not faile.  The sanity checks limit the values of the map
  // parameters so that the map doesn't get too close to singular.
  // With epsilon=0, the map might be singular for particularly
  // unlucky choices of the random numbers; with epsilon>0 the map is
  // always nonsingular (but might be close to singular depending on how
  // small epsilon is and what random numbers are chosen).
  const double epsilon = 0.1;

  // To pass sanity checks in the constructor, we must have
  // epsilon*outer_radius <= lower_face_x_width <= (1-epsilon)*outer_radius
  const double lower_face_x_width =
      outer_radius * (epsilon + (1.0 - 2.0 * epsilon) * unit_dis(gen));
  CAPTURE(lower_face_x_width);

  // To pass sanity checks in the constructor, we must have
  // L > epsilon*R and 2L^2 < (1-epsilon)^2*(R^2-D^2)
  // where L is lower_face_y_half_width and D is lower_face_x_width
  const double lower_face_y_half_width_min = epsilon * outer_radius;
  const double lower_face_y_half_width_max =
      (1.0 - epsilon) *
      sqrt((square(outer_radius) - square(lower_face_x_width)) / 2.0);
  const double lower_face_y_half_width =
      lower_face_y_half_width_min +
      unit_dis(gen) *
          (lower_face_y_half_width_max - lower_face_y_half_width_min);
  CAPTURE(lower_face_y_half_width);

  const CoordinateMaps::FlatOffsetWedge map(lower_face_y_half_width,
                                            lower_face_x_width, outer_radius);
  test_suite_for_map_on_unit_cube(map);

  // The following are tests that the inverse function correctly
  // returns an invalid std::optional when called for a point that is
  // outside the range of the map.

  // point with x < 0
  CHECK_FALSE(map.inverse({{-1.0, 0.0, 0.0}}));

  // point with x > lower_face_x_width
  CHECK_FALSE(map.inverse({{1.1 * lower_face_x_width, 0.0, 0.0}}));

  // point with z==0 but x is ok
  CHECK_FALSE(map.inverse({{0.5 * lower_face_x_width, 0.0, 0.0}}));

  // point outside of opening angle in y
  CHECK_FALSE(
      map.inverse({{0.5 * lower_face_x_width, 1.2 * lower_face_y_half_width,
                    1.02 * lower_face_y_half_width}}));

  // point with z < lower_face_y_half_width
  CHECK_FALSE(map.inverse(
      {{0.5 * lower_face_x_width, 0.0, 0.5 * lower_face_y_half_width}}));

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetWedge{0.0, lower_face_x_width,
                                         outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring(
          "Cannot have zero lower_face_y_half_width"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetWedge{-lower_face_y_half_width,
                                         lower_face_x_width, outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring(
          "Cannot have negative lower_face_y_half_width"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetWedge{lower_face_y_half_width,
                                         -lower_face_x_width, outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring(
          "Cannot have negative lower_face_x_width"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetWedge{lower_face_y_half_width, 0.0,
                                         outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring(
          "Cannot have zero lower_face_x_width"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetWedge{lower_face_y_half_width,
                                         lower_face_x_width, -outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring("Cannot have negative outer_radius"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetWedge{lower_face_y_half_width,
                                         lower_face_x_width, 0.0});
      }(),
      Catch::Matchers::ContainsSubstring("Cannot have zero outer_radius"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetWedge{
            sqrt(square(outer_radius) - square(lower_face_x_width)),
            lower_face_x_width, outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring("Must have R^2-D^2 > 2 L^2"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetWedge{lower_face_y_half_width,
                                         0.99 * epsilon * outer_radius,
                                         outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring(
          "The map is not tested if lower_face_x_width"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetWedge{
            outer_radius * sqrt(1.0 - square(1.001 * (1.0 - epsilon))) /
                sqrt(2.01),
            1.001 * (1 - epsilon) * outer_radius, outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring(
          "The map is not tested if lower_face_x_width"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetWedge{0.99 * epsilon * outer_radius,
                                         lower_face_x_width, outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring(
          "The map is not tested if lower_face_y_half_width"));
  CHECK_THROWS_WITH(
      [&]() {
        (CoordinateMaps::FlatOffsetWedge{
            sqrt(square(outer_radius) - square(lower_face_x_width)) *
                (1 - epsilon) / sqrt(1.999),
            lower_face_x_width, outer_radius});
      }(),
      Catch::Matchers::ContainsSubstring(
          "The map is not tested if 2L^2 > (1-epsilon)"));
#endif
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.FlatOffsetWedge",
                  "[Domain][Unit]") {
  test_flat_offset_wedge();
  CHECK(not CoordinateMaps::FlatOffsetWedge{}.is_identity());
}
}  // namespace domain
