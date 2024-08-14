// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <random>

#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

namespace domain {
namespace {
using Wedge2D = CoordinateMaps::Wedge<2>;

void test_wedge2d_all_orientations(const bool with_equiangular_map) {
  INFO("Wedge2d all orientations");
  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1, 1);
  std::uniform_real_distribution<> unit_dis(0, 1);
  std::uniform_real_distribution<> inner_dis(1, 3);
  std::uniform_real_distribution<> outer_dis(4, 7);
  std::uniform_real_distribution<> cube_half_length_dist(8, 10);
  std::uniform_real_distribution<> offset_coord_dist(-1, 1);

  // Check that points on the corners of the reference square map to the correct
  // corners of the wedge.
  const std::array<double, 2> lower_right_corner{{1.0, -1.0}};
  const std::array<double, 2> upper_right_corner{{1.0, 1.0}};
  CAPTURE(gsl::at(lower_right_corner, 0));
  CAPTURE(gsl::at(upper_right_corner, 0));
  CAPTURE(gsl::at(lower_right_corner, 1));
  CAPTURE(gsl::at(upper_right_corner, 1));

  const double random_inner_radius_upper_xi = inner_dis(gen);
  CAPTURE(random_inner_radius_upper_xi);
  const double random_inner_radius_upper_eta = inner_dis(gen);
  CAPTURE(random_inner_radius_upper_eta);
  const double random_inner_radius_lower_xi = inner_dis(gen);
  CAPTURE(random_inner_radius_lower_xi);
  const double random_inner_radius_lower_eta = inner_dis(gen);
  CAPTURE(random_inner_radius_lower_eta);
  const double random_outer_radius_upper_xi = outer_dis(gen);
  CAPTURE(random_outer_radius_upper_xi);
  const double random_outer_radius_upper_eta = outer_dis(gen);
  CAPTURE(random_outer_radius_upper_eta);
  const double random_outer_radius_lower_xi = outer_dis(gen);
  CAPTURE(random_outer_radius_lower_xi);
  const double random_outer_radius_lower_eta = outer_dis(gen);
  CAPTURE(random_outer_radius_lower_eta);

  const Wedge2D map_upper_xi(
      random_inner_radius_upper_xi, random_outer_radius_upper_xi, 0.0, 1.0,
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
      with_equiangular_map);
  const Wedge2D map_upper_eta(
      random_inner_radius_upper_eta, random_outer_radius_upper_eta, 0.0, 1.0,
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
      with_equiangular_map);
  const Wedge2D map_lower_xi(
      random_inner_radius_lower_xi, random_outer_radius_lower_xi, 0.0, 1.0,
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
      with_equiangular_map);
  const Wedge2D map_lower_eta(
      random_inner_radius_lower_eta, random_outer_radius_lower_eta, 0.0, 1.0,
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
      with_equiangular_map);
  CHECK(map_lower_eta != map_lower_xi);
  CHECK(map_upper_eta != map_lower_eta);
  CHECK(map_lower_eta != map_upper_xi);

  CHECK(map_upper_xi(lower_right_corner)[0] ==
        approx(random_outer_radius_upper_xi / sqrt(2.0)));
  CHECK(map_upper_eta(lower_right_corner)[1] ==
        approx(-random_outer_radius_upper_eta / sqrt(2.0)));
  CHECK(map_lower_xi(upper_right_corner)[0] ==
        approx(-random_outer_radius_lower_xi / sqrt(2.0)));
  CHECK(map_lower_eta(upper_right_corner)[1] ==
        approx(random_outer_radius_lower_eta / sqrt(2.0)));

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
        approx(-random_inner_radius_upper_eta / sqrt(2.0)));
  CHECK(magnitude(map_lower_xi(random_right_edge)) ==
        approx(random_outer_radius_lower_xi));
  CHECK(map_lower_xi(random_left_edge)[0] ==
        approx(-random_inner_radius_lower_xi / sqrt(2.0)));
  CHECK(magnitude(map_lower_eta(random_right_edge)) ==
        approx(random_outer_radius_lower_eta));
  CHECK(map_lower_eta(random_left_edge)[1] ==
        approx(random_inner_radius_lower_eta / sqrt(2.0)));

  const double inner_radius = inner_dis(gen);
  CAPTURE(inner_radius);
  const double cube_half_length = cube_half_length_dist(gen);
  CAPTURE(cube_half_length);

  using WedgeHalves = Wedge2D::WedgeHalves;
  const std::array<WedgeHalves, 3> possible_halves = {
      {WedgeHalves::UpperOnly, WedgeHalves::LowerOnly, WedgeHalves::Both}};

  const std::array<double, 2> zero_offset{{0.0, 0.0}};
  const std::array<std::array<double, 2>, 2> focal_offsets = {
      {zero_offset, {{offset_coord_dist(gen), offset_coord_dist(gen)}}}};
  for (OrientationMapIterator<2> map_i{}; map_i; ++map_i) {
    if (get(determinant(discrete_rotation_jacobian(*map_i))) < 0.0) {
      continue;
    }
    const auto& orientation = map_i();
    CAPTURE(orientation);
    for (const auto& halves : possible_halves) {
      CAPTURE(halves);
      for (const auto radial_distribution :
           {CoordinateMaps::Distribution::Linear,
            CoordinateMaps::Distribution::Logarithmic,
            CoordinateMaps::Distribution::Inverse}) {
        CAPTURE(radial_distribution);
        for (const auto& focal_offset : focal_offsets) {
          CAPTURE(focal_offset);
          if (focal_offset == zero_offset) {
            // test centered Wedge
            {
              const double outer_radius = outer_dis(gen);
              CAPTURE(outer_radius);
              // circularity != 1.0 is only supported for Wedges where the
              // radial distribution is linear and there is no focal offset
              const bool use_random_circularity =
                  (radial_distribution == CoordinateMaps::Distribution::Linear);
              const double inner_circularity =
                  use_random_circularity ? unit_dis(gen) : 1.0;
              CAPTURE(inner_circularity);
              const double outer_circularity =
                  use_random_circularity ? unit_dis(gen) : 1.0;
              CAPTURE(outer_circularity);

              test_suite_for_map_on_unit_cube(
                  Wedge2D{inner_radius, outer_radius, inner_circularity,
                          outer_circularity, orientation, with_equiangular_map,
                          halves, radial_distribution});
            }
            {
              // test spherical offset Wedge that is centered
              const double outer_radius = outer_dis(gen);
              CAPTURE(outer_radius);

              const Wedge2D offset_wedge_with_no_offset(
                  inner_radius, outer_radius, cube_half_length, focal_offset,
                  orientation, with_equiangular_map, halves,
                  radial_distribution);
              test_suite_for_map_on_unit_cube(offset_wedge_with_no_offset);

              // make sure offset wedge with no offset reduces to centered wedge
              const Wedge2D expected_centered_wedge(
                  inner_radius, outer_radius, 1.0, 1.0, orientation,
                  with_equiangular_map, halves, radial_distribution);
              check_if_maps_are_equal(
                  domain::make_coordinate_map<Frame::Inertial, Frame::Grid>(
                      offset_wedge_with_no_offset),
                  domain::make_coordinate_map<Frame::Inertial, Frame::Grid>(
                      expected_centered_wedge));
            }
            {
              if (radial_distribution == CoordinateMaps::Distribution::Linear) {
                // test cubical offset Wedge that is centered
                const std::optional<double> outer_radius = std::nullopt;
                CAPTURE(outer_radius);

                const Wedge2D offset_wedge_with_no_offset(
                    inner_radius, outer_radius, cube_half_length, focal_offset,
                    orientation, with_equiangular_map, halves,
                    radial_distribution);
                test_suite_for_map_on_unit_cube(offset_wedge_with_no_offset);

                // make sure offset wedge with no offset reduces to centered
                // wedge
                const Wedge2D expected_centered_wedge(
                    inner_radius, cube_half_length * sqrt(2.0), 1.0, 0.0,
                    orientation, with_equiangular_map, halves,
                    radial_distribution);
                check_if_maps_are_equal(
                    domain::make_coordinate_map<Frame::Inertial, Frame::Grid>(
                        offset_wedge_with_no_offset),
                    domain::make_coordinate_map<Frame::Inertial, Frame::Grid>(
                        expected_centered_wedge));
              }
            }
          } else {
            // test offset Wedge
            if (radial_distribution == CoordinateMaps::Distribution::Linear) {
              {
                // test spherical offset Wedge with non-zero offset
                const double outer_radius = outer_dis(gen);
                CAPTURE(outer_radius);
                test_suite_for_map_on_unit_cube(
                    Wedge2D{inner_radius, outer_radius, cube_half_length,
                            focal_offset, orientation, with_equiangular_map,
                            halves, radial_distribution});
              }
              {
                // test cubical offset Wedge with non-zero offset
                const std::optional<double> outer_radius = std::nullopt;
                CAPTURE(outer_radius);
                test_suite_for_map_on_unit_cube(
                    Wedge2D{inner_radius, outer_radius, cube_half_length,
                            focal_offset, orientation, with_equiangular_map,
                            halves, radial_distribution});
              }
            } else {
              {
                // test spherical offset Wedge with non-zero offset
                const double outer_radius = outer_dis(gen);
                CAPTURE(outer_radius);
                test_suite_for_map_on_unit_cube(
                    Wedge2D{inner_radius, outer_radius, cube_half_length,
                            focal_offset, orientation, with_equiangular_map,
                            halves, radial_distribution});
              }
            }
          }
        }
      }
    }
  }
}

void test_wedge2d_fail() {
  INFO("Wedge2d fail");
  const auto centered_map =
      Wedge2D(0.2, 4.0, 1.0, 1.0, OrientationMap<2>::create_aligned(), true);
  const auto offset_map = Wedge2D(0.2, 2.0, 4.0, {{0.1, 0.}},
                                  OrientationMap<2>::create_aligned(), true);

  // Any point with x <= 0 should fail the inverse map with no focal offset
  const std::array<double, 2> test_mapped_point1{{0.0, 3.0}};
  const std::array<double, 2> test_mapped_point2{{0.0, -6.0}};
  const std::array<double, 2> test_mapped_point3{{-1.0, 3.0}};

  // Any point with x <= 0.1 should fail the inverse map with the focal offset
  const std::array<double, 2> test_mapped_point4{{0.0, 3.0}};
  const std::array<double, 2> test_mapped_point5{{0.0, -6.0}};
  const std::array<double, 2> test_mapped_point6{{-1.0, 3.0}};

  // This point is outside the mapped Wedges, so inverse should either return
  // the correct inverse (which happens to be computable for this point for both
  // Wedges) or it should return nullopt.
  const std::array<double, 2> test_mapped_point7{{100.0, -6.0}};

  // Check expected behavior for Wedge without offset
  CHECK_FALSE(centered_map.inverse(test_mapped_point1).has_value());
  CHECK_FALSE(centered_map.inverse(test_mapped_point2).has_value());
  CHECK_FALSE(centered_map.inverse(test_mapped_point3).has_value());
  if (centered_map.inverse(test_mapped_point7).has_value()) {
    CHECK_ITERABLE_APPROX(
        centered_map(centered_map.inverse(test_mapped_point7).value()),
        test_mapped_point7);
  }

  // Check expected behavior for Wedge with offset
  CHECK_FALSE(offset_map.inverse(test_mapped_point4).has_value());
  CHECK_FALSE(offset_map.inverse(test_mapped_point5).has_value());
  CHECK_FALSE(offset_map.inverse(test_mapped_point6).has_value());
  if (offset_map.inverse(test_mapped_point7).has_value()) {
    CHECK_ITERABLE_APPROX(
        offset_map(offset_map.inverse(test_mapped_point7).value()),
        test_mapped_point7);
  }
}

void test_equality() {
  INFO("Equality");

  const Wedge2D::WedgeHalves halves_to_use = Wedge2D::WedgeHalves::Both;
  const Wedge2D::WedgeHalves changed_halves_to_use =
      Wedge2D::WedgeHalves::UpperOnly;

  const domain::CoordinateMaps::Distribution radial_distribution =
      CoordinateMaps::Distribution::Linear;
  const domain::CoordinateMaps::Distribution changed_radial_distribution =
      CoordinateMaps::Distribution::Logarithmic;

  const std::array<double, 1>& opening_angles{{M_PI_2}};
  const std::array<double, 1>& changed_opening_angles{{M_PI_2 / 2.0}};

  // centered wedges
  const auto wedge2d =
      Wedge2D(0.2, 4.0, 1.0, 1.0, OrientationMap<2>::create_aligned(), true,
              halves_to_use, radial_distribution, opening_angles);
  const auto wedge2d_inner_radius_changed =
      Wedge2D(0.3, 4.0, 1.0, 1.0, OrientationMap<2>::create_aligned(), true,
              halves_to_use, radial_distribution, opening_angles);
  const auto wedge2d_outer_radius_changed =
      Wedge2D(0.2, 4.2, 1.0, 1.0, OrientationMap<2>::create_aligned(), true,
              halves_to_use, radial_distribution, opening_angles);
  const auto wedge2d_inner_circularity_changed =
      Wedge2D(0.2, 4.0, 0.3, 1.0, OrientationMap<2>::create_aligned(), true,
              halves_to_use, radial_distribution, opening_angles);
  const auto wedge2d_outer_circularity_changed =
      Wedge2D(0.2, 4.0, 1.0, 0.9, OrientationMap<2>::create_aligned(), true,
              halves_to_use, radial_distribution, opening_angles);
  const auto wedge2d_orientation_map_changed =
      Wedge2D(0.2, 4.0, 1.0, 1.0,
              OrientationMap<2>{std::array<Direction<2>, 2>{
                  {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
              true, halves_to_use, radial_distribution, opening_angles);
  const auto wedge2d_use_equiangular_map_changed =
      Wedge2D(0.2, 4.0, 1.0, 1.0, OrientationMap<2>::create_aligned(), false,
              halves_to_use, radial_distribution, opening_angles);
  const auto wedge2d_halves_to_use_changed =
      Wedge2D(0.2, 4.0, 1.0, 1.0, OrientationMap<2>::create_aligned(), true,
              changed_halves_to_use, radial_distribution, opening_angles);
  const auto wedge2d_radial_distribution_changed =
      Wedge2D(0.2, 4.0, 1.0, 1.0, OrientationMap<2>::create_aligned(), true,
              halves_to_use, changed_radial_distribution, opening_angles);
  const auto wedge2d_opening_angles_changed =
      Wedge2D(0.2, 4.0, 1.0, 1.0, OrientationMap<2>::create_aligned(), true,
              halves_to_use, radial_distribution, changed_opening_angles);

  CHECK_FALSE(wedge2d == wedge2d_inner_radius_changed);
  CHECK_FALSE(wedge2d == wedge2d_outer_radius_changed);
  CHECK_FALSE(wedge2d == wedge2d_inner_circularity_changed);
  CHECK_FALSE(wedge2d == wedge2d_outer_circularity_changed);
  CHECK_FALSE(wedge2d == wedge2d_orientation_map_changed);
  CHECK_FALSE(wedge2d == wedge2d_use_equiangular_map_changed);
  CHECK_FALSE(wedge2d == wedge2d_halves_to_use_changed);
  CHECK_FALSE(wedge2d == wedge2d_radial_distribution_changed);
  CHECK_FALSE(wedge2d == wedge2d_opening_angles_changed);

  // offset wedges
  const auto wedge2d_offset =
      Wedge2D(0.2, 4.0, 6.0, {{0.1, 0.0}}, OrientationMap<2>::create_aligned(),
              true, halves_to_use, radial_distribution);
  const auto wedge2d_offset_inner_radius_changed =
      Wedge2D(0.1, 4.0, 6.0, {{0.1, 0.0}}, OrientationMap<2>::create_aligned(),
              true, halves_to_use, radial_distribution);
  const auto wedge2d_offset_outer_radius_changed =
      Wedge2D(0.2, 3.0, 6.0, {{0.1, 0.0}}, OrientationMap<2>::create_aligned(),
              true, halves_to_use, radial_distribution);
  const auto wedge2d_offset_outer_circularity_changed = Wedge2D(
      0.2, std::nullopt, 6.0, {{0.1, 0.0}}, OrientationMap<2>::create_aligned(),
      true, halves_to_use, radial_distribution);
  const auto wedge2d_offset_cube_half_length_changed =
      Wedge2D(0.2, 4.0, 7.0, {{0.1, 0.0}}, OrientationMap<2>::create_aligned(),
              true, halves_to_use, radial_distribution);
  const auto wedge2d_offset_focal_offset_changed =
      Wedge2D(0.2, 4.0, 6.0, {{0.2, 0.0}}, OrientationMap<2>::create_aligned(),
              true, halves_to_use, radial_distribution);
  const auto wedge2d_offset_orientation_map_changed =
      Wedge2D(0.2, 4.0, 6.0, {{0.1, 0.0}},
              OrientationMap<2>{std::array<Direction<2>, 2>{
                  {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
              true, halves_to_use, radial_distribution);
  const auto wedge2d_offset_use_equiangular_map_changed =
      Wedge2D(0.2, 4.0, 6.0, {{0.1, 0.0}}, OrientationMap<2>::create_aligned(),
              false, halves_to_use, radial_distribution);
  const auto wedge2d_offset_halves_to_use_changed =
      Wedge2D(0.2, 4.0, 6.0, {{0.1, 0.0}}, OrientationMap<2>::create_aligned(),
              true, changed_halves_to_use, radial_distribution);
  const auto wedge2d_offset_radial_distribution_changed =
      Wedge2D(0.2, 4.0, 6.0, {{0.1, 0.0}}, OrientationMap<2>::create_aligned(),
              true, halves_to_use, changed_radial_distribution);

  CHECK_FALSE(wedge2d_offset == wedge2d_offset_inner_radius_changed);
  CHECK_FALSE(wedge2d_offset == wedge2d_offset_outer_radius_changed);
  CHECK_FALSE(wedge2d_offset == wedge2d_offset_outer_circularity_changed);
  CHECK_FALSE(wedge2d_offset == wedge2d_offset_cube_half_length_changed);
  CHECK_FALSE(wedge2d_offset == wedge2d_offset_focal_offset_changed);
  CHECK_FALSE(wedge2d_offset == wedge2d_offset_orientation_map_changed);
  CHECK_FALSE(wedge2d_offset == wedge2d_offset_use_equiangular_map_changed);
  CHECK_FALSE(wedge2d_offset == wedge2d_offset_halves_to_use_changed);
  CHECK_FALSE(wedge2d_offset == wedge2d_offset_radial_distribution_changed);

  // make sure spherical offset wedge with zero offset reduces to centered wedge
  const auto wedge2d_offset_centered_with_spherical_outer_circularity =
      Wedge2D(0.2, 4.0, 6.0, {{0.0, 0.0}}, OrientationMap<2>::create_aligned(),
              true, halves_to_use, radial_distribution);
  CHECK(wedge2d == wedge2d_offset_centered_with_spherical_outer_circularity);

  // make sure cubical offset wedge with zero offset reduces to centered wedge
  const auto wedge2d_centered_with_flat_outer_circularity =
      Wedge2D(0.2, sqrt(2.0), 1.0, 0.0, OrientationMap<2>::create_aligned(),
              true, halves_to_use, radial_distribution, opening_angles);
  const auto wedge2d_offset_centered_with_flat_outer_circularity = Wedge2D(
      0.2, std::nullopt, 1.0, {{0.0, 0.0}}, OrientationMap<2>::create_aligned(),
      true, halves_to_use, radial_distribution);
  CHECK(wedge2d_centered_with_flat_outer_circularity ==
        wedge2d_offset_centered_with_flat_outer_circularity);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge2D.Map", "[Domain][Unit]") {
  test_wedge2d_fail();
  test_wedge2d_all_orientations(false);  // Equidistant
  test_wedge2d_all_orientations(true);   // Equiangular
  test_equality();
  CHECK(not Wedge2D{}.is_identity());

#ifdef SPECTRE_DEBUG
  // centered wedge checks
  CHECK_THROWS_WITH(
      Wedge2D(-0.2, 4.0, 0.0, 1.0, OrientationMap<2>::create_aligned(), true),
      Catch::Matchers::ContainsSubstring(
          "The radius of the inner surface must be greater than zero."));
  CHECK_THROWS_WITH(
      Wedge2D(0.2, 4.0, -0.2, 1.0, OrientationMap<2>::create_aligned(), true),
      Catch::Matchers::ContainsSubstring(
          "Sphericity of the inner surface must be between 0 and 1"));
  CHECK_THROWS_WITH(
      Wedge2D(0.2, 4.0, 0.0, -0.2, OrientationMap<2>::create_aligned(), true),
      Catch::Matchers::ContainsSubstring(
          "Sphericity of the outer surface must be between 0 and 1"));
  CHECK_THROWS_WITH(
      Wedge2D(4.2, 4.0, 0.0, 1.0, OrientationMap<2>::create_aligned(), true),
      Catch::Matchers::ContainsSubstring(
          "The radius of the outer surface must be greater than "
          "the radius of the inner surface."));
  CHECK_THROWS_WITH(
      Wedge2D(3.0, 4.0, 1.0, 0.0, OrientationMap<2>::create_aligned(), true),
      Catch::Matchers::ContainsSubstring(
          "The arguments passed into the constructor for Wedge result in an "
          "object where the outer surface is pierced by the inner surface."));
  CHECK_THROWS_WITH(
      Wedge2D(0.2, 4.0, 0.0, 1.0, OrientationMap<2>::create_aligned(), true,
              Wedge2D::WedgeHalves::Both,
              domain::CoordinateMaps::Distribution::Logarithmic),
      Catch::Matchers::ContainsSubstring(
          "Only the 'Linear' radial distribution is supported for "
          "non-spherical wedges."));
  CHECK_THROWS_WITH(
      Wedge2D(0.2, 4.0, 0.2, 1.0, OrientationMap<2>::create_aligned(), false,
              Wedge2D::WedgeHalves::Both,
              domain::CoordinateMaps::Distribution::Linear,
              std::array<double, 1>{{M_PI_4}}),
      Catch::Matchers::ContainsSubstring(
          "If using opening angles other than pi/2, then the "
          "equiangular map option must be turned on."));

  // offset wedge checks
  CHECK_THROWS_WITH(
      Wedge2D(-0.2, 4.0, 6.0, {{0.1, 0.0}}, OrientationMap<2>::create_aligned(),
              true),
      Catch::Matchers::ContainsSubstring(
          "The radius of the inner surface must be greater than zero."));
  CHECK_THROWS_WITH(Wedge2D(4.2, 4.0, 6.0, {{0.1, 0.0}},
                            OrientationMap<2>::create_aligned(), true),
                    Catch::Matchers::ContainsSubstring(
                        "The radius of the outer surface must be greater than "
                        "the radius of the inner surface."));
  CHECK_THROWS_WITH(
      Wedge2D(0.2, std::nullopt, 6.0, {{0.1, 0.0}},
              OrientationMap<2>::create_aligned(), true,
              Wedge2D::WedgeHalves::Both,
              domain::CoordinateMaps::Distribution::Logarithmic),
      Catch::Matchers::ContainsSubstring(
          "Only the 'Linear' radial distribution is supported for "
          "non-spherical wedges."));
  CHECK_THROWS_WITH(
      Wedge2D(3.0, std::nullopt, 3.0, {{0.0, 0.0}},
              OrientationMap<2>::create_aligned(), true),
      Catch::Matchers::ContainsSubstring(
          "The arguments passed into the constructor for Wedge result in an "
          "object where the outer surface is pierced by the inner surface."));
  CHECK_THROWS_WITH(
      Wedge2D(0.2, 4.0, 1.0, {{4., 0.}}, OrientationMap<2>::create_aligned(),
              true),
      Catch::Matchers::ContainsSubstring(
          "For a spherical focally offset Wedge, the sum of the outer radius "
          "and the coordinate of the focal offset with the largest magnitude "
          "must be less than the cube half length. In other words, the "
          "spherical surface at the given outer radius centered at the focal "
          "offset must not pierce the cube of length 2 * cube_half_length_ "
          "centered at the origin. See the Wedge class documentation for a "
          "visual representation of this sphere and cube."));
  CHECK_THROWS_WITH(
      Wedge2D(0.2, std::nullopt, 1.0, {{4., 0.}},
              OrientationMap<2>::create_aligned(), true),
      Catch::Matchers::ContainsSubstring(
          "For a cubical focally offset Wedge, the sum of the inner radius "
          "and the coordinate of the focal offset with the largest magnitude "
          "must be less than the cube half length. In other words, the "
          "spherical surface at the given inner radius centered at the focal "
          "offset must not pierce the cube of length 2 * cube_half_length_ "
          "centered at the origin. See the Wedge class documentation for a "
          "visual representation of this sphere and cube."));
#endif
}
}  // namespace domain
