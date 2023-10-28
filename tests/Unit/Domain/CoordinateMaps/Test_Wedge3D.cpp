// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <random>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Identity.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

namespace domain {
namespace {
using Wedge3D = CoordinateMaps::Wedge<3>;

void test_wedge3d_all_directions() {
  INFO("Wedge3d all directions");
  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> unit_dis(0, 1);
  std::uniform_real_distribution<> inner_dis(1, 3);
  std::uniform_real_distribution<> outer_dis(5.2, 7);
  std::uniform_real_distribution<> angle_dis(80.0, 90.0);
  const double inner_radius = inner_dis(gen);
  CAPTURE(inner_radius);
  const double outer_radius = outer_dis(gen);
  CAPTURE(outer_radius);
  const double inner_sphericity = unit_dis(gen);
  CAPTURE(inner_sphericity);
  const double outer_sphericity = unit_dis(gen);
  CAPTURE(outer_sphericity);
  const double opening_angle_xi = angle_dis(gen) * M_PI / 180.0;
  CAPTURE(opening_angle_xi * 180.0 / M_PI);
  const double opening_angle_eta = angle_dis(gen) * M_PI / 180.0;
  CAPTURE(opening_angle_eta * 180.0 / M_PI);

  const bool with_adapted_equiangular_map = false;
  using WedgeHalves = Wedge3D::WedgeHalves;
  const std::array<WedgeHalves, 3> halves_array = {
      {WedgeHalves::UpperOnly, WedgeHalves::LowerOnly, WedgeHalves::Both}};
  // [cartesian_product_loop]
  for (const auto& [halves, orientation, with_equiangular_map,
                    radial_distribution] :
       random_sample<5>(
           cartesian_product(
               halves_array, all_wedge_directions(), make_array(true, false),
               make_array(CoordinateMaps::Distribution::Linear,
                          CoordinateMaps::Distribution::Logarithmic,
                          CoordinateMaps::Distribution::Inverse)),
           make_not_null(&gen))) {
    // [cartesian_product_loop]
    CAPTURE(halves);
    CAPTURE(orientation);
    CAPTURE(with_equiangular_map);
    CAPTURE(radial_distribution);
    const Wedge3D wedge_map(
        inner_radius, outer_radius,
        radial_distribution == CoordinateMaps::Distribution::Linear
            ? inner_sphericity
            : 1.0,
        radial_distribution == CoordinateMaps::Distribution::Linear
            ? outer_sphericity
            : 1.0,
        orientation, with_equiangular_map, halves, radial_distribution,
        with_equiangular_map
            ? std::array<double, 2>{{opening_angle_xi, opening_angle_eta}}
            : std::array<double, 2>{{M_PI_2, M_PI_2}},
        with_adapted_equiangular_map);
    test_suite_for_map_on_unit_cube(wedge_map);
  }
}

void test_wedge3d_alignment() {
  INFO("Wedge3d alignment");
  // This test tests that the logical axes point along the expected directions
  // in physical space

  const double inner_r = sqrt(3.0);
  const double outer_r = 2.0 * sqrt(3.0);

  using WedgeHalves = Wedge3D::WedgeHalves;
  const auto wedge_directions = all_wedge_directions();

  for (const auto& with_equiangular_map : {true, false}) {
    CAPTURE(with_equiangular_map);
    for (const auto radial_distribution :
         {CoordinateMaps::Distribution::Linear,
          CoordinateMaps::Distribution::Logarithmic,
          CoordinateMaps::Distribution::Inverse}) {
      CAPTURE(radial_distribution);
      const double inner_sphericity =
          radial_distribution == CoordinateMaps::Distribution::Linear ? 0.0
                                                                      : 1.0;
      const Wedge3D map_upper_zeta(inner_r, outer_r, inner_sphericity, 1.0,
                                   wedge_directions[0], with_equiangular_map,
                                   WedgeHalves::Both,
                                   radial_distribution);  // Upper Z wedge
      const Wedge3D map_upper_eta(inner_r, outer_r, inner_sphericity, 1.0,
                                  wedge_directions[2], with_equiangular_map,
                                  WedgeHalves::Both,
                                  radial_distribution);  // Upper Y wedge
      const Wedge3D map_upper_xi(inner_r, outer_r, inner_sphericity, 1.0,
                                 wedge_directions[4], with_equiangular_map,
                                 WedgeHalves::Both,
                                 radial_distribution);  // Upper X Wedge
      const Wedge3D map_lower_zeta(inner_r, outer_r, inner_sphericity, 1.0,
                                   wedge_directions[1], with_equiangular_map,
                                   WedgeHalves::Both,
                                   radial_distribution);  // Lower Z wedge
      const Wedge3D map_lower_eta(inner_r, outer_r, inner_sphericity, 1.0,
                                  wedge_directions[3], with_equiangular_map,
                                  WedgeHalves::Both,
                                  radial_distribution);  // Lower Y wedge
      const Wedge3D map_lower_xi(inner_r, outer_r, inner_sphericity, 1.0,
                                 wedge_directions[5], with_equiangular_map,
                                 WedgeHalves::Both,
                                 radial_distribution);  // Lower X wedge
      const std::array<double, 3> lowest_corner{{-1.0, -1.0, -1.0}};
      const std::array<double, 3> along_xi{{1.0, -1.0, -1.0}};
      const std::array<double, 3> along_eta{{-1.0, 1.0, -1.0}};
      const std::array<double, 3> along_zeta{{-1.0, -1.0, 1.0}};

      const std::array<double, 3> lowest_physical_corner_upper_zeta{
          {-1.0, -1.0, 1.0}};
      const std::array<double, 3> lowest_physical_corner_upper_eta{
          {-1.0, 1.0, -1.0}};
      const std::array<double, 3> lowest_physical_corner_upper_xi{
          {1.0, -1.0, -1.0}};
      const std::array<double, 3> lowest_physical_corner_lower_zeta{
          {-1.0, 1.0, -1.0}};
      const std::array<double, 3> lowest_physical_corner_lower_eta{
          {-1.0, -1.0, 1.0}};
      const std::array<double, 3> lowest_physical_corner_lower_xi{
          {-1.0, 1.0, -1.0}};

      // Test that this map's logical axes point along +X, +Y, +Z:
      CHECK(map_upper_zeta(along_xi)[0] == approx(1.0));
      CHECK(map_upper_zeta(along_eta)[1] == approx(1.0));
      CHECK(map_upper_zeta(along_zeta)[2] == approx(2.0));
      CHECK_ITERABLE_APPROX(map_upper_zeta(lowest_corner),
                            lowest_physical_corner_upper_zeta);

      // Test that this map's logical axes point along +Z, +X, +Y:
      CHECK(map_upper_eta(along_xi)[2] == approx(1.0));
      CHECK(map_upper_eta(along_eta)[0] == approx(1.0));
      CHECK(map_upper_eta(along_zeta)[1] == approx(2.0));
      CHECK_ITERABLE_APPROX(map_upper_eta(lowest_corner),
                            lowest_physical_corner_upper_eta);

      // Test that this map's logical axes point along +Y, +Z, +X:
      CHECK(map_upper_xi(along_xi)[1] == approx(1.0));
      CHECK(map_upper_xi(along_eta)[2] == approx(1.0));
      CHECK(map_upper_xi(along_zeta)[0] == approx(2.0));
      CHECK_ITERABLE_APPROX(map_upper_xi(lowest_corner),
                            lowest_physical_corner_upper_xi);

      // Test that this map's logical axes point along +X, -Y, -Z:
      CHECK(map_lower_zeta(along_xi)[0] == approx(1.0));
      CHECK(map_lower_zeta(along_eta)[1] == approx(-1.0));
      CHECK(map_lower_zeta(along_zeta)[2] == approx(-2.0));
      CHECK_ITERABLE_APPROX(map_lower_zeta(lowest_corner),
                            lowest_physical_corner_lower_zeta);

      // Test that this map's logical axes point along -Z, +X, -Y:
      CHECK(map_lower_eta(along_xi)[2] == approx(-1.0));
      CHECK(map_lower_eta(along_eta)[0] == approx(1.0));
      CHECK(map_lower_eta(along_zeta)[1] == approx(-2.0));
      CHECK_ITERABLE_APPROX(map_lower_eta(lowest_corner),
                            lowest_physical_corner_lower_eta);

      // Test that this map's logical axes point along -Y, +Z, -X:
      CHECK(map_lower_xi(along_xi)[1] == approx(-1.0));
      CHECK(map_lower_xi(along_eta)[2] == approx(1.0));
      CHECK(map_lower_xi(along_zeta)[0] == approx(-2.0));
      CHECK_ITERABLE_APPROX(map_lower_xi(lowest_corner),
                            lowest_physical_corner_lower_xi);
    }
  }
}

void test_wedge3d_random_radii() {
  INFO("Wedge3d random radii");
  // Set up random number generator
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1, 1);
  std::uniform_real_distribution<> inner_dis(1, 3);
  std::uniform_real_distribution<> outer_dis(4, 7);
  std::uniform_real_distribution<> angle_dis(55.0, 125.0);

  // Check that points on the corners of the reference cube map to the correct
  // corners of the wedge.
  const std::array<double, 3> inner_corner{{-1.0, -1.0, -1.0}};
  const std::array<double, 3> outer_corner{{1.0, 1.0, 1.0}};
  const double random_inner_radius_lower_xi = inner_dis(gen);
  CAPTURE(random_inner_radius_lower_xi);
  const double random_inner_radius_lower_eta = inner_dis(gen);
  CAPTURE(random_inner_radius_lower_eta);
  const double random_inner_radius_lower_zeta = inner_dis(gen);
  CAPTURE(random_inner_radius_lower_zeta);
  const double random_inner_radius_upper_xi = inner_dis(gen);
  CAPTURE(random_inner_radius_upper_xi);
  const double random_inner_radius_upper_eta = inner_dis(gen);
  CAPTURE(random_inner_radius_upper_eta);
  const double random_inner_radius_upper_zeta = inner_dis(gen);
  CAPTURE(random_inner_radius_upper_zeta);

  const double random_outer_radius_lower_xi = outer_dis(gen);
  CAPTURE(random_outer_radius_lower_xi);
  const double random_outer_radius_lower_eta = outer_dis(gen);
  CAPTURE(random_outer_radius_lower_eta);
  const double random_outer_radius_lower_zeta = outer_dis(gen);
  CAPTURE(random_outer_radius_lower_zeta);
  const double random_outer_radius_upper_xi = outer_dis(gen);
  CAPTURE(random_outer_radius_upper_xi);
  const double random_outer_radius_upper_eta = outer_dis(gen);
  CAPTURE(random_outer_radius_upper_eta);
  const double random_outer_radius_upper_zeta = outer_dis(gen);
  CAPTURE(random_outer_radius_upper_zeta);

  const double opening_angle_xi = angle_dis(gen) * M_PI / 180.0;
  CAPTURE(opening_angle_xi * 180.0 / M_PI);
  const double opening_angle_eta = angle_dis(gen) * M_PI / 180.0;
  CAPTURE(opening_angle_eta * 180.0 / M_PI);
  std::array<double, 2> opening_angles{{opening_angle_xi, opening_angle_eta}};
  std::array<double, 2> default_angles{{M_PI_2, M_PI_2}};

  using WedgeHalves = Wedge3D::WedgeHalves;
  const auto wedge_directions = all_wedge_directions();
  for (const auto& with_equiangular_map : {true, false}) {
    CAPTURE(with_equiangular_map);
    for (const auto radial_distribution :
         {CoordinateMaps::Distribution::Linear,
          CoordinateMaps::Distribution::Logarithmic,
          CoordinateMaps::Distribution::Inverse}) {
      CAPTURE(radial_distribution);
      const double inner_sphericity =
          radial_distribution == CoordinateMaps::Distribution::Linear ? 0.0
                                                                      : 1.0;
      const Wedge3D map_lower_xi(
          random_inner_radius_lower_xi, random_outer_radius_lower_xi,
          inner_sphericity, 1.0, wedge_directions[5], with_equiangular_map,
          WedgeHalves::Both, radial_distribution,
          with_equiangular_map ? opening_angles : default_angles);
      const Wedge3D map_lower_eta(
          random_inner_radius_lower_eta, random_outer_radius_lower_eta,
          inner_sphericity, 1.0, wedge_directions[3], with_equiangular_map,
          WedgeHalves::Both, radial_distribution,
          with_equiangular_map ? opening_angles : default_angles);
      const Wedge3D map_lower_zeta(
          random_inner_radius_lower_zeta, random_outer_radius_lower_zeta,
          inner_sphericity, 1.0, wedge_directions[1], with_equiangular_map,
          WedgeHalves::Both, radial_distribution,
          with_equiangular_map ? opening_angles : default_angles);
      const Wedge3D map_upper_xi(
          random_inner_radius_upper_xi, random_outer_radius_upper_xi,
          inner_sphericity, 1.0, wedge_directions[4], with_equiangular_map,
          WedgeHalves::Both, radial_distribution,
          with_equiangular_map ? opening_angles : default_angles);
      const Wedge3D map_upper_eta(
          random_inner_radius_upper_eta, random_outer_radius_upper_eta,
          inner_sphericity, 1.0, wedge_directions[2], with_equiangular_map,
          WedgeHalves::Both, radial_distribution,
          with_equiangular_map ? opening_angles : default_angles);
      const Wedge3D map_upper_zeta(
          random_inner_radius_upper_zeta, random_outer_radius_upper_zeta,
          inner_sphericity, 1.0, wedge_directions[0], with_equiangular_map,
          WedgeHalves::Both, radial_distribution,
          with_equiangular_map ? opening_angles : default_angles);
      const double cap_xi_one =
          tan(with_equiangular_map ? 0.5 * opening_angle_xi : M_PI_4);
      const double cap_eta_one =
          tan(with_equiangular_map ? 0.5 * opening_angle_eta : M_PI_4);
      const double one_over_denominator =
          1.0 / sqrt(1.0 + square(cap_xi_one) + square(cap_eta_one));

      if (radial_distribution != CoordinateMaps::Distribution::Linear) {
        CHECK(map_lower_xi(outer_corner)[0] ==
              approx(-random_outer_radius_lower_xi * one_over_denominator));
        CHECK(map_lower_eta(outer_corner)[1] ==
              approx(-random_outer_radius_lower_eta * one_over_denominator));
        CHECK(map_lower_zeta(outer_corner)[2] ==
              approx(-random_outer_radius_lower_zeta * one_over_denominator));
        CHECK(map_upper_xi(inner_corner)[0] ==
              approx(random_inner_radius_upper_xi * one_over_denominator));
        CHECK(map_upper_eta(inner_corner)[1] ==
              approx(random_inner_radius_upper_eta * one_over_denominator));
        CHECK(map_upper_zeta(inner_corner)[2] ==
              approx(random_inner_radius_upper_zeta * one_over_denominator));
      }
      // Check that random points on the edges of the reference cube map to the
      // correct edges of the wedge.
      const std::array<double, 3> random_outer_face{
          {real_dis(gen), real_dis(gen), 1.0}};
      const std::array<double, 3> random_inner_face{
          {real_dis(gen), real_dis(gen), -1.0}};
      CAPTURE(random_outer_face);
      CAPTURE(random_inner_face);

      if (radial_distribution == CoordinateMaps::Distribution::Linear) {
        CHECK(map_lower_xi(random_inner_face)[0] ==
              approx(-random_inner_radius_lower_xi / sqrt(3.0)));
        CHECK(map_lower_eta(random_inner_face)[1] ==
              approx(-random_inner_radius_lower_eta / sqrt(3.0)));
        CHECK(map_upper_xi(random_inner_face)[0] ==
              approx(random_inner_radius_upper_xi / sqrt(3.0)));
        CHECK(map_upper_eta(random_inner_face)[1] ==
              approx(random_inner_radius_upper_eta / sqrt(3.0)));
      }
      CHECK(magnitude(map_lower_xi(random_outer_face)) ==
            approx(random_outer_radius_lower_xi));
      CHECK(magnitude(map_lower_eta(random_outer_face)) ==
            approx(random_outer_radius_lower_eta));
      CHECK(magnitude(map_upper_xi(random_outer_face)) ==
            approx(random_outer_radius_upper_xi));
      CHECK(magnitude(map_upper_eta(random_outer_face)) ==
            approx(random_outer_radius_upper_eta));
      CHECK(magnitude(map_lower_zeta(random_outer_face)) ==
            approx(random_outer_radius_lower_zeta));
      CHECK(magnitude(map_upper_zeta(random_outer_face)) ==
            approx(random_outer_radius_upper_zeta));
    }
  }
}

void test_wedge3d_large_radius() {
  INFO("Wedge3d large radius");
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> real_dis(-1, 1);
  std::uniform_real_distribution<> angle_dis(55.0, 125.0);

  const double inner_radius = 1.5;
  const double outer_radius = 1.0e11;

  const double opening_angle_xi = angle_dis(gen) * M_PI / 180.0;
  CAPTURE(opening_angle_xi * 180.0 / M_PI);
  const double opening_angle_eta = angle_dis(gen) * M_PI / 180.0;
  CAPTURE(opening_angle_eta * 180.0 / M_PI);
  std::array<double, 2> opening_angles{{opening_angle_xi, opening_angle_eta}};
  std::array<double, 2> default_angles{{M_PI_2, M_PI_2}};

  // Check that random points on the edges of the reference cube map to the
  // correct edges of the wedge.
  const std::array<double, 3> random_inner_logical{
      {real_dis(gen), real_dis(gen), -1.0}};
  const std::array<double, 3> random_outer_logical{
      {real_dis(gen), real_dis(gen), 1.0}};

  using WedgeHalves = Wedge3D::WedgeHalves;
  for (const auto& with_equiangular_map : {true, false}) {
    CAPTURE(with_equiangular_map);
    for (const auto& which_wedges :
         {WedgeHalves::Both, WedgeHalves::UpperOnly, WedgeHalves::LowerOnly}) {
      const Wedge3D map(inner_radius, outer_radius, 1.0, 1.0,
                        OrientationMap<3>{}, with_equiangular_map, which_wedges,
                        CoordinateMaps::Distribution::Inverse,
                        with_equiangular_map ? opening_angles : default_angles);
      const double cap_xi_one =
          tan(with_equiangular_map ? 0.5 * opening_angle_xi : M_PI_4);
      const double cap_eta_one =
          tan(with_equiangular_map ? 0.5 * opening_angle_eta : M_PI_4);
      const double one_over_denominator =
          1.0 / sqrt(1.0 + square(cap_xi_one) + square(cap_eta_one));

      // 1/r^2 dr/dzeta
      const double radius_scale_factor =
          (outer_radius - inner_radius) / (2.0 * outer_radius * inner_radius);
      const auto check_point = [&](const std::array<double, 3>& logical_point,
                                   const double expected_radius) {
        CAPTURE(logical_point);
        CAPTURE(expected_radius);
        const auto mapped_point = map(logical_point);
        CAPTURE(mapped_point);

        CHECK(magnitude(mapped_point) == approx(expected_radius));
        CHECK_ITERABLE_APPROX(map.inverse(mapped_point).value(), logical_point);

        const auto jacobian = map.jacobian(logical_point);
        const auto inv_jacobian = map.inv_jacobian(logical_point);
        CAPTURE(jacobian);
        CAPTURE(inv_jacobian);

        {
          const std::array<double, 3> radial_vector =
              mapped_point / magnitude(mapped_point);
          std::array r_dot_jacobian{0.0, 0.0, 0.0};
          for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
              gsl::at(r_dot_jacobian, i) +=
                  gsl::at(radial_vector, j) * jacobian.get(j, i);
            }
          }

          CAPTURE(r_dot_jacobian);
          auto jacobian_approx = approx.scale(square(expected_radius));
          CHECK(gsl::at(r_dot_jacobian, 0) == jacobian_approx(0.0));
          CHECK(gsl::at(r_dot_jacobian, 1) == jacobian_approx(0.0));
          CHECK(gsl::at(r_dot_jacobian, 2) ==
                jacobian_approx(radius_scale_factor * square(expected_radius)));
        }

        const auto jacobian_times_inverse = tenex::evaluate<ti::I, ti::j>(
            jacobian(ti::I, ti::k) * inv_jacobian(ti::K, ti::j));
        const auto inverse_times_jacobian = tenex::evaluate<ti::I, ti::j>(
            inv_jacobian(ti::I, ti::k) * jacobian(ti::K, ti::j));
        CAPTURE(jacobian_times_inverse);
        CAPTURE(inverse_times_jacobian);
        CHECK_ITERABLE_APPROX(jacobian_times_inverse,
                              (identity<3, double>(0.0)));
        CHECK_ITERABLE_APPROX(inverse_times_jacobian,
                              (identity<3, double>(0.0)));
      };

      // Check that points on the corners of the reference cube map to
      // the correct corners of the wedge.
      for (const double x : {-1.0, 1.0}) {
        for (const double y : {-1.0, 1.0}) {
          const std::array inner_logical{x, y, -1.0};
          const std::array outer_logical{x, y, 1.0};
          check_point(inner_logical, inner_radius);
          check_point(outer_logical, outer_radius);

          if (not(which_wedges == WedgeHalves::LowerOnly and x == 1.0) and
              not(which_wedges == WedgeHalves::UpperOnly and x == -1.0)) {
            CHECK(map(inner_logical)[2] ==
                  approx(inner_radius * one_over_denominator));
            CHECK(map(outer_logical)[2] ==
                  approx(outer_radius * one_over_denominator));
          }
        }
      }

      // Check at the center of the full wedge, as it is easy to
      // calculate expected values there.
      const auto check_center = [&](const double zeta,
                                    const double expected_radius) {
        const double xi_center = which_wedges == WedgeHalves::UpperOnly   ? -1.0
                                 : which_wedges == WedgeHalves::LowerOnly ? 1.0
                                                                          : 0.0;
        const std::array logical_point{xi_center, 0.0, zeta};
        check_point(logical_point, expected_radius);

        CAPTURE(logical_point);

        const auto mapped_point = map(logical_point);
        CAPTURE(mapped_point);

        CHECK(mapped_point[0] == approx(0.0));
        CHECK(mapped_point[1] == approx(0.0));
        CHECK(mapped_point[2] == approx(expected_radius));

        const auto jacobian = map.jacobian(logical_point);
        const auto inv_jacobian = map.inv_jacobian(logical_point);

        std::array expected_jacobian_diagonal{
            expected_radius *
                (with_equiangular_map ? 0.5 * opening_angle_xi : 1.0),
            expected_radius *
                (with_equiangular_map ? 0.5 * opening_angle_eta : 1.0),
            radius_scale_factor * square(expected_radius)};
        if (which_wedges != WedgeHalves::Both) {
          expected_jacobian_diagonal[0] *= 0.5;
        }
        for (size_t i = 0; i < 3; ++i) {
          CAPTURE(i);
          for (size_t j = 0; j < 3; ++j) {
            CAPTURE(j);
            if (i == j) {
              CHECK(jacobian.get(i, i) ==
                    approx(gsl::at(expected_jacobian_diagonal, i)));
              CHECK(inv_jacobian.get(i, i) ==
                    approx(1.0 / gsl::at(expected_jacobian_diagonal, i)));
            } else {
              CHECK(jacobian.get(i, j) == approx(0.0));
              CHECK(inv_jacobian.get(i, j) == approx(0.0));
            }
          }
        }
      };
      check_center(-1.0, inner_radius);
      check_center(1.0, outer_radius);

      check_point(random_inner_logical, inner_radius);
      check_point(random_outer_logical, outer_radius);
    }
  }
}

void test_wedge3d_fail() {
  INFO("Wedge3d fail");
  const Wedge3D map(0.2, 4.0, 0.0, 1.0, OrientationMap<3>{}, true);
  // Any point with z=0 should fail the inverse map.
  const std::array<double, 3> test_mapped_point1{{3.0, 3.0, 0.0}};
  const std::array<double, 3> test_mapped_point2{{-3.0, 3.0, 0.0}};

  // Any point with (x^2+y^2)/z^2 >= 1199 should fail the inverse map.
  const std::array<double, 3> test_mapped_point3{{sqrt(1198.0), 1.0, 1.0}};
  const std::array<double, 3> test_mapped_point4{{30.0, sqrt(299.0), 1.0}};
  const std::array<double, 3> test_mapped_point5{{30.0, sqrt(300.0), 1.0}};

  // These points are outside the mapped wedge. So inverse should either
  // return the correct inverse (which happens to be computable for
  // these points) or it should return nullopt.
  const std::array<double, 3> test_mapped_point6{{30.0, sqrt(298.0), 1.0}};
  const std::array<double, 3> test_mapped_point7{{2.0, 4.0, 6.0}};

  CHECK_FALSE(map.inverse(test_mapped_point1).has_value());
  CHECK_FALSE(map.inverse(test_mapped_point2).has_value());
  CHECK_FALSE(map.inverse(test_mapped_point3).has_value());
  CHECK_FALSE(map.inverse(test_mapped_point4).has_value());
  CHECK_FALSE(map.inverse(test_mapped_point5).has_value());
  if (map.inverse(test_mapped_point6).has_value()) {
    Approx my_approx = Approx::custom().epsilon(1.e-10).scale(1.0);
    CHECK_ITERABLE_CUSTOM_APPROX(map(map.inverse(test_mapped_point6).value()),
                                 test_mapped_point6, my_approx);
  }
  if (map.inverse(test_mapped_point7).has_value()) {
    CHECK_ITERABLE_APPROX(map(map.inverse(test_mapped_point7).value()),
                          test_mapped_point7);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge3D.Map", "[Domain][Unit]") {
  test_wedge3d_fail();
  test_wedge3d_all_directions();
  test_wedge3d_alignment();
  test_wedge3d_random_radii();
  test_wedge3d_large_radius();
  CHECK(not Wedge3D{}.is_identity());

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      Wedge3D(-0.2, 4.0, 0.0, 1.0, OrientationMap<3>{}, true),
      Catch::Matchers::ContainsSubstring(
          "The radius of the inner surface must be greater than zero."));
  CHECK_THROWS_WITH(
      Wedge3D(0.2, 4.0, -0.2, 1.0, OrientationMap<3>{}, true),
      Catch::Matchers::ContainsSubstring(
          "Sphericity of the inner surface must be between 0 and 1"));
  CHECK_THROWS_WITH(
      Wedge3D(0.2, 4.0, 0.0, -0.2, OrientationMap<3>{}, true),
      Catch::Matchers::ContainsSubstring(
          "Sphericity of the outer surface must be between 0 and 1"));
  CHECK_THROWS_WITH(
      Wedge3D(4.2, 4.0, 0.0, 1.0, OrientationMap<3>{}, true),
      Catch::Matchers::ContainsSubstring(
          "The radius of the outer surface must be greater than the "
          "radius of the inner surface."));
  CHECK_THROWS_WITH(
      Wedge3D(3.0, 4.0, 1.0, 0.0, OrientationMap<3>{}, true),
      Catch::Matchers::ContainsSubstring(
          "The arguments passed into the constructor for Wedge result in an "
          "object where the outer surface is pierced by the inner surface."));
  CHECK_THROWS_WITH(Wedge3D(0.2, 4.0, 0.8, 0.9, OrientationMap<3>{}, true,
                            Wedge3D::WedgeHalves::Both,
                            domain::CoordinateMaps::Distribution::Logarithmic),
                    Catch::Matchers::ContainsSubstring(
                        "Only the 'Linear' radial distribution is "
                        "supported for non-spherical wedges."));
  CHECK_THROWS_WITH(Wedge3D(0.2, 4.0, 0.8, 0.9, OrientationMap<3>{}, false,
                            Wedge3D::WedgeHalves::Both,
                            domain::CoordinateMaps::Distribution::Linear,
                            std::array<double, 2>{{M_PI_4 * 0.70, M_PI_4}}),
                    Catch::Matchers::ContainsSubstring(
                        "If using opening angles other than pi/2, then the "
                        "equiangular map option must be turned on."));
#endif
}
}  // namespace domain
