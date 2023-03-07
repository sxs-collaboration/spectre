// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/RadiallyCompressedCoordinates.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"

namespace domain {
namespace {

void test_radially_compressed_coordinates(
    const CoordinateMaps::Distribution radial_distribution) {
  CAPTURE(radial_distribution);
  MAKE_GENERATOR(gen);
  const double inner_radius = 10.;
  const double outer_radius = 1.e9;
  const double expected_compressed_outer_radius = 90.;
  // Helper function that maps inertial to compressed radius for a single point
  const auto compressed_radius =
      [&inner_radius, &outer_radius,
       &radial_distribution](const double inertial_radius) -> double {
    return get<0>(radially_compressed_coordinates(
        tnsr::I<double, 3, Frame::Inertial>{{{inertial_radius, 0., 0.}}},
        inner_radius, outer_radius, radial_distribution));
  };
  {
    INFO("Inner radius is identical");
    CHECK(compressed_radius(inner_radius) == approx(inner_radius));
  }
  {
    INFO("Interior is identical");
    CHECK(compressed_radius(inner_radius / 2.) == approx(inner_radius / 2.));
  }
  {
    INFO("Outer radius is compressed");
    CHECK(compressed_radius(outer_radius) ==
          approx(expected_compressed_outer_radius));
  }
  {
    INFO("Consistency with Wedge map");
    // Check that grid points are distributed linearly in radially compressed
    // coordinates if the compression is consistent with the Wedge map
    const auto wedge_map =
        CoordinateMaps::Wedge<3>{inner_radius,
                                 outer_radius,
                                 1.0,
                                 1.0,
                                 {},
                                 true,
                                 CoordinateMaps::Wedge<3>::WedgeHalves::Both,
                                 radial_distribution};
    // Set up a point at a fraction of the radial distance covered by the wedge
    std::uniform_real_distribution<double> dist_fraction(0., 1.);
    const double radial_fraction = dist_fraction(gen);
    const double r_logical = -1. + radial_fraction * 2.;
    const double r_inertial =
        wedge_map(std::array<double, 3>{{0., 0., r_logical}})[2];
    const double r_compressed = compressed_radius(r_inertial);
    const double expected_r_compressed =
        inner_radius +
        radial_fraction * (expected_compressed_outer_radius - inner_radius);
    CHECK(r_compressed == approx(expected_r_compressed));
  }
  {
    INFO("Coordinates preserve angles");
    const size_t num_angles = 6;
    // extend below inner radius to check identity there
    std::uniform_real_distribution<double> dist_r(inner_radius / 2.,
                                                  outer_radius);
    std::uniform_real_distribution<double> dist_theta(0., M_PI);
    std::uniform_real_distribution<double> dist_phi(-M_PI, M_PI);
    const auto r = make_with_random_values<DataVector>(
        make_not_null(&gen), make_not_null(&dist_r), DataVector(5));
    const auto theta = make_with_random_values<DataVector>(
        make_not_null(&gen), make_not_null(&dist_theta),
        DataVector(num_angles));
    const auto phi = make_with_random_values<DataVector>(
        make_not_null(&gen), make_not_null(&dist_phi), DataVector(num_angles));
    for (size_t i = 0; i < r.size(); ++i) {
      tnsr::I<DataVector, 3, Frame::Inertial> x{num_angles};
      get<0>(x) = r[i] * cos(phi) * sin(theta);
      get<1>(x) = r[i] * sin(phi) * sin(theta);
      get<2>(x) = r[i] * cos(theta);
      const auto x_compressed = radially_compressed_coordinates(
          x, inner_radius, outer_radius, radial_distribution);
      const auto r_compressed = get(magnitude(x_compressed));
      // Check all compressed radii are the same
      CHECK_ITERABLE_APPROX(r_compressed,
                            DataVector(num_angles, compressed_radius(r[i])));
      // Check the angles didn't change
      const auto theta_compressed =
          atan2(hypot(get<0>(x_compressed), get<1>(x_compressed)),
                get<2>(x_compressed));
      const auto phi_compressed =
          atan2(get<1>(x_compressed), get<0>(x_compressed));
      CHECK_ITERABLE_APPROX(theta_compressed, theta);
      CHECK_ITERABLE_APPROX(phi_compressed, phi);
      // Also check some more consistency
      if (r[i] <= inner_radius) {
        CHECK(r_compressed[i] == approx(r[i]));
      } else {
        CHECK(r_compressed[i] < r[i]);
        CHECK(r_compressed[i] <= expected_compressed_outer_radius + 1.e-14);
      }
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.RadiallyCompressedCoordinates",
                  "[Domain][Unit]") {
  TestHelpers::db::test_simple_tag<
      Tags::RadiallyCompressedCoordinates<3, Frame::Inertial>>(
      "RadiallyCompressedCoordinates");
  TestHelpers::db::test_compute_tag<
      Tags::RadiallyCompressedCoordinatesCompute<3, Frame::Inertial>>(
      "RadiallyCompressedCoordinates");
  test_radially_compressed_coordinates(
      CoordinateMaps::Distribution::Logarithmic);
  test_radially_compressed_coordinates(CoordinateMaps::Distribution::Inverse);
}

}  // namespace domain
