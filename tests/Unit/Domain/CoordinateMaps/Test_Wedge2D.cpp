// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <random>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Utilities/StdHelpers.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge2D.Equidistant",
                  "[Domain][Unit]") {
  // Set up random number generator:
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> real_dis(-1, 1);
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

  const CoordinateMaps::Wedge2D map_upper_xi(random_inner_radius_upper_xi,
                                             random_outer_radius_upper_xi,
                                             Direction<2>::upper_xi(), false);
  const CoordinateMaps::Wedge2D map_upper_eta(random_inner_radius_upper_eta,
                                              random_outer_radius_upper_eta,
                                              Direction<2>::upper_eta(), false);
  const CoordinateMaps::Wedge2D map_lower_xi(random_inner_radius_lower_xi,
                                             random_outer_radius_lower_xi,
                                             Direction<2>::lower_xi(), false);
  const CoordinateMaps::Wedge2D map_lower_eta(random_inner_radius_lower_eta,
                                              random_outer_radius_lower_eta,
                                              Direction<2>::lower_eta(), false);
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

  const double dxi = 1e-5;
  const double deta = 1e-5;
  const double xi = real_dis(gen);
  CAPTURE_PRECISE(xi);
  const double eta = real_dis(gen);
  CAPTURE_PRECISE(eta);
  const double inner_radius = inner_dis(gen);
  CAPTURE_PRECISE(inner_radius);
  const double outer_radius = outer_dis(gen);
  CAPTURE_PRECISE(outer_radius);
  const std::array<double, 2> test_point{{xi, eta}};

  for (const auto& direction : Direction<2>::all_directions()) {
    const CoordinateMaps::Wedge2D map{inner_radius, outer_radius, direction,
                                      false};
    test_jacobian(map, test_point);

    test_inv_jacobian(map, test_point);

    test_serialization(map);

    test_coordinate_map_implementation(map);

    test_coordinate_map_argument_types<true>(map, test_point);

    test_inverse_map(map, test_point);
  }
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge2D.Equiangular",
                  "[Domain][Unit]") {
  // Set up random number generator:
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> real_dis(-1, 1);
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

  const CoordinateMaps::Wedge2D map_upper_xi(random_inner_radius_upper_xi,
                                             random_outer_radius_upper_xi,
                                             Direction<2>::upper_xi(), true);
  const CoordinateMaps::Wedge2D map_upper_eta(random_inner_radius_upper_eta,
                                              random_outer_radius_upper_eta,
                                              Direction<2>::upper_eta(), true);
  const CoordinateMaps::Wedge2D map_lower_xi(random_inner_radius_lower_xi,
                                             random_outer_radius_lower_xi,
                                             Direction<2>::lower_xi(), true);
  const CoordinateMaps::Wedge2D map_lower_eta(random_inner_radius_lower_eta,
                                              random_outer_radius_lower_eta,
                                              Direction<2>::lower_eta(), true);
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

  const double dxi = 1e-5;
  const double deta = 1e-5;
  const double xi = real_dis(gen);
  CAPTURE_PRECISE(xi);
  const double eta = real_dis(gen);
  CAPTURE_PRECISE(eta);
  const double inner_radius = inner_dis(gen);
  CAPTURE_PRECISE(inner_radius);
  const double outer_radius = outer_dis(gen);
  CAPTURE_PRECISE(outer_radius);
  const std::array<double, 2> test_point{{xi, eta}};

  for (const auto& direction : Direction<2>::all_directions()) {
    const CoordinateMaps::Wedge2D map{inner_radius, outer_radius, direction,
                                      true};
    test_jacobian(map, test_point);

    test_inv_jacobian(map, test_point);

    test_serialization(map);

    test_coordinate_map_implementation(map);

    test_coordinate_map_argument_types<true>(map, test_point);

    test_inverse_map(map, test_point);
  }
}
