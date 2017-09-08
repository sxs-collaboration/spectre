// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <random>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Utilities/StdHelpers.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Wedge2D", "[Domain][Unit]") {
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
  for (size_t i = 0; i < 2; ++i) {
    CAPTURE_PRECISE(gsl::at(lower_right_corner, i));
    CAPTURE_PRECISE(gsl::at(upper_right_corner, i));
  }

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
                                             Direction<2>::upper_xi());
  const CoordinateMaps::Wedge2D map_upper_eta(random_inner_radius_upper_eta,
                                              random_outer_radius_upper_eta,
                                              Direction<2>::upper_eta());
  const CoordinateMaps::Wedge2D map_lower_xi(random_inner_radius_lower_xi,
                                             random_outer_radius_lower_xi,
                                             Direction<2>::lower_xi());
  const CoordinateMaps::Wedge2D map_lower_eta(random_inner_radius_lower_eta,
                                              random_outer_radius_lower_eta,
                                              Direction<2>::lower_eta());
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

  for (const auto& direction : Direction<2>::all_directions()) {
    const CoordinateMaps::Wedge2D map{inner_radius, outer_radius, direction};
    const std::array<double, 2> test_point{{xi, eta}};

    test_forward_jacobian(map, test_point);
    test_inverse_jacobian(map, test_point);

    CHECK(serialize_and_deserialize(map) == map);

    // Test that each function compiles when used with CoordinateMap
    const auto coord_map = make_coordinate_map<Frame::Logical, Frame::Grid>(
        CoordinateMaps::Wedge2D{inner_radius, outer_radius, direction});

    const auto lower_right_corner_tensor = [&lower_right_corner]() {
      tnsr::I<double, 2, Frame::Logical> point_as_tensor{};
      for (size_t i = 0; i < 2; ++i) {
        point_as_tensor.get(i) = gsl::at(lower_right_corner, i);
      }
      return point_as_tensor;
    }();

    CHECK(coord_map(lower_right_corner_tensor).get(0) ==
          approx(gsl::at(map(lower_right_corner), 0)));
    CHECK(coord_map.jacobian(lower_right_corner_tensor).get(0, 0) ==
          map.jacobian(lower_right_corner).get(0, 0));
    CHECK(coord_map.inv_jacobian(lower_right_corner_tensor).get(0, 0) ==
          map.inv_jacobian(lower_right_corner).get(0, 0));
  }
}
