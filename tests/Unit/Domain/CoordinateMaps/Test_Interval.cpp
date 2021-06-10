// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <optional>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"

namespace domain {
namespace {
void test_coordinate_map(
    const double xA, const double xB, const double xa, const double xb,
    const CoordinateMaps::Distribution distribution,
    const std::optional<double> singularity_pos = std::nullopt) {
  CoordinateMaps::Interval map(xA, xB, xa, xb, distribution, singularity_pos);
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(xA, xB);
  const auto sample_points = make_with_random_values<std::array<double, 10>>(
      make_not_null(&generator), make_not_null(&dist));
  for (double scalar : sample_points) {
    const std::array<double, 1> random_point{{scalar}};
    test_coordinate_map_argument_types(map, random_point);
    test_inverse_map(map, random_point);
    test_jacobian(map, random_point);
    test_inv_jacobian(map, random_point);
  }
  const std::array<double, 1> point_xA{{xA}};
  const std::array<double, 1> point_xB{{xB}};
  CHECK(get<0>(map(point_xA)) == approx(xa));
  CHECK(get<0>(map(point_xB)) == approx(xb));

  // Check inequivalence operator
  CHECK_FALSE(map != map);
  test_serialization(map);
  CHECK(not map.is_identity());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Interval", "[Domain][Unit]") {
  const std::array<double, 1> point_1{{-0.5}};
  const std::array<double, 1> point_2{{1 / 3.}};
  const std::array<double, 1> point_3{{12 / 7.}};

  test_coordinate_map(-1.0, 2.0, -3.1, 2.7,
                      CoordinateMaps::Distribution::Linear);
  CoordinateMaps::Interval linear_map(-1.0, 2.0, -3.1, 2.7,
                                      CoordinateMaps::Distribution::Linear);
  CHECK(get<0>(linear_map(point_1)) == approx(-6.4 / 3.));
  CHECK(get<0>(linear_map(point_2)) == approx(-4.7 / 9.));
  CHECK(get<0>(linear_map(point_3)) == approx(90.2 / 42.));

  test_coordinate_map(-1.0, 2.0, -3.1, 2.7,
                      CoordinateMaps::Distribution::Equiangular);
  CoordinateMaps::Interval equiangular_map(
      -1.0, 2.0, -3.1, 2.7, CoordinateMaps::Distribution::Equiangular);
  CHECK(get<0>(equiangular_map(point_1)) == approx(-1.87431578064991));
  CHECK(get<0>(equiangular_map(point_2)) == approx(-0.45371712422518));
  CHECK(get<0>(equiangular_map(point_3)) == approx(1.9402974025886));

  test_coordinate_map(-1.0, 2.0, -3.1, 2.7,
                      CoordinateMaps::Distribution::Logarithmic, -3.2);
  CoordinateMaps::Interval logarithmic_map_left(
      -1.0, 2.0, -3.1, 2.7, CoordinateMaps::Distribution::Logarithmic, -3.2);
  CHECK(get<0>(logarithmic_map_left(point_1)) == approx(-3.0026932232316066));
  CHECK(get<0>(logarithmic_map_left(point_2)) == approx(-2.5875856781465209));
  CHECK(get<0>(logarithmic_map_left(point_3)) == approx(0.80128456770888800));

  test_coordinate_map(-1.0, 2.0, -3.1, 2.7,
                      CoordinateMaps::Distribution::Logarithmic, 2.8);
  CoordinateMaps::Interval logarithmic_map_right(
      -1.0, 2.0, -3.1, 2.7, CoordinateMaps::Distribution::Logarithmic, 2.8);
  CHECK(get<0>(logarithmic_map_right(point_1)) == approx(-0.190267286625263));
  CHECK(get<0>(logarithmic_map_right(point_2)) == approx(1.836599930886074));
  CHECK(get<0>(logarithmic_map_right(point_3)) == approx(2.652547353227159));

  test_coordinate_map(-1.0, 2.0, -3.1, 2.7,
                      CoordinateMaps::Distribution::Inverse, -3.2);
  CoordinateMaps::Interval inverse_map_left(
      -1.0, 2.0, -3.1, 2.7, CoordinateMaps::Distribution::Inverse, -3.2);
  CHECK(get<0>(inverse_map_left(point_1)) == approx(-3.080405405405405));
  CHECK(get<0>(inverse_map_left(point_2)) == approx(-3.022408026755852));
  CHECK(get<0>(inverse_map_left(point_3)) == approx(-2.295620437956204));

  test_coordinate_map(-1.0, 2.0, -3.1, 2.7,
                      CoordinateMaps::Distribution::Inverse, 2.8);
  CoordinateMaps::Interval inverse_map_right(
      -1.0, 2.0, -3.1, 2.7, CoordinateMaps::Distribution::Inverse, 2.8);
  CHECK(get<0>(inverse_map_right(point_1)) == approx(2.246875));
  CHECK(get<0>(inverse_map_right(point_2)) == approx(2.579668049792531120));
  CHECK(get<0>(inverse_map_right(point_3)) == approx(2.6896705253784505788));

  check_if_map_is_identity(CoordinateMaps::Interval{
      -1.5, 1.0, -1.5, 1.0, CoordinateMaps::Distribution::Linear});
}
}  // namespace domain
