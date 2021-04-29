// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/KerrHorizonConforming.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace {

void test_map_helpers(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution spin_dist{-1. / sqrt(3), 1. / sqrt(3)};
  const auto spin =
      make_with_random_values<std::array<double, 3>>(generator, spin_dist, 3);
  const domain::CoordinateMaps::KerrHorizonConforming map(spin);
  std::uniform_real_distribution point_dist{-10., 10.};
  const auto random_point =
      make_with_random_values<std::array<double, 3>>(generator, point_dist, 3);
  test_serialization(map);
  test_coordinate_map_argument_types(map, random_point);
  test_inverse_map(map, random_point);
  test_jacobian(map, random_point);
  test_inv_jacobian(map, random_point);
}

void test_no_spin(const gsl::not_null<std::mt19937*> generator) {
  const std::array<double, 3> spin{0., 0., 0.};
  std::uniform_real_distribution dist{-10., 10.};
  const auto coords =
      make_with_random_values<std::array<double, 3>>(generator, dist, 3);
  const auto map = domain::CoordinateMaps::KerrHorizonConforming(spin);
  const auto res = map(coords);
  CHECK_ITERABLE_APPROX(coords, res);
}

void test_random_spin(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution spin_dist{-1. / sqrt(3), 1. / sqrt(3)};
  const auto spin =
      make_with_random_values<std::array<double, 3>>(generator, spin_dist, 3);
  // test coordinates on unit sphere
  std::uniform_real_distribution point_dist{-10., 10.};
  const size_t number_of_points = 1000;
  auto coords = make_with_random_values<std::array<DataVector, 3>>(
      generator, point_dist, number_of_points);
  coords = coords / magnitude(coords);

  const auto map = domain::CoordinateMaps::KerrHorizonConforming(spin);
  const auto mapped_coords = map(coords);

  const DataVector coords_sq_min_spin_sq =
      dot(mapped_coords, mapped_coords) - dot(spin, spin);

  // explicit expression for Kerr-Schild radius r
  DataVector r = coords_sq_min_spin_sq +
                 sqrt(coords_sq_min_spin_sq * coords_sq_min_spin_sq +
                      4 * dot(mapped_coords, spin) * dot(mapped_coords, spin));
  r = sqrt(r / 2.);

  for (const auto& val : r) {
    CHECK(val == approx(1.));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.KerrHorizonConforming",
                  "[Domain][Unit]") {
  MAKE_GENERATOR(generator);
  test_map_helpers(make_not_null(&generator));
  test_no_spin(make_not_null(&generator));
  test_random_spin(make_not_null(&generator));
}
