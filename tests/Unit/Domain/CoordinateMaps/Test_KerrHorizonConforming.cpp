// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/KerrHorizonConforming.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace {

void test_map_helpers(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution mass_dist{0.5, 2.};
  const double mass = mass_dist(*generator);
  std::uniform_real_distribution spin_dist{-1. / sqrt(3), 1. / sqrt(3)};
  const auto spin =
      make_with_random_values<std::array<double, 3>>(generator, spin_dist, 3);
  const domain::CoordinateMaps::KerrHorizonConforming map(mass, spin);
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
  std::uniform_real_distribution mass_dist{0.5, 2.};
  const double mass = mass_dist(*generator);
  const std::array<double, 3> spin{0., 0., 0.};
  std::uniform_real_distribution dist{-10., 10.};
  const auto coords =
      make_with_random_values<std::array<double, 3>>(generator, dist, 3);
  const auto map = domain::CoordinateMaps::KerrHorizonConforming(mass, spin);
  const auto res = map(coords);
  CHECK_ITERABLE_APPROX(coords, res);
}

void test_random_spin(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution mass_dist{0.5, 2.};
  const double mass = mass_dist(*generator);
  std::uniform_real_distribution spin_dist{-1. / sqrt(3), 1. / sqrt(3)};
  const auto spin =
      make_with_random_values<std::array<double, 3>>(generator, spin_dist, 3);
  // test coordinates on unit sphere
  std::uniform_real_distribution point_dist{-10., 10.};
  const size_t number_of_points = 1000;
  auto coords = make_with_random_values<std::array<DataVector, 3>>(
      generator, point_dist, number_of_points);
  coords = coords / magnitude(coords);

  const auto map = domain::CoordinateMaps::KerrHorizonConforming(mass, spin);
  const auto mapped_coords = map(coords);

  const DataVector coords_sq_min_spin_sq =
      dot(mapped_coords, mapped_coords) - square(mass) * dot(spin, spin);

  // explicit expression for Kerr-Schild radius r
  DataVector r = coords_sq_min_spin_sq +
                 sqrt(coords_sq_min_spin_sq * coords_sq_min_spin_sq +
                      4 * square(mass * dot(mapped_coords, spin)));
  r = sqrt(r / 2.);

  for (const auto& val : r) {
    CHECK(val == approx(1.));
  }
}

void test_with_kerr_horizon(const double mass,
                            const std::array<double, 3>& dimensionless_spin) {
  CAPTURE(mass);
  CAPTURE(dimensionless_spin);

  // This is r_+
  const double horizon_kerrschild_radius =
      mass * (1. + sqrt(1. - dot(dimensionless_spin, dimensionless_spin)));
  CAPTURE(horizon_kerrschild_radius);

  // Set up a wedge with a Kerr-horizon-conforming inner surface.
  // The inner radius of r_+ will be mapped to an ellipsoid with constant
  // Boyer-Lindquist radius r_+.
  const domain::CoordinateMaps::Wedge<3> wedge_map{
      horizon_kerrschild_radius,
      2. * horizon_kerrschild_radius,
      1.,
      1.,
      {},
      false};
  const domain::CoordinateMaps::KerrHorizonConforming horizon_map{
      mass, dimensionless_spin};
  const auto coord_map =
      domain::make_coordinate_map_base<Frame::ElementLogical, Frame::Inertial>(
          wedge_map, horizon_map);

  // Get some coordinates on the inner face
  const Mesh<2> face_mesh{12, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const auto logical_coords =
      interface_logical_coordinates(face_mesh, Direction<3>::lower_zeta());
  const tnsr::I<DataVector, 3> x = (*coord_map)(logical_coords);

  // Check the coordinates on the inner face are on a Kerr horizon
  const std::array<DataVector, 2> theta_phi{
      {atan2(sqrt(square(get<0>(x)) + square(get<1>(x))), get<2>(x)),
       atan2(get<1>(x), get<0>(x))}};
  const DataVector expected_horizon_radii = get(
      gr::Solutions::kerr_horizon_radius(theta_phi, mass, dimensionless_spin));
  CHECK_ITERABLE_APPROX(get(magnitude(x)), expected_horizon_radii);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.KerrHorizonConforming",
                  "[Domain][Unit]") {
  MAKE_GENERATOR(generator);
  test_map_helpers(make_not_null(&generator));
  test_no_spin(make_not_null(&generator));
  test_random_spin(make_not_null(&generator));
  test_with_kerr_horizon(0.45, {{0., 0.2, 0.8}});
}
