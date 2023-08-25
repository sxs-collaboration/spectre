// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/QuadrupoleFormula.hpp"

namespace hydro {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.QuadrupoleFormula",
                  "[Unit][Hydro]") {
  TestHelpers::db::test_simple_tag<
      Tags::QuadrupoleMoment<
                 DataVector, 3, Frame::Inertial>>("QuadrupoleMoment");
  TestHelpers::db::test_simple_tag<
      Tags::QuadrupoleMomentDerivative<
                 DataVector, 3, Frame::Inertial>>("QuadrupoleMomentDerivative");
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Hydro/");
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(
      &quadrupole_moment<DataVector, 1, Frame::Inertial>,
      "QuadrupoleFormula", {"quadrupole_moment"}, {{{0.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<1>(
      &quadrupole_moment<DataVector, 3, Frame::Inertial>,
      "QuadrupoleFormula", {"quadrupole_moment"}, {{{0.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<1>(
      &quadrupole_moment_derivative<DataVector, 1, Frame::Inertial>,
      "QuadrupoleFormula", {"quadrupole_moment_derivative"}, {{{0.0, 1.0}}},
      used_for_size);
  pypp::check_with_random_values<1>(
      &quadrupole_moment_derivative<DataVector, 3, Frame::Inertial>,
      "QuadrupoleFormula", {"quadrupole_moment_derivative"}, {{{0.0, 1.0}}},
      used_for_size);

  const Mesh<3> mesh{12, SpatialDiscretization::Basis::Legendre,
                     SpatialDiscretization::Quadrature::GaussLobatto};
  const Scalar<DataVector> rho_flat{mesh.number_of_grid_points(), 1.28e-3};
  const Scalar<DataVector> velocity_squared{mesh.number_of_grid_points(), 0.04};
  const Scalar<DataVector> W = lorentz_factor(velocity_squared);
  const Scalar<DataVector> tildeD{get(W) * get(rho_flat)};

  const auto logical_coords = logical_coordinates(mesh);
  const size_t Dim=3;
  ElementMap<Dim, Frame::Inertial> logical_to_inertial_map{
      ElementId<Dim>{0},
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{})};
  const auto inertial_coords = logical_to_inertial_map(logical_coords);

  tnsr::I<DataVector, 3> velocity_x{mesh.number_of_grid_points(), 0.};
  get<0>(velocity_x) = 0.2;
  tnsr::I<DataVector, 3> velocity_y{mesh.number_of_grid_points(), 0.};
  get<1>(velocity_y) = 0.2;
  tnsr::I<DataVector, 3> velocity_z{mesh.number_of_grid_points(), 0.};
  get<2>(velocity_z) = 0.2;

  tnsr::ii<DataVector, 3, Frame::Inertial> q{};
  quadrupole_moment(make_not_null(&q), tildeD, inertial_coords);
  tnsr::ii<DataVector, 3, Frame::Inertial> qd_x{};
  quadrupole_moment_derivative(make_not_null(&qd_x), tildeD, inertial_coords,
                                                                    velocity_x);
  tnsr::ii<DataVector, 3, Frame::Inertial> qd_y{};
  quadrupole_moment_derivative(make_not_null(&qd_y), tildeD, inertial_coords,
                                                                    velocity_y);
  tnsr::ii<DataVector, 3, Frame::Inertial> qd_z{};
  quadrupole_moment_derivative(make_not_null(&qd_z), tildeD, inertial_coords,
                                                                    velocity_z);

  tnsr::ii<double, 3> q_integral{};
  tnsr::ii<double, 3> qd_x_integral{};
  tnsr::ii<double, 3> qd_y_integral{};
  tnsr::ii<double, 3> qd_z_integral{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      q_integral.get(i,j) = definite_integral(q.get(i,j), mesh);
      qd_x_integral.get(i,j) = definite_integral(qd_x.get(i,j), mesh);
      qd_y_integral.get(i,j) = definite_integral(qd_y.get(i,j), mesh);
      qd_z_integral.get(i,j) = definite_integral(qd_z.get(i,j), mesh);
      CHECK(qd_x_integral.get(i, j) == approx(0.0));
      CHECK(qd_y_integral.get(i, j) == approx(0.0));
      CHECK(qd_z_integral.get(i, j) == approx(0.0));
      if (i == j) {
        CHECK(q_integral.get(i, j) == approx(3.483718745291631e-3));
      }
      else {
        CHECK(q_integral.get(i, j) == approx(0.0));
      }
    }
  }
}

}  // namespace hydro
