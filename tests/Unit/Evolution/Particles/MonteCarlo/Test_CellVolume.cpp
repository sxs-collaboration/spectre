// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/CellVolume.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Particles.MonteCarloCellVolume",
                  "[Unit][Evolution]") {
  const Mesh<3> mesh(3, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered);

  const size_t dv_size = 27;
  DataVector zero_dv(dv_size, 0.0);
  Scalar<DataVector> lapse{DataVector(dv_size, 1.2)};
  Scalar<DataVector> determinant_spatial_metric{DataVector(dv_size, 0.9)};
  Scalar<DataVector> det_jacobian_logical_to_inertial{DataVector(dv_size, 1.1)};
  const double time_step = 0.6;

  Scalar<DataVector> cell_proper_four_volume{DataVector(dv_size, 0.0)};
  Scalar<DataVector> expected_cell_proper_four_volume{
      DataVector(dv_size, 8.0 / 27.0 * 1.2 * 0.9 * 0.6 * 1.1)};
  Particles::MonteCarlo::cell_proper_four_volume_finite_difference(
      &cell_proper_four_volume, lapse, determinant_spatial_metric, time_step,
      mesh, det_jacobian_logical_to_inertial);

  Scalar<DataVector> cell_inertial_three_volume{DataVector(dv_size, 0.0)};
  Scalar<DataVector> expected_cell_inertial_three_volume{
      DataVector(dv_size, 8.0 / 27.0 * 1.1)};
  Particles::MonteCarlo::
      cell_inertial_coordinate_three_volume_finite_difference(
          &cell_inertial_three_volume, mesh, det_jacobian_logical_to_inertial);

  CHECK(expected_cell_proper_four_volume == cell_proper_four_volume);
  CHECK(expected_cell_inertial_three_volume == cell_inertial_three_volume);
}
