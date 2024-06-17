// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Particles/MonteCarlo/CellVolume.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace Particles::MonteCarlo {

void cell_proper_four_volume_finite_difference(
    const gsl::not_null<Scalar<DataVector>*> cell_proper_four_volume,
    const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& determinant_spatial_metric,
    const double time_step, const Mesh<3>& mesh,
    const Scalar<DataVector>& det_jacobian_logical_to_inertial) {
  const double cell_logical_volume =
    8.0 / static_cast<double>(mesh.number_of_grid_points());
  cell_proper_four_volume->get() =
      get(lapse) * get(determinant_spatial_metric) * time_step *
      cell_logical_volume * get(det_jacobian_logical_to_inertial);
}

void cell_inertial_coordinate_three_volume_finite_difference(
    gsl::not_null<Scalar<DataVector>*> cell_inertial_three_volume,
    const Mesh<3>& mesh,
    const Scalar<DataVector>& det_jacobian_logical_to_inertial) {
  const double cell_logical_volume =
    8.0 / static_cast<double>(mesh.number_of_grid_points());
  cell_inertial_three_volume->get() =
      cell_logical_volume * get(det_jacobian_logical_to_inertial);
}

}  // namespace Particles::MonteCarlo
