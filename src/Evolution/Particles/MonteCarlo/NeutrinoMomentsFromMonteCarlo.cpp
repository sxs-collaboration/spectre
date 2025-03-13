// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Particles/MonteCarlo/NeutrinoMomentsFromMonteCarlo.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/CellVolume.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace Particles::MonteCarlo {

void inertial_frame_energy_density(
    gsl::not_null<Scalar<DataVector>*> fluid_frame_energy_density,
    const std::vector<Packet>& packets, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_determinant_spatial_metric,
    const Mesh<3>& mesh,
    const Scalar<DataVector>& det_jacobian_logical_to_inertial) {
  Scalar<DataVector> cell_inertial_three_volume(lapse);
  cell_inertial_coordinate_three_volume_finite_difference(
      &cell_inertial_three_volume, mesh, det_jacobian_logical_to_inertial);
  *fluid_frame_energy_density = make_with_value<Scalar<DataVector>>(lapse, 0.0);
  ;
  for (const auto& packet : packets) {
    const size_t& idx = packet.index_of_closest_grid_point;
    get(*fluid_frame_energy_density)[idx] +=
        get(lapse)[idx] / get(cell_inertial_three_volume)[idx] /
        get(sqrt_determinant_spatial_metric)[idx] * packet.momentum_upper_t *
        packet.number_of_neutrinos;
  }
}

void InertialFrameEnergyDensityCompute::function(
    gsl::not_null<return_type*> inertial_frame_density,
    const std::vector<Packet>& packets, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_determinant_spatial_metric,
    const Mesh<3>& mesh,
    const Scalar<DataVector>& det_inv_jacobian_logical_to_inertial) {
  Scalar<DataVector> det_jacobian_logical_to_inertial(lapse);
  get(det_jacobian_logical_to_inertial) =
      1.0 / get(det_inv_jacobian_logical_to_inertial);
  inertial_frame_energy_density(inertial_frame_density, packets, lapse,
                                sqrt_determinant_spatial_metric, mesh,
                                det_jacobian_logical_to_inertial);
}

}  // namespace Particles::MonteCarlo
