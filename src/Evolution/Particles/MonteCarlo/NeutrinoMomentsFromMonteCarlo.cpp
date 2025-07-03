// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Particles/MonteCarlo/NeutrinoMomentsFromMonteCarlo.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/CellVolume.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"

using hydro::units::nuclear::proton_mass;

namespace Particles::MonteCarlo {

void inertial_frame_energy_density_function(
    gsl::not_null<Scalar<DataVector>*> inertial_frame_energy_density,
    const std::vector<Packet>& packets, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_determinant_spatial_metric,
    const Mesh<3>& mesh,
    const Scalar<DataVector>& det_jacobian_logical_to_inertial) {
  Scalar<DataVector> cell_inertial_three_volume(lapse);
  cell_inertial_coordinate_three_volume_finite_difference(
      &cell_inertial_three_volume, mesh, det_jacobian_logical_to_inertial);
  *inertial_frame_energy_density =
      make_with_value<Scalar<DataVector>>(lapse, 0.0);

  for (const auto& packet : packets) {
    const size_t& idx = packet.index_of_closest_grid_point;
    get(*inertial_frame_energy_density)[idx] +=
        get(lapse)[idx] / get(cell_inertial_three_volume)[idx] /
        get(sqrt_determinant_spatial_metric)[idx] * packet.momentum_upper_t *
        packet.number_of_neutrinos;
  }
}

void InertialFrameEnergyDensityCompute::function(
    gsl::not_null<return_type*> inertial_frame_energy_density,
    const std::vector<Packet>& packets, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_determinant_spatial_metric,
    const Mesh<3>& mesh,
    const Scalar<DataVector>& det_inv_jacobian_logical_to_inertial) {
  Scalar<DataVector> det_jacobian_logical_to_inertial(lapse);
  get(det_jacobian_logical_to_inertial) =
      1.0 / get(det_inv_jacobian_logical_to_inertial);
  inertial_frame_energy_density_function(
      inertial_frame_energy_density, packets, lapse,
      sqrt_determinant_spatial_metric, mesh, det_jacobian_logical_to_inertial);
}

template <size_t Nspecies>
void inertial_frame_energy_density_per_species_function(
    gsl::not_null<tnsr::i<DataVector, Nspecies, Frame::Inertial>*>
        inertial_frame_energy_density,
    const std::vector<Packet>& packets, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_determinant_spatial_metric,
    const Mesh<3>& mesh,
    const Scalar<DataVector>& det_jacobian_logical_to_inertial) {
  Scalar<DataVector> cell_inertial_three_volume(lapse);
  cell_inertial_coordinate_three_volume_finite_difference(
      &cell_inertial_three_volume, mesh, det_jacobian_logical_to_inertial);
  *inertial_frame_energy_density =
      make_with_value<tnsr::i<DataVector, Nspecies, Frame::Inertial>>(lapse,
                                                                      0.0);

  for (const auto& packet : packets) {
    const size_t& idx = packet.index_of_closest_grid_point;
    const size_t& sp = packet.species;
    inertial_frame_energy_density->get(sp)[idx] +=
        get(lapse)[idx] / get(cell_inertial_three_volume)[idx] /
        get(sqrt_determinant_spatial_metric)[idx] * packet.momentum_upper_t *
        packet.number_of_neutrinos;
  }
}

template <size_t Nspecies>
void InertialFrameEnergyDensityPerSpeciesCompute<Nspecies>::function(
    gsl::not_null<return_type*> inertial_frame_energy_density,
    const std::vector<Packet>& packets, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_determinant_spatial_metric,
    const Mesh<3>& mesh,
    const Scalar<DataVector>& det_inv_jacobian_logical_to_inertial) {
  Scalar<DataVector> det_jacobian_logical_to_inertial(lapse);
  get(det_jacobian_logical_to_inertial) =
      1.0 / get(det_inv_jacobian_logical_to_inertial);
  inertial_frame_energy_density_per_species_function<Nspecies>(
      inertial_frame_energy_density, packets, lapse,
      sqrt_determinant_spatial_metric, mesh, det_jacobian_logical_to_inertial);
}

void inertial_frame_lepton_number_density_function(
    gsl::not_null<Scalar<DataVector>*> inertial_frame_lepton_number_density,
    const std::vector<Packet>& packets,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& lower_spatial_four_velocity,
    const Scalar<DataVector>& lapse,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_determinant_spatial_metric,
    const Mesh<3>& mesh,
    const Scalar<DataVector>& det_jacobian_logical_to_inertial) {
  Scalar<DataVector> cell_inertial_three_volume(
      sqrt_determinant_spatial_metric);
  cell_inertial_coordinate_three_volume_finite_difference(
      &cell_inertial_three_volume, mesh, det_jacobian_logical_to_inertial);
  *inertial_frame_lepton_number_density =
      make_with_value<Scalar<DataVector>>(sqrt_determinant_spatial_metric, 0.0);

  for (const auto& packet : packets) {
    if (packet.species < 2) {
      const size_t& idx = packet.index_of_closest_grid_point;
      const double fluid_frame_energy = compute_fluid_frame_energy(
          packet, lorentz_factor, lower_spatial_four_velocity, lapse,
          inv_spatial_metric);
      get(*inertial_frame_lepton_number_density)[idx] +=
          (packet.species == 0 ? 1.0 : -1.0) * proton_mass /
          get(cell_inertial_three_volume)[idx] /
          get(sqrt_determinant_spatial_metric)[idx] *
          packet.number_of_neutrinos * fluid_frame_energy /
          packet.momentum_upper_t;
    }
  }
}

void InertialFrameLeptonNumberDensityCompute::function(
    gsl::not_null<return_type*> inertial_frame_lepton_number_density,
    const std::vector<Packet>& packets,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lapse,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_determinant_spatial_metric,
    const Mesh<3>& mesh,
    const Scalar<DataVector>& det_inv_jacobian_logical_to_inertial) {
  tnsr::i<DataVector, 3, Frame::Inertial> lower_spatial_four_velocity =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(lapse, 0.0);
  raise_or_lower_index(make_not_null(&lower_spatial_four_velocity),
                       spatial_velocity, spatial_metric);
  for (size_t i = 0; i < 3; i++) {
    lower_spatial_four_velocity.get(i) *= get(lorentz_factor);
  }
  Scalar<DataVector> det_jacobian_logical_to_inertial(
      sqrt_determinant_spatial_metric);
  get(det_jacobian_logical_to_inertial) =
      1.0 / get(det_inv_jacobian_logical_to_inertial);
  inertial_frame_lepton_number_density_function(
      inertial_frame_lepton_number_density, packets, lorentz_factor,
      lower_spatial_four_velocity, lapse, inv_spatial_metric,
      sqrt_determinant_spatial_metric, mesh, det_jacobian_logical_to_inertial);
}

}  // namespace Particles::MonteCarlo

template class Particles::MonteCarlo::
    InertialFrameEnergyDensityPerSpeciesCompute<3>;
template void
Particles::MonteCarlo::inertial_frame_energy_density_per_species_function<3>(
    gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        inertial_frame_energy_density,
    const std::vector<Particles::MonteCarlo::Packet>& packets,
    const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_determinant_spatial_metric,
    const Mesh<3>& mesh,
    const Scalar<DataVector>& det_jacobian_logical_to_inertial);
