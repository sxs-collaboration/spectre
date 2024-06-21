// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <optional>
#include <random>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;

template <size_t Dim, typename T>
class DirectionalIdMap;

namespace Frame {
struct Fluid;
}  // namespace Frame

namespace gsl {
  template <typename T>
  class not_null;
}  // namespace gsl

template<size_t Dim>
class Mesh;

namespace Particles::MonteCarlo {
template<size_t EnergyBins,size_t NeutrinoSpecies>
class NeutrinoInteractionTable;

struct Packet;
}  // namespace Particles::MonteCarlo

namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
/// \endcond

/// Items related to the evolution of particles
/// Items related to Monte-Carlo radiation transport
namespace Particles::MonteCarlo {

namespace detail {

void draw_single_packet(gsl::not_null<double*> time,
                        gsl::not_null<std::array<double, 3>*> coord,
                        gsl::not_null<std::array<double, 3>*> momentum,
                        gsl::not_null<std::mt19937*> random_number_generator);
}  // namespace detail


/// Structure containing Monte-Carlo function templated on EnergyBins
/// and/or NeutrinoSpecies
template <size_t EnergyBins, size_t NeutrinoSpecies>
struct TemplatedLocalFunctions {
  /*!
   * \brief Function to take a single Monte Carlo time step on a
   * finite difference element.
   */
  void take_time_step_on_element(
      gsl::not_null<std::vector<Packet>*> packets,
      gsl::not_null<std::mt19937*> random_number_generator,
      gsl::not_null<std::array<DataVector, NeutrinoSpecies>*>
          single_packet_energy,

      double start_time, double target_end_time,
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
      const NeutrinoInteractionTable<EnergyBins, NeutrinoSpecies>&
          interaction_table,
      const Scalar<DataVector>& electron_fraction,
      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& temperature,
      const Scalar<DataVector>& lorentz_factor,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          lower_spatial_four_velocity,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
      const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
      const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
      const tnsr::iJJ<DataVector, 3, Frame::Inertial>& d_inv_spatial_metric,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& determinant_spatial_metric,
      const Scalar<DataVector>& cell_light_crossing_time, const Mesh<3>& mesh,
      const tnsr::I<DataVector, 3, Frame::ElementLogical>& mesh_coordinates,
      size_t num_ghost_zones,
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          mesh_velocity,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>&
          inverse_jacobian_logical_to_inertial,
      const Scalar<DataVector>& det_jacobian_logical_to_inertial,
      const Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
          inertial_to_fluid_jacobian,
      const InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
          inertial_to_fluid_inverse_jacobian,
      const DirectionalIdMap<3, std::optional<DataVector>>&
          electron_fraction_ghost,
      const DirectionalIdMap<3, std::optional<DataVector>>&
          baryon_density_ghost,
      const DirectionalIdMap<3, std::optional<DataVector>>& temperature_ghost,
      const DirectionalIdMap<3, std::optional<DataVector>>&
          cell_light_crossing_time_ghost);

  /*!
   * \brief Function emitting Monte Carlo packets
   *
   * We emit a total energy emissivity_in_cell * cell_proper_four_volume
   * for each grid
   * cell, neutrino species, and neutrino energy bin. We aim for packets of
   * energy single_packet_energy in the fluid frame. The number of packets
   * is, in each bin b and for each species s,
   * `N = emission_in_cell[s][b] / single_packet_energy[s]`
   * and randomly rounded up or down to get integer number of packets (i.e.
   * if we want N=2.6 packets, there is a 60 percent chance of creating a 3rd
   * packet). The position of the packets is drawn from a homogeneous
   * distribution in the logical frame, and the direction of propagation of the
   * neutrinos is drawn from an isotropic distribution in the fluid frame.
   * Specifically, the 4-momentum of a packet of energy nu in the fluid frame
   * is
   * \f{align}
   * p^t &= \nu \\
   * p^x &= \nu * sin(\theta) * cos(\phi) \\
   * p^y &= \nu * sin(\theta) * sin(\phi) \\
   * p^z &= \nu * cos(theta)
   * \f}
   * with \f$cos(\theta)\f$ drawn from a uniform distribution in [-1,1] and
   * \f$\phi\f$ from a uniform distribution in \f$[0,2*\pi]\f$. We
   * transform to the inertial frame \f$p_t\f$ and \f$p^x,p^y,p^z\f$ using the
   * jacobian/inverse jacobian passed as option. The number of neutrinos in
   * each packet is defined as
   * `n = single_packet_energy[s] / energy_at_bin_center[b]`
   * Note that the packet energy is in code units and energy of a bin in MeV.
   *
   * All tensors are assumed to correspond to live points only, except for
   * the coupling terms and emissivity, which include ghost zones.
   */
  void emit_packets(
      gsl::not_null<std::vector<Packet>*> packets,
      gsl::not_null<std::mt19937*> random_number_generator,
      gsl::not_null<Scalar<DataVector>*> coupling_tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> coupling_tilde_s,
      gsl::not_null<Scalar<DataVector>*> coupling_rho_ye,
      const double& time_start_step, const double& time_step,
      const Mesh<3>& mesh, size_t num_ghost_zones,
      const std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>&
          emissivity_in_cell,
      const std::array<DataVector, NeutrinoSpecies>& single_packet_energy,
      const std::array<double, EnergyBins>& energy_at_bin_center,
      const Scalar<DataVector>& lorentz_factor,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          lower_spatial_four_velocity,
      const Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
          inertial_to_fluid_jacobian,
      const InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
        inertial_to_fluid_inverse_jacobian,
      const Scalar<DataVector>& cell_proper_four_volume);

  /*!
   * \brief Evolve Monte-Carlo packets within an element.
   *
   * Evolve packets until (approximately) the provided final time
   * following the methods of \cite Foucart:2021mcb.
   * The vector of Packets should contain all MC packets that we wish
   * to advance in time. Note that this function only handles
   * propagation / absorption / scattering of packets, but not
   * emission. The final time of the packet may differ from the
   * desired final time by up to 5 percent, due to the fact that when
   * using the diffusion approximation (for large scattering
   * opacities) we take fixed time steps in the fluid frame,
   * leading to unpredictable time steps in the inertial frame.
   *
   * The absorption and scattering opacity tables include ghost
   * zones, and so do the coupling terms. Other variables are
   * only using live points.
   */
  void evolve_packets(
      gsl::not_null<std::vector<Packet>*> packets,
      gsl::not_null<std::mt19937*> random_number_generator,
      gsl::not_null<Scalar<DataVector>*> coupling_tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> coupling_tilde_s,
      gsl::not_null<Scalar<DataVector>*> coupling_rho_ye, double final_time,
      const Mesh<3>& mesh,
      const tnsr::I<DataVector, 3, Frame::ElementLogical>& mesh_coordinates,
      size_t num_ghost_zones,
      const std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>&
          absorption_opacity_table,
      const std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>&
          scattering_opacity_table,
      const std::array<double, EnergyBins>& energy_at_bin_center,
      const Scalar<DataVector>& lorentz_factor,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          lower_spatial_four_velocity,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
      const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
      const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
      const tnsr::iJJ<DataVector, 3, Frame::Inertial>& d_inv_spatial_metric,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& cell_light_crossing_time,
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          mesh_velocity,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>&
          inverse_jacobian_logical_to_inertial,
      const Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
          inertial_to_fluid_jacobian,
      const InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
          inertial_to_fluid_inverse_jacobian);

  /// Interpolate the opacities from values tabulated as a function of
  /// neutrino energy in the fluid frame, to the value at the given
  /// fluid frame energy.
  /// We interpolate the logarithm of the opacity, linearly in fluid
  /// frame energy.
  void interpolate_opacities_at_fluid_energy(
      gsl::not_null<double*> absorption_opacity_packet,
      gsl::not_null<double*> scattering_opacity_packet,
      double fluid_frame_energy, size_t species, size_t index,
      const std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>&
          absorption_opacity_table,
      const std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>&
          scattering_opacity_table,
      const std::array<double, EnergyBins>& energy_at_bin_center);

  /// Function responsible to correct emissivity, absorption, and scattering
  /// rates in regions where there is a stiff coupling between the neutrinos
  /// and the fluid. This is done by transferring a fraction
  /// fraction_ka_to_ks of the absorption opacity to scattering opacity,
  /// while multiplying the emissivity by ( 1 - fraction_ka_to_ks ) to
  /// keep the equilibrium energy density constant.
  void implicit_monte_carlo_interaction_rates(
      gsl::not_null<
          std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>*>
          emissivity_in_cell,
      gsl::not_null<
          std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>*>
          absorption_opacity,
      gsl::not_null<
          std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>*>
          scattering_opacity,
      gsl::not_null<
          std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>*>
          fraction_ka_to_ks,
      const Scalar<DataVector>& cell_light_crossing_time,
      const Scalar<DataVector>& electron_fraction,
      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& temperature, double minimum_temperature,
      const NeutrinoInteractionTable<EnergyBins, NeutrinoSpecies>&
          interaction_table,
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state);

  // Floor for opacity values (absorption or scattering).
  // This is used in two places. First, when interpolating opacities
  // in energy space, we interpolate log(max(kappa,opacity_floor)),
  // with kappa the tabulated value. Second, if kappa = opacity_floor
  // for some interaction, we assume that the interaction never happens.
  const double opacity_floor = 1.e-100;
};

}  // namespace Particles::MonteCarlo
