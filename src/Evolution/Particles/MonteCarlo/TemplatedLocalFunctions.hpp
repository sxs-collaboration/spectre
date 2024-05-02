// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace Frame {
struct Fluid;
}  // namespace Frame
/// \endcond

/// Items related to the evolution of particles
/// Items related to Monte-Carlo radiation transport
namespace Particles::MonteCarlo {

struct Packet;

namespace detail {

void draw_single_packet(
    gsl::not_null<double*> time,
    gsl::not_null<std::array<double, 3>*> coord,
    gsl::not_null<std::array<double, 3>*> momentum,
    gsl::not_null<std::mt19937*> random_number_generator);
}  // namespace detail


/// \brief Contribution to the neutrino-matter coupling
/// terms from evolving a packet by time dt, given
/// absorption and scattering opacities absorption_opacity,
/// scattering_opacity.
///
/// \details We consider total momentum exchances, not
/// densities; thus when coupling to the evolution of the
/// energy/momentum density, division by the spatial volume is
/// necessary. We calculate the source terms of Eqs (62-64) of
/// \cite Foucart:2021mcb, with integrals calculated as in
/// Eqs (6-10) of that manuscript (except that we do not
/// divide by the coordinate volume V).
void AddCouplingTermsForPropagation(
    gsl::not_null<Scalar<DataVector>*> coupling_tilde_tau,
    gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        coupling_tilde_s,
    gsl::not_null<Scalar<DataVector>*> coupling_rho_ye,
    const Packet& packet, double dt, double absorption_opacity,
    double scattering_opacity, double fluid_frame_energy,
    double lapse, double lorentz_factor,
    const std::array<double, 3>& lower_spatial_four_velocity_packet);

/// Structure containing Monte-Carlo function templated on EnergyBins
/// and/or NeutrinoSpecies
template <size_t EnergyBins, size_t NeutrinoSpecies>
struct TemplatedLocalFunctions {
  /*!
   * \brief Function emitting Monte Carlo packets
   *
   * We emit a total energy emission_in_cell for each grid
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
   */
  void emit_packets(
      gsl::not_null<std::vector<Packet>*> packets,
      gsl::not_null<std::mt19937*> random_number_generator,
      gsl::not_null<Scalar<DataVector>*> coupling_tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> coupling_tilde_s,
      gsl::not_null<Scalar<DataVector>*> coupling_rho_ye,
      const double& time_start_step, const double& time_step,
      const Mesh<3>& mesh,
      const std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>&
          emission_in_cell,
      const std::array<DataVector, NeutrinoSpecies>& single_packet_energy,
      const std::array<double, EnergyBins>& energy_at_bin_center,
      const Scalar<DataVector>& lorentz_factor,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          lower_spatial_four_velocity,
      const Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
          inertial_to_fluid_jacobian,
      const InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
          inertial_to_fluid_inverse_jacobian);

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
   */
  void evolve_packets(
      gsl::not_null<std::vector<Packet>*> packets,
      gsl::not_null<std::mt19937*> random_number_generator,
      gsl::not_null<Scalar<DataVector>*> coupling_tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        coupling_tilde_s,
      gsl::not_null<Scalar<DataVector>*> coupling_rho_ye,
      double final_time, const Mesh<3>& mesh,
      const tnsr::I<DataVector, 3, Frame::ElementLogical>& mesh_coordinates,
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

  // Floor for opacity values (absorption or scattering).
  // This is used in two places. First, when interpolating opacities
  // in energy space, we interpolate log(max(kappa,opacity_floor)),
  // with kappa the tabulated value. Second, if kappa = opacity_floor
  // for some interaction, we assume that the interaction never happens.
  const double opacity_floor = 1.e-100;
};

}  // namespace Particles::MonteCarlo
