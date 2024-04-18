// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Particles/MonteCarlo/CouplingTermsForPropagation.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/Gsl.hpp"

using hydro::units::nuclear::proton_mass;

namespace Particles::MonteCarlo {

  void AddCouplingTermsForPropagation(
    const gsl::not_null<Scalar<DataVector>*> coupling_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        coupling_tilde_s,
    const gsl::not_null<Scalar<DataVector>*> coupling_rho_ye,
    const Packet& packet, const double dt, const double absorption_opacity,
    const double scattering_opacity, const double fluid_frame_energy,
    const double lapse, const double lorentz_factor,
    const std::array<double, 3>& lower_spatial_four_velocity_packet) {
  const size_t& idx = packet.index_of_closest_grid_point;
  // Energy coupling term.
  coupling_tilde_tau->get()[idx] +=
      dt * absorption_opacity * packet.number_of_neutrinos *
          fluid_frame_energy * lapse +
      dt * scattering_opacity * packet.number_of_neutrinos *
          fluid_frame_energy *
          (lapse -
           fluid_frame_energy * lorentz_factor / packet.momentum_upper_t);
  // Momentum coupling term
  for (size_t d = 0; d < 3; d++) {
    coupling_tilde_s->get(d)[idx] +=
        dt / packet.momentum_upper_t * packet.number_of_neutrinos *
        fluid_frame_energy *
        (packet.momentum.get(d) * (absorption_opacity + scattering_opacity) -
         scattering_opacity * fluid_frame_energy *
             gsl::at(lower_spatial_four_velocity_packet, d));
  }
  // Lepton number coupling term
  if (packet.species < 2) {
    coupling_rho_ye->get()[idx] +=
        (packet.species == 0 ? 1.0 : -1.0) * proton_mass * dt /
        packet.momentum_upper_t * absorption_opacity * fluid_frame_energy *
        packet.number_of_neutrinos;
  }
}

} // namespace Particles::MonteCarlo
