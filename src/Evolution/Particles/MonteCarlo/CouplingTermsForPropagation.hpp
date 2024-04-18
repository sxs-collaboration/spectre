// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace gsl {
  template <typename T>
  class not_null;
}  // namespace gsl

class DataVector;
namespace Particles::MonteCarlo {
  class Packet;
} // namespace Particles::MonteCarlo
/// \endcond

namespace Particles::MonteCarlo {

/// \brief Contribution to the neutrino-matter coupling
/// terms from evolving a packet by time dt, given
/// absorption and scattering opacities.
///
/// \details We calculate total momentum exchanges, not
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

} // namespace Particles::MonteCarlo
