// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Particles::MonteCarlo {

/// Structure used to gather ghost zone data for Monte-Carlo evolution.
/// We need the rest mass density, electron fraction, temperature, and
/// an estimate of the light-crossing time one cell deep within each
/// neighboring element.
template <size_t Dim>
struct MortarData {
  DirectionalIdMap<Dim, std::optional<DataVector>> rest_mass_density{};
  DirectionalIdMap<Dim, std::optional<DataVector>> electron_fraction{};
  DirectionalIdMap<Dim, std::optional<DataVector>> temperature{};
  DirectionalIdMap<Dim, std::optional<DataVector>> cell_light_crossing_time{};

  void pup(PUP::er& p) {
    p | rest_mass_density;
    p | electron_fraction;
    p | temperature;
    p | cell_light_crossing_time;
  }
};

/// Structure used to gather fluid coupling data for Monte-Carlo evolution.
/// We need the energy, momentum, and composition coupling
template <size_t Dim>
struct GhostZoneCouplingData {
  DirectionalIdMap<Dim, std::optional<DataVector>> coupling_tilde_tau{};
  DirectionalIdMap<Dim, std::optional<DataVector>> coupling_tilde_rho_ye{};
  DirectionalIdMap<Dim,
                   std::optional<tnsr::i<DataVector, Dim, Frame::Inertial>>>
      coupling_tilde_s{};

  void pup(PUP::er& p) {
    p | coupling_tilde_tau;
    p | coupling_tilde_rho_ye;
    p | coupling_tilde_s;
  }
};

namespace Tags {

/// Simple tag containing the fluid and metric data in the ghost zones
/// for Monte-Carlo packets evolution.
template <size_t Dim>
struct MortarDataTag : db::SimpleTag {
  using type = MortarData<Dim>;
};

/// Simple tag containing the coupling data in the ghost zones
/// for Monte-Carlo packets evolution.
template <size_t Dim>
struct GhostZoneCouplingDataTag : db::SimpleTag {
  using type = GhostZoneCouplingData<Dim>;
};

}  // namespace Tags

}  // namespace Particles::MonteCarlo
