// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
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

namespace Tags {

/// Simple tag containing the fluid and metric data in the ghost zones
/// for Monte-Carlo packets evolution.
template <size_t Dim>
struct MortarDataTag : db::SimpleTag {
  using type = MortarData<Dim>;
};

}  // namespace Tags

}  // namespace Particles::MonteCarlo
