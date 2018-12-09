// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

/// Namespace for neutrino physics
namespace neutrinos {
template <size_t EnergyBin>
struct ElectronNeutrinos {
  static constexpr size_t energy_bin = EnergyBin;
};
template <size_t EnergyBin>
struct ElectronAntiNeutrinos {
  static constexpr size_t energy_bin = EnergyBin;
};
template <size_t EnergyBin>
struct HeavyLeptonNeutrinos {
  static constexpr size_t energy_bin = EnergyBin;
};

template <template <size_t> class U, size_t EnergyBin>
std::string get_name(const U<EnergyBin>& /*species*/) noexcept {
  return pretty_type::short_name<U<EnergyBin>>() + std::to_string(EnergyBin);
}

}  // namespace neutrinos
