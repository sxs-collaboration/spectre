// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

/// \ingroup EvolutionSystemsGroup
/// \brief Items related to general relativistic radiation transport
namespace RadiationTransport::NoNeutrinos {
/// No neutrino placeholder
struct System {
  static std::string name() { return "NoNeutrinos"; }
};
}  // namespace RadiationTransport::NoNeutrinos
