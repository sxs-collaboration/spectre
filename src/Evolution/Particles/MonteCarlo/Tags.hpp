// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"


namespace Particles::MonteCarlo{

struct Packet;

namespace Tags{

// The list of packets living on a given element
struct McPacketsOnElement : db::SimpleTag{
  using type = std::vector<Particles::MonteCarlo::Packet>;
};

// The emissivity of neutrinos within a cell, for a given
// energy bin and species
template <typename DataType, size_t EnergyBins, size_t NeutrinoSpecies>
struct EmissionInCell : db::SimpleTag{
  using type = std::array<std::array<DataType, EnergyBins>, NeutrinoSpecies>;
};


} // namespace Tags
} // namespace Particles::MonteCarlo
