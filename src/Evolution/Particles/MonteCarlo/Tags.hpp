// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <random>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Particles/MonteCarlo/NeutrinoInteractionTable.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"

/// Items related to the evolution of particles
/// Items related to Monte-Carlo radiation transport
/// Tags for MC
namespace Particles::MonteCarlo::Tags {

/// Simple tag containing the vector of Monte-Carlo
/// packets belonging to an element.
struct PacketsOnElement : db::SimpleTag {
  using type = std::vector<Particles::MonteCarlo::Packet>;
};

/// Simple tag containing an approximation of the light
/// crossing time for each cell (the shortest time among
/// all coordinate axis directions).
template <typename DataType>
struct CellLightCrossingTime : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// Simple tag storing the random number generator
/// used by Monte-Carlo
struct RandomNumberGenerator : db::SimpleTag {
  using type = std::mt19937;
};

/// Simple tag containing the desired energy of
/// packets in low-density regions. The energy
/// can be different for each neutrino species.
template <size_t NeutrinoSpecies>
struct DesiredPacketEnergyAtEmission : db::SimpleTag {
  using type = std::array<DataVector, NeutrinoSpecies>;
};

/// Simple tag for the table of neutrino-matter interaction
/// rates (emission, absorption and scattering for each
/// energy bin and neutrino species).
template <size_t EnergyBins, size_t NeutrinoSpecies>
struct InteractionRatesTable : db::SimpleTag {
  using type =
      std::unique_ptr<NeutrinoInteractionTable<EnergyBins, NeutrinoSpecies>>;
};

}  // namespace Particles::MonteCarlo::Tags
