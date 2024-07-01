// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
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

}  // namespace Particles::MonteCarlo::Tags
