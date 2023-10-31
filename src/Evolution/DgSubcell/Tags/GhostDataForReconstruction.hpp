// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionId.hpp"
#include "Domain/Structure/DirectionIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"

namespace evolution::dg::subcell::Tags {
/// The ghost data used for reconstructing the solution on the interfaces
/// between elements
///
/// The `DirectionIdMap` stores a `evolution::dg::subcell::GhostData` which
/// stores both local and received neighbor ghost data. Even though subcell
/// doesn't use it's local ghost data during reconstruction, we will need to
/// store it when using Charm++ messages.
template <size_t Dim>
struct GhostDataForReconstruction : db::SimpleTag {
  using type = DirectionIdMap<Dim, GhostData>;
};
}  // namespace evolution::dg::subcell::Tags
