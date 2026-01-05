// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"

namespace evolution::dg::subcell::Tags {
/// Inverse Jacobian data stored in the ghost zone.
///
/// The `DirectionalIdMap` stores the grid coordinates and logical to grid
/// inverse Jacobians in the ghost zone to avoid recomputing these data when
/// full Jacobians are needed. We store the grid coordinates to be able to
/// compute the grid to inertial Jacobian at each time step.
template <size_t Dim>
struct GhostZoneInverseJacobian : db::SimpleTag {
  using type = DirectionMap<
      Dim, Variables<tmpl::list<Coordinates<Dim, Frame::Grid>,
                                evolution::dg::subcell::fd::Tags::
                                    InverseJacobianLogicalToGrid<Dim>>>>;
};
}  // namespace evolution::dg::subcell::Tags
