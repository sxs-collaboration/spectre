// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DomainHelper functions

#pragma once

#include <algorithm>
#include <numeric>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockNeighbor.hpp"
#include "Utilities/ConstantExpressions.hpp"

/// Sets up the BlockNeighbors using the corner numbering scheme
/// to deduce the correct neighbors and orientations. Does not set
/// up periodic boundary conditions.
template <size_t VolumeDim>
void set_internal_boundaries(
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks,
    gsl::not_null<std::vector<
        std::unordered_map<Direction<VolumeDim>, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks);

/// Sets up additional BlockNeighbors corresponding to any
/// periodic boundary condtions provided by the user. These are
/// stored in identifications.
template <size_t VolumeDim>
void set_periodic_boundaries(
    const std::vector<std::vector<size_t>>& identifications,
    const std::vector<std::array<size_t, two_to_the(VolumeDim)>>&
        corners_of_all_blocks,
    gsl::not_null<std::vector<
        std::unordered_map<Direction<VolumeDim>, BlockNeighbor<VolumeDim>>>*>
        neighbors_of_all_blocks);
