// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>

#include "Domain/Amr/Flag.hpp"

/// \cond
template <size_t VolumeDim>
class Direction;
template <size_t VolumeDim>
class ElementId;
template <size_t VolumeDim>
class Neighbors;
/// \endcond

namespace amr {
/// \ingroup AmrGroup
/// \brief returns the ElementId%s of the neighbors in the given `direction` of
/// the Element whose ElementId is `my_id` given the
/// `previous_neighbors_in_direction` and their amr::Flag%s.
///
/// \note `previous_neighbors_in_direction` should be from the parent (or a
/// child) of the Element with `my_id` if `my_id` corresponds to a newly created
/// child (or parent) Element.
///
/// \note `previous_neighbors_amr_flags` may contain flags from neighbors in
/// directions other than `direction`
template <size_t VolumeDim>
std::unordered_set<ElementId<VolumeDim>> new_neighbor_ids(
    const ElementId<VolumeDim>& my_id, const Direction<VolumeDim>& direction,
    const Neighbors<VolumeDim>& previous_neighbors_in_direction,
    const std::unordered_map<ElementId<VolumeDim>,
                             std::array<amr::Flag, VolumeDim>>&
        previous_neighbors_amr_flags);
}  // namespace amr
