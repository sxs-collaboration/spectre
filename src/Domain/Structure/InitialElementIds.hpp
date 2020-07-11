// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

/// \cond
template <size_t VolumeDim>
class ElementId;
/// \endcond

/// \ingroup ComputationalDomainGroup
/// \brief Create the `ElementId`s of the a single Block
template <size_t VolumeDim>
std::vector<ElementId<VolumeDim>> initial_element_ids(
    size_t block_id, std::array<size_t, VolumeDim> initial_ref_levs) noexcept;

/// \ingroup ComputationalDomainGroup
/// \brief Create the `ElementId`s of the initial computational domain.
template <size_t VolumeDim>
std::vector<ElementId<VolumeDim>> initial_element_ids(
    const std::vector<std::array<size_t, VolumeDim>>&
        initial_refinement_levels) noexcept;
