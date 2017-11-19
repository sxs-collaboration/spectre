// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <vector>

template <size_t VolumeDim>
class ElementId;

/// \ingroup ComputationalDomainGroup
/// \brief Create the ElementIds of the initial computational domain.
template <size_t VolumeDim>
std::vector<ElementId<VolumeDim>> initial_element_ids(
    const std::vector<std::array<size_t, VolumeDim>>&
        initial_refinement_levels) noexcept;
