// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "Domain/Structure/Element.hpp"

/// \cond
template <size_t Dim>
class Block;
template <size_t Dim>
class ElementId;
/// \endcond

namespace domain {
namespace Initialization {
/*!
 * \ingroup InitializationGroup
 * \brief Creates an initial element of a Block.
 *
 * \details This function creates an element at the refinement level and
 * position specified by the `element_id` within the `block`. It assumes
 * that all elements in a given block have the same refinement level,
 * given in `initial_refinement_levels`.
 */
template <size_t VolumeDim>
Element<VolumeDim> create_initial_element(
    const ElementId<VolumeDim>& element_id, const Block<VolumeDim>& block,
    const std::vector<std::array<size_t, VolumeDim>>&
        initial_refinement_levels) noexcept;
}  // namespace Initialization
}  // namespace domain
