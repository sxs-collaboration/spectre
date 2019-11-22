// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Element.hpp"

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
 * that all of its neighboring elements are at the same refinement level. Thus,
 * the created element's `neighbors` has one neighbor per direction (or zero on
 * an external boundary), with each neighbor at the same refinement level as the
 * element.
 */
template <size_t VolumeDim>
Element<VolumeDim> create_initial_element(
    const ElementId<VolumeDim>& element_id,
    const Block<VolumeDim>& block) noexcept;
}  // namespace Initialization
}  // namespace domain
