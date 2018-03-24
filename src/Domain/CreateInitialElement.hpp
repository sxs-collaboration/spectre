// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Element.hpp"

/// \cond
template <size_t Dim, typename TargetFrame>
class Block;
template <size_t Dim>
class ElementId;
/// \endcond

/*!
 * \ingroup ComputationalDomainGroup
 * \brief Creates an initial element of a Block.
 */
template <size_t VolumeDim, typename TargetFrame>
Element<VolumeDim> create_initial_element(
    const ElementId<VolumeDim>& element_id,
    const Block<VolumeDim, TargetFrame>& block) noexcept;
