// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "Domain/Amr/Flag.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

template <size_t VolumeDim>
class Element;

template <size_t VolumeDim>
class ElementId;
/// \endcond

namespace amr {
/// \ingroup ComputationalDomainGroup
/// \brief Updates the AMR decisions `my_current_amr_flags` of the Element
/// `element` based on the AMR decisions `neighbor_amr_flags` of a neighbor
/// Element with ElementId `neighbor_id`.
///
/// \details  This function is called by each element when it receives the AMR
/// decisions of one of its neighbors.  If any of its flags are updated, the
/// element should send its new decisions to each of its neighbors.  The
/// following changes are made to the current flags of the element:
/// - If the neighbor wants to be two or more refinement levels higher than
///   the element, the flag is updated to bring the element to within one level
/// - If the element wants to join, and the neighbor is a potential sibling but
///   wants to be at a different refinement level in any dimension, the flag is
///   updated to not do h-refinement.
///
/// \returns true if any flag is changed
///
/// \note Modifies `my_current_amr_flags` which are the AMR decisions of
/// `element`.
template <size_t VolumeDim>
bool update_amr_decision(
    gsl::not_null<std::array<amr::Flag, VolumeDim>*> my_current_amr_flags,
    const Element<VolumeDim>& element, const ElementId<VolumeDim>& neighbor_id,
    const std::array<amr::Flag, VolumeDim>& neighbor_amr_flags) noexcept;
}  // namespace amr
