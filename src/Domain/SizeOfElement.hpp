// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim, typename Frame>
class ElementMap;
namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct ElementMap;
}  // namespace Tags
}  // namespace domain
/// \endcond

/*!
 * \ingroup ComputationalDomainGroup
 * \brief Compute the inertial-coordinate size of an element along each of its
 * logical directions.
 *
 * For each logical direction, compute the distance (in inertial coordinates)
 * between the element's lower and upper faces in that logical direction.
 * The distance is measured between centers of the faces, with the centers
 * defined in the logical coordinates.
 * Note that for curved elements, this is an approximate measurement of size.
 *
 * \details
 * Because this quantity is defined in terms of specific coordinates, it is
 * not well represented by a `Tensor`, so we use a `std::array`.
 */
template <size_t VolumeDim>
std::array<double, VolumeDim> size_of_element(
    const ElementMap<VolumeDim, Frame::Inertial>& element_map) noexcept;

namespace domain {
namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The inertial-coordinate size of an element along each of its logical
/// directions.
template <size_t VolumeDim>
struct SizeOfElement : db::ComputeTag {
  static std::string name() noexcept { return "SizeOfElement"; }
  using argument_tags =
      tmpl::list<Tags::ElementMap<VolumeDim, Frame::Inertial>>;
  static constexpr auto function = size_of_element<VolumeDim>;
};
}  // namespace Tags
}  // namespace domain
