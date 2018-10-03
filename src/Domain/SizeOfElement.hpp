// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t VolumeDim>
class Mesh;
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
template <size_t VolumeDim>
struct Mesh;
}  // namespace Tags
/// \endcond

/*!
 * \ingroup ComputationalDomainGroup
 * \brief Compute the inertial-coordinate size of an element along each of its
 * logical directions.
 *
 * For each logical direction, compute the mean position (in inertial
 * coordinates) of the element's lower and upper faces in that direction.
 * This is done by simply averaging the coordinates of the face grid points.
 * The size of the element along this logical direction is then the distance
 * between the mean positions of the lower and upper faces.
 * Note that for curved elements, this is an approximate measurement of size.
 *
 * \details
 * Because this quantity is defined in terms of specific coordinates, it is
 * not well represented by a `Tensor`, so we use a `std::array`.
 */
template <size_t VolumeDim>
std::array<double, VolumeDim> size_of_element(
    const Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim>& inertial_coords) noexcept;

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The inertial-coordinate size of an element along each of its logical
/// directions.
template <size_t VolumeDim>
struct SizeOfElement : db::ComputeTag {
  static std::string name() noexcept { return "SizeOfElement"; }
  using argument_tags =
      tmpl::list<Tags::Mesh<VolumeDim>,
                 Tags::Coordinates<VolumeDim, Frame::Inertial>>;
  static constexpr auto function = size_of_element<VolumeDim>;
};
}  // namespace Tags
