// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions logical_coordinates and interface_logical_coordinates

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
template<size_t Dim>
struct Mesh;
template <size_t, typename>
struct Coordinates;  // IWYU pragma: keep
}  // namespace Tags
template <size_t Dim>
class Mesh;
class DataVector;
template <size_t Dim>
class Direction;
/// \endcond

/*!
 * \ingroup ComputationalDomainGroup
 * \brief Compute the logical coordinates in an Element.
 *
 * \details The logical coordinates are the collocation points associated to the
 * spectral basis functions and quadrature of the \p mesh.
 *
 * \returns logical-frame vector holding coordinates
 *
 * \example
 * \snippet Test_LogicalCoordinates.cpp logical_coordinates_example
 */
template <size_t VolumeDim>
tnsr::I<DataVector, VolumeDim, Frame::Logical> logical_coordinates(
    const Mesh<VolumeDim>& mesh) noexcept;

/*!
 * \ingroup ComputationalDomainGroup
 * \brief Compute the logical coordinates on a face of an Element.
 *
 * \returns logical-frame vector holding coordinates
 *
 * \example
 * \snippet Test_LogicalCoordinates.cpp interface_logical_coordinates_example
 */
template <size_t VolumeDim>
tnsr::I<DataVector, VolumeDim, Frame::Logical> interface_logical_coordinates(
    const Mesh<VolumeDim - 1>& mesh,
    const Direction<VolumeDim>& direction) noexcept;

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The logical coordinates in the Element
template <size_t VolumeDim>
struct LogicalCoordinates : Coordinates<VolumeDim, Frame::Logical>,
                            db::ComputeTag {
  using argument_tags = tmpl::list<Tags::Mesh<VolumeDim>>;
  static constexpr auto function = logical_coordinates<VolumeDim>;
};
}  // namespace Tags
