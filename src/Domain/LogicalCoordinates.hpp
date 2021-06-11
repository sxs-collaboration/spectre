// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions logical_coordinates and interface_logical_coordinates

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace Tags {
template<size_t Dim>
struct Mesh;
template <size_t, typename>
struct Coordinates;  // IWYU pragma: keep
}  // namespace Tags
}  // namespace domain
template <size_t Dim>
class Mesh;
class DataVector;
template <size_t Dim>
class Direction;

namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

/// @{
/*!
 * \ingroup ComputationalDomainGroup
 * \brief Compute the logical coordinates in an Element.
 *
 * \details The logical coordinates are the collocation points associated to the
 * spectral basis functions and quadrature of the \p mesh.
 *
 * \example
 * \snippet Test_LogicalCoordinates.cpp logical_coordinates_example
 */
template <size_t VolumeDim>
void logical_coordinates(
    gsl::not_null<tnsr::I<DataVector, VolumeDim, Frame::Logical>*>
        logical_coords,
    const Mesh<VolumeDim>& mesh) noexcept;

template <size_t VolumeDim>
tnsr::I<DataVector, VolumeDim, Frame::Logical> logical_coordinates(
    const Mesh<VolumeDim>& mesh) noexcept;
/// @}

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

namespace domain {
namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The logical coordinates in the Element
template <size_t VolumeDim>
struct LogicalCoordinates : Coordinates<VolumeDim, Frame::Logical>,
                            db::ComputeTag {
  using base = Coordinates<VolumeDim, Frame::Logical>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<Mesh<VolumeDim>>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<return_type*>, const ::Mesh<VolumeDim>&) noexcept>(
      &logical_coordinates<VolumeDim>);
};
}  // namespace Tags
}  // namespace domain
