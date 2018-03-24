// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares function unnormalized_face_normal

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename, typename, size_t>
class CoordinateMapBase;
class DataVector;
template <size_t>
class Direction;
template <size_t Dim, typename Frame>
class ElementMap;
template <size_t>
class Index;
/// \endcond

namespace Tags {
template <size_t>
struct Direction;
template <size_t, typename>
struct ElementMap;
template <size_t>
struct Extents;
}  // namespace Tags
/// \endcond

// @{
/*!
 * \ingroup ComputationalDomainGroup
 * \brief Compute the outward grid normal on a face of an Element
 *
 * \returns outward grid-frame one-form holding the normal
 *
 * \details
 * Computes the grid-frame normal by taking the logical-frame unit
 * one-form in the given Direction and mapping it to the grid frame
 * with the given map.
 *
 * \example
 * \snippet Test_FaceNormal.cpp face_normal_example
 */
template <size_t VolumeDim, typename TargetFrame>
tnsr::i<DataVector, VolumeDim, TargetFrame> unnormalized_face_normal(
    const Index<VolumeDim - 1>& interface_extents,
    const ElementMap<VolumeDim, TargetFrame>& map,
    const Direction<VolumeDim>& direction) noexcept;

template <size_t VolumeDim, typename TargetFrame>
tnsr::i<DataVector, VolumeDim, TargetFrame> unnormalized_face_normal(
    const Index<VolumeDim - 1>& interface_extents,
    const CoordinateMapBase<Frame::Logical, TargetFrame, VolumeDim>& map,
    const Direction<VolumeDim>& direction) noexcept;
// @}

namespace Tags {
/// \ingroup DataBoxTags
/// \ingroup ComputationalDomain
/// The unnormalized face normal one form
template <size_t VolumeDim, typename Frame = ::Frame::Inertial>
struct UnnormalizedFaceNormal : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "UnnormalizedFaceNormal";
  static constexpr tnsr::i<DataVector, VolumeDim, Frame> (*function)(
      const ::Index<VolumeDim - 1>&, const ::ElementMap<VolumeDim, Frame>&,
      const ::Direction<VolumeDim>&) = unnormalized_face_normal;
  using argument_tags = tmpl::list<
    Extents<VolumeDim - 1>, ElementMap<VolumeDim, Frame>, Direction<VolumeDim>>;
  using volume_tags = tmpl::list<ElementMap<VolumeDim, Frame>>;
};
}  // namespace Tags
