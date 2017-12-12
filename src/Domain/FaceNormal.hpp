// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares function unnormalized_face_normal

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"

template <typename, typename, size_t>
class CoordinateMapBase;
class DataVector;
template <size_t>
class Direction;
template <size_t Dim, typename Frame>
class ElementMap;
template <size_t>
class Index;

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
