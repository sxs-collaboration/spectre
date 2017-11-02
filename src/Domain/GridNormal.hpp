// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares function unnormalized_grid_normal

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"

template <typename, typename, size_t>
class CoordinateMapBase;
class DataVector;
template <size_t>
class Direction;
template <size_t>
class Index;

/*!
 * \ingroup ComputationalDomain
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
 * \snippet Test_GridNormal.cpp grid_normal_example
 */
template <size_t VolumeDim>
tnsr::i<DataVector, VolumeDim, Frame::Grid> unnormalized_grid_normal(
    const Index<VolumeDim - 1>& interface_extents,
    const CoordinateMapBase<Frame::Logical, Frame::Grid, VolumeDim>& map,
    const Direction<VolumeDim>& direction) noexcept;
