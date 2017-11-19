// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions logical_coordinates and interface_logical_coordinates

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

template<size_t Dim>
class Index;
class DataVector;
template<size_t Dim>
class Direction;

/*!
 * \ingroup ComputationalDomainGroup
 * \brief Compute the Legendre-Gauss-Lobatto coordinates in an Element
 *
 * \returns logical-frame vector holding coordinates
 *
 * \example
 * \snippet Test_LogicalCoordinates.cpp logical_coordinates_example
 */
template <size_t VolumeDim>
tnsr::I<DataVector, VolumeDim, Frame::Logical> logical_coordinates(
    const Index<VolumeDim>& extents);

/*!
 * \ingroup ComputationalDomainGroup
 * \brief Compute the logical coordinates on a face of an Element
 *
 * \returns logical-frame vector holding coordinates
 *
 * \example
 * \snippet Test_LogicalCoordinates.cpp interface_logical_coordinates_example
 */
template <size_t VolumeDim>
tnsr::I<DataVector, VolumeDim, Frame::Logical> interface_logical_coordinates(
    const Index<VolumeDim - 1>& extents, const Direction<VolumeDim>& direction);
