// Distributed under the MIT License.
// See LICENSE.txt for details.

/// Defines functions interface_logical_coordinates

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
template <size_t Dim>
class Mesh;
class DataVector;
template <size_t Dim>
class Direction;
/// \endcond

/*!
 * \ingroup ComputationalDomainGroup
 * \brief Compute the logical coordinates on a face of an Element.
 *
 * \returns element logical-frame vector holding coordinates
 *
 * \example
 * \snippet Test_InterfaceLogicalCoordinates.cpp interface_logical_coordinates_example
 */
template <size_t VolumeDim>
tnsr::I<DataVector, VolumeDim, Frame::ElementLogical>
interface_logical_coordinates(const Mesh<VolumeDim - 1>& mesh,
                              const Direction<VolumeDim>& direction);
