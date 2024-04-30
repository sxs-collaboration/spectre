// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

/// @{
/*!
 * \brief Convert between Cartesian and spherical coordinates (r, theta, phi)
 *
 * In 2D the order is (r, phi), and in 3D the order is (r, theta, phi), as
 * defined by `Frame::Spherical`. The conventions for the angles are as follows:
 *
 * - phi is the azimuthal angle (-pi, pi], measuring the angle from the x-axis
 * - theta is the polar angle [0, pi], measuring the angle from the z-axis
 */
template <typename DataType, size_t Dim, typename CoordsFrame>
void cartesian_to_spherical(
    gsl::not_null<tnsr::I<DataType, Dim, Frame::Spherical<CoordsFrame>>*>
        result,
    const tnsr::I<DataType, Dim, CoordsFrame>& x);

template <typename DataType, size_t Dim, typename CoordsFrame>
tnsr::I<DataType, Dim, Frame::Spherical<CoordsFrame>> cartesian_to_spherical(
    const tnsr::I<DataType, Dim, CoordsFrame>& x);

template <typename DataType, size_t Dim, typename CoordsFrame>
void spherical_to_cartesian(
    gsl::not_null<tnsr::I<DataType, Dim, CoordsFrame>*> result,
    const tnsr::I<DataType, Dim, Frame::Spherical<CoordsFrame>>& x);

template <typename DataType, size_t Dim, typename CoordsFrame>
tnsr::I<DataType, Dim, CoordsFrame> spherical_to_cartesian(
    const tnsr::I<DataType, Dim, Frame::Spherical<CoordsFrame>>& x);
/// @}
