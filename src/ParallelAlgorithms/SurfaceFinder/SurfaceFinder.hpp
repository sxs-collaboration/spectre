// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

/// \ingroup SurfacesGroup
/// Contains functions that are used to find contour levels.
namespace SurfaceFinder {
/*!
 * \ingroup SurfacesGroup

 * \brief Function that interpolates data onto radial rays in the direction
 * of the logical coordinates \f$\xi\f$ and \f$\eta\f$ and tries to perform a
 * root find of \f$\text{data}-\text{target}\f$. Returns the logical coordinates
 * of the roots found.
 *
 * \details We assume the element is part of a wedge block, so that the
 * \f$\zeta\f$ logical coordinate points in the radial direction. Will fail if
 * multiple roots are along a ray. This could be generalized to domains other
 * than a wedge if necessary by passing in which logical direction points
 * radially.
 *
 * \param data data to find the contour in.
 * \param target target value for the contour level in the data.
 * \param mesh mesh for the element.
 * \param angular_coords tensor containing the \f$\xi\f$ and \f$\eta\f$
 * values of the rays extending into \f$\zeta\f$ direction.
 * \param relative_tolerance relative tolerance for toms748 rootfind.
 * \param absolute_tolerance relative tolerance for toms748 rootfind.
 */
std::vector<std::optional<double>> find_radial_surface(
    const Scalar<DataVector>& data, const double target, const Mesh<3>& mesh,
    const tnsr::I<DataVector, 2, Frame::ElementLogical>& angular_coords,
    const double relative_tolerance = 1e-10,
    const double absolute_tolerance = 1e-10);

}  // namespace SurfaceFinder
