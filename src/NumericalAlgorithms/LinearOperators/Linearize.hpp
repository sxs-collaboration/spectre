// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function linearize.

#pragma once

#include <cstddef>

class DataVector;
namespace domain {
template <size_t>
class Mesh;
}  // namespace domain

/*!
 * \ingroup NumericalAlgorithmsGroup
 * Truncate u to a linear function in each dimension.
 * Ex in 2D: \f$u^{Lin} = U_0 + U_x x + U_y y + U_{xy} xy\f$
 *
 * \returns the linearization of `u`
 */
template <size_t Dim>
DataVector linearize(const DataVector& u, const domain::Mesh<Dim>& mesh);

/*!
 * \ingroup NumericalAlgorithmsGroup
 * Truncate u to a linear function in the given dimension.
 *
 * \returns the linearization of `u`
 * \param u the function to linearize.
 * \param mesh the Mesh of the grid on the manifold on which `u` is
 * located.
 * \param d the dimension that is to be linearized.
 */
template <size_t Dim>
DataVector linearize(const DataVector& u, const domain::Mesh<Dim>& mesh,
                     size_t d);
