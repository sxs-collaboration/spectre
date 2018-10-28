// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function linearize.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
template <size_t>
class Mesh;
namespace gsl {
template <class>
class not_null;
}  // namespace gsl
/// \endcond

// @{
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Truncate u to a linear function in each dimension.
 *
 * Ex in 2D: \f$u^{Lin} = U_0 + U_x x + U_y y + U_{xy} xy\f$
 *
 * \warning the `gsl::not_null` variant assumes `*result` is of the correct
 * size.
 */
template <size_t Dim>
void linearize(gsl::not_null<DataVector*> result, const DataVector& u,
               const Mesh<Dim>& mesh) noexcept;
template <size_t Dim>
DataVector linearize(const DataVector& u, const Mesh<Dim>& mesh) noexcept;
// @}

// @{
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Truncate u to a linear function in the given dimension.
 *
 * **Parameters**
 * - `u` the function to linearize.
 * - `mesh` the Mesh of the grid on the manifold on which `u` is
 * located.
 * - `d` the dimension that is to be linearized.
 *
 * \warning the `gsl::not_null` variant assumes `*result` is of the correct
 * size.
 */
template <size_t Dim>
void linearize(gsl::not_null<DataVector*> result, const DataVector& u,
               const Mesh<Dim>& mesh, size_t d) noexcept;
template <size_t Dim>
DataVector linearize(const DataVector& u, const Mesh<Dim>& mesh,
                     size_t d) noexcept;
// @}
