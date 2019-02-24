// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
template <size_t>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

// @{
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Compute the indefinite integral of a function in the
 * `dim_to_integrate`. Applying a zero boundary condition on each stripe.
 *
 * \requires number of points in `integrand` and `mesh` are equal.
 */
template <size_t Dim>
void indefinite_integral(gsl::not_null<DataVector*> integral,
                         const DataVector& integrand, const Mesh<Dim>& mesh,
                         size_t dim_to_integrate) noexcept;

template <size_t Dim>
DataVector indefinite_integral(const DataVector& integrand,
                               const Mesh<Dim>& mesh,
                               size_t dim_to_integrate) noexcept;
// @}
