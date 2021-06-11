// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
template <size_t>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

/// @{
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Compute the indefinite integral of a function in the
 * `dim_to_integrate`, applying a zero boundary condition on each stripe.
 *
 * Integrates with respect to one of the logical coordinates
 * \f$\boldsymbol{\xi} = (\xi, \eta, \zeta)\f$.
 *
 * The integral w.r.t. a different set of coordinates
 * \f$\boldsymbol{x} = \boldsymbol{x}(\boldsymbol{\xi})\f$ can be computed
 * by pre-multiplying `integrand` by the Jacobian determinant
 * \f$J = \det d\boldsymbol{x}/d\boldsymbol{\xi}\f$ of the mapping
 * \f$\boldsymbol{x}(\boldsymbol{\xi})\f$. The integration is still performed
 * along one logical-coordinate direction, indicated by `dim_to_integrate`.
 *
 * \requires number of points in `integrand` and `mesh` are equal.
 */
template <size_t Dim, typename VectorType>
void indefinite_integral(gsl::not_null<VectorType*> integral,
                         const VectorType& integrand, const Mesh<Dim>& mesh,
                         size_t dim_to_integrate) noexcept;

template <size_t Dim, typename VectorType>
VectorType indefinite_integral(const VectorType& integrand,
                               const Mesh<Dim>& mesh,
                               size_t dim_to_integrate) noexcept;
/// @}
