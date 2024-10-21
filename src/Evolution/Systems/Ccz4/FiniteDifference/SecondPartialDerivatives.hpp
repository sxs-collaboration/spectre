// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
template <size_t Dim, typename T>
class DirectionMap;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
/// \endcond

namespace Ccz4::fd {
/*!
 * \brief Compute the pure and mixed second logical partial derivatives using
 * finite difference derivatives. Only 3 dimensions 4th-order fd is supported.
 *
 * We use a symmetric stencil in the bulk and slightly asymetric stencil
 * for the mixed partial derivatives where ghost data from two
 * neighbors is needed (to avoid communication across diagonal elements).
 * The `mixed_second_logical_derivatives` return an 3D array corresponding to
 * xy, yz, xz mixed partial derivatives.
 *
 * The stencil for the pure second derivatives is
 \f[ \frac{\partial^2 f}{\partial x^2} =
    - \frac{1}{12\delta^2}[f(x+2\delta,y)+f(x-2\delta,y)]
    + \frac{4}{3\delta^2}[f(x+\delta,y)+f(x-\delta,y)]-\frac{5}{2\delta^2}f(x,y)
 + O(\delta^4). \f]
 *
 * The stencil for the mixed second derivatives in the bulk is
 \f[ \begin{aligned} \frac{\partial^2 f}{\partial x \partial y} =&
      \frac{1}{3\delta^2}[f(x+\delta,y+\delta) + f(x-\delta,y-\delta) -
 f(x+\delta,y-\delta) - f(x-\delta,y+\delta)] \\
    &+ \frac{1}{48\delta^2}[f(x+2\delta,y-2\delta) + f(x-2\delta,y+2\delta)
        - f(x+2\delta,y+2\delta) - f(x-2\delta,y-2\delta)] + O(\delta^4).
 \end{aligned}\f]
 *
 * The stencil for the mixed second derivatives in the ghost zone is (with
 * descending diagonal)
 \f[ \begin{aligned} \frac{\partial^2 f}{\partial x \partial y} =&
      \frac{1}{24\delta^2}[f(x+2\delta,y-2\delta) + f(x-2\delta,y+2\delta)
        - f(x+2\delta,y) - f(x-2\delta,y) - f(x,y+2\delta) - f(x,y-2\delta)] \\
    &+ \frac{2}{3\delta^2}[f(x+\delta,y) + f(x-\delta,y) + f(x,y+\delta) +
        f(x,y-\delta) - f(x+\delta,y-\delta) - f(x-\delta,y+\delta)]
    - \frac{5}{4\delta^2}f(x,y) + O(\delta^4),
 \end{aligned}\f]
 *
 * and the ascending diagonal stencil has the opposite weights.
 *
 * \note Currently the stride is always one because we transpose the data before
 * reconstruction. Only 3D 4-th order second derivatives are implemented.
 */
template <size_t Dim>
void second_logical_partial_derivatives(
    gsl::not_null<std::array<gsl::span<double>, Dim>*>
        pure_second_logical_derivatives,
    gsl::not_null<std::array<gsl::span<double>, Dim>*>
        mixed_second_logical_derivatives,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, size_t number_of_variables, size_t fd_order);

/*!
 * \brief Compute the second partial derivatives on the `DerivativeFrame` using
 * the `inverse_jacobian` and `inverse_hessian`.
 * Only 3 dimensions 4th-order fd is supported.
 *
 * First and second logical partial derivatives are first computed using the
 * `fd::logical_partial_derivatives()` and
 * `Ccz4::fd::second_logical_partial_derivatives()` function.
 *
 * \note The `inverse_hessian` has not been implemented, so currently inertial
 * coordinates cannot mix logical coordinates.
 */
template <typename DerivativeTags, size_t Dim, typename DerivativeFrame>
void second_partial_derivatives(
    gsl::not_null<
        Variables<db::wrap_tags_in<::Tags::second_deriv, DerivativeTags,
                                   tmpl::size_t<Dim>, DerivativeFrame>>*>
        second_partial_derivatives,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, size_t number_of_variables,
    size_t fd_order,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian);
}  // namespace Ccz4::fd
