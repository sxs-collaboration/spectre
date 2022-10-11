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

namespace fd {
/*!
 * \brief Compute the logical partial derivatives using cell-centered finite
 * difference derivatives.
 *
 * Up to 8th order stencils are supported.
 *
 * \note Currently the stride is always one because we transpose the data before
 * reconstruction. However, it may be faster to have a non-unit stride without
 * the transpose. We have the `stride` parameter in the derivative stencils
 * to make testing performance easier in the future.
 *
 * \note This code does not do any explicit SIMD vectorization. We will want to
 * profile and decide if optimization are possible. The Vc SIMD library has an
 * example of vectorizing single-precision FD derivatives. There is also a paper
 * "Optimization of Finite-Differencing Kernels for Numerical Relativity
 * Applications" by Alfieri, Bernuzzi, Perego, and Radice that uses compiler
 * auto-vectorization.
 */
template <size_t Dim>
void logical_partial_derivatives(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        logical_derivatives,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, size_t number_of_variables, size_t fd_order);

/*!
 * \brief Compute the partial derivative on the `DerivativeFrame` using the
 * `inverse_jacobian`.
 *
 * Logical partial derivatives are first computed using the
 * `fd::logical_partial_derivatives()` function.
 */
template <typename DerivativeTags, size_t Dim, typename DerivativeFrame>
void partial_derivatives(
    gsl::not_null<Variables<db::wrap_tags_in<
        Tags::deriv, DerivativeTags, tmpl::size_t<Dim>, DerivativeFrame>>*>
        partial_derivatives,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, size_t number_of_variables, size_t fd_order,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian);
}  // namespace fd
