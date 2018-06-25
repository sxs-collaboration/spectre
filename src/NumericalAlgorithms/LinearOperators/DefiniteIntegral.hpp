// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function definite_integral.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
template <size_t>
class Mesh;
/// \endcond

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Compute the definite integral of a grid-function over a manifold.
 *
 * The integral is computed on the reference element by multiplying the
 * DataVector with the Spectral::quadrature_weights() in that
 * dimension.
 * \requires number of points in `integrand` and `mesh` are equal.
 * \param integrand the grid function to integrate.
 * \param mesh the Mesh of the manifold on which `integrand` is located.
 * \returns the definite integral of `integrand` on the manifold.
 */
template <size_t Dim>
double definite_integral(const DataVector& integrand,
                         const Mesh<Dim>& mesh) noexcept;
