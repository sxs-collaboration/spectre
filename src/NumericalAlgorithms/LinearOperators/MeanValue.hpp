// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function mean_value and mean_value_on_boundary.

#pragma once

#include <cstddef>

#include "Domain/Side.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "Utilities/ConstantExpressions.hpp"

/// \cond
class DataVector;
namespace domain {
template <size_t>
class Mesh;
}  // namespace domain
/// \endcond

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Compute the mean value of a grid-function over a manifold.
 * \f$mean value = \int f dV / \int dV\f$
 *
 * \remarks The mean value is computed on the reference element(s).
 * \note The mean w.r.t. a different set of coordinates x can be computed
 * by pre-multiplying the argument f by the Jacobian J = dx/dxi of the mapping
 * from the reference coordinates xi to the coordinates x.
 *
 * \returns the mean value of `f` on the manifold
 * \param f the grid function of which to find the mean.
 * \param mesh the Mesh of the manifold on which f is located.
 */
template <size_t Dim>
double mean_value(const DataVector& f, const domain::Mesh<Dim>& mesh) {
  return definite_integral(f, mesh) / two_to_the(Dim);
}

/*!
 * \ingroup NumericalAlgorithmsGroup
 * Compute the mean value of a grid-function on a boundary of a manifold.
 * \f$mean value = \int f dV / \int dV\f$
 *
 * \remarks The mean value is computed on the reference element(s).
 *
 * \returns the mean value of `f` on the boundary of the manifold
 *
 * \param f the grid function of which to find the mean.
 * \param mesh the Mesh of the manifold on which f is located.
 * \param d the dimension which is sliced away to get the boundary.
 * \param side whether it is the lower or upper boundary in the d-th dimension.
 */
template <size_t Dim>
double mean_value_on_boundary(const DataVector& f,
                              const domain::Mesh<Dim>& mesh, size_t d,
                              domain::Side side);
