// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function definite_integral.

#pragma once

#include <cstddef>

class DataVector;
template <size_t>
class Index;

namespace Basis {
namespace lgl {

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Compute the definite integral of a grid-function over a manifold.
 *
 * The integral is computed on the reference element by multiplying the
 * DataVector with the Basis::lgl integration weights in that
 * dimension.
 * \requires number of points in `integrand` and `extents` are equal.
 * \param integrand the grid function to integrate.
 * \param extents the extents of the manifold on which `integrand` is located.
 * \returns the definite integral of `integrand` on the manifold.
 */
template <size_t Dim>
double definite_integral(const DataVector& integrand,
                         const Index<Dim>& extents) noexcept;

}  // namespace lgl
}  // namespace Basis
