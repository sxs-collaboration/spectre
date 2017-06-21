
/// \file
/// Defines function definite_integral.

#pragma once

#include <cstddef>

class DataVector;
template <size_t>
class Index;

namespace Basis {
namespace lgl {

/*! \ingroup NumericalAlgorithms
 * \brief Compute the definite integral of a grid-function over a manifold.
 * The integral is computed on the reference element by multiplying the
 * DataVector with the Basis::lgl integration weights in that
 * dimension.
 * \requires number of points in `f` and `extents` are equal.
 * \param f the grid function to integrate.
 * \param extents the extents of the manifold on which `f` is located.
 * \returns the definite integral of `f` on the manifold.
 */
template <size_t Dim>
double definite_integral(const DataVector& f, const Index<Dim>& extents) noexcept;

}  // namespace lgl
}  // namespace Basis
