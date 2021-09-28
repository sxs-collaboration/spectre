// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim>
class Mesh;
class ComplexDataVector;
/// \endcond

namespace Cce {
/*!
 * \brief Computes the partial derivative along a particular direction
 * determined by the `dimension_to_differentiate`.
 * The input `u` is differentiated with the spectral matrix and the solution is
 * placed in `d_u`.
 *
 * \note This is placed in Cce Utilities for its currently narrow use-case. If
 * more general uses desire a single partial derivative of complex values, this
 * should be moved to `NumericalAlgorithms`. This utility currently assumes the
 * spatial dimensionality is 3, which would also need to be generalized, likely
 * by creating a wrapping struct with partial template specializations.
 */
void logical_partial_directional_derivative_of_complex(
    gsl::not_null<ComplexDataVector*> d_u, const ComplexDataVector& u,
    const Mesh<3>& mesh, size_t dimension_to_differentiate);
}  // namespace Cce
