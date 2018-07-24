// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Matrix.hpp"  // IWYU pragma: keep

/// \cond
template <size_t Dim>
class Mesh;
/// \endcond

namespace Spectral {

/// The portion of an element covered by a mortar.
enum class MortarSize { Full, UpperHalf, LowerHalf };

/*!
 * \brief The projection matrix from a 1D mortar to an element.
 *
 * \details
 * The projection matrices returned by this function (and by
 * projection_matrix_element_to_mortar()) define orthogonal projection
 * operators between the spaces of functions on the boundary of the
 * element and the mortar.  These projections are usually the correct
 * way to transfer data onto and off of the mortars.
 *
 * The half-interval projections are based on an equation derived by
 * Saul.  This shows that the projection from the spectral basis for
 * the entire interval to the spectral basis for the upper half
 * interval is
 * \f{equation*}
 * T_{jk} = \frac{2 j + 1}{2} 2^j \sum_{n=0}^{j-k} \binom{j}{k+n}
 * \binom{(j + k + n - 1)/2}{j} \frac{(k + n)!^2}{(2 k + n + 1)! n!}
 * \f}
 */
const Matrix& projection_matrix_mortar_to_element(
    MortarSize size, const Mesh<1>& element_mesh,
    const Mesh<1>& mortar_mesh) noexcept;

/// The projection matrix from a 1D element to a mortar.  See
/// projection_matrix_mortar_to_element() for details.
const Matrix& projection_matrix_element_to_mortar(
    MortarSize size, const Mesh<1>& mortar_mesh,
    const Mesh<1>& element_mesh) noexcept;

}  // namespace Spectral
