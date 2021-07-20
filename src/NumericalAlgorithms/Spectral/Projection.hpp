// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <ostream>

#include "DataStructures/Matrix.hpp"

/// \cond
template <size_t Dim>
class Mesh;
/// \endcond

namespace Spectral {

/// The portion of a mesh covered by a child mesh.
enum class ChildSize { Full, UpperHalf, LowerHalf };

/// The portion of an element covered by a mortar.
using MortarSize = ChildSize;

std::ostream& operator<<(std::ostream& os, ChildSize mortar_size) noexcept;

/// Determine whether data needs to be projected between a child mesh and its
/// parent mesh. If no projection is necessary the data may be used as-is.
/// Projection is necessary if the child is either p-refined or h-refined
/// relative to its parent, or both. This operation is symmetric, i.e. it is
/// irrelevant in which order the child and the parent mesh are passed in.
template <size_t Dim>
bool needs_projection(const Mesh<Dim>& mesh1, const Mesh<Dim>& mesh2,
                      const std::array<ChildSize, Dim>& child_sizes) noexcept;

/*!
 * \brief The projection matrix from a child mesh to its parent.
 *
 * The projection matrices returned by this function (and by
 * projection_matrix_parent_to_child()) define orthogonal projection operators
 * between the spaces of functions on a parent mesh and its children. These
 * projections are usually the correct way to transfer data between meshes in
 * a mesh-refinement hierarchy, as well as between an element face and its
 * adjacent mortars.
 *
 * These functions assume that the `child_mesh` is at least as fine as the
 * `parent_mesh`, i.e. functions on the `parent_mesh` can be represented exactly
 * on the `child_mesh`. In practice this means that functions can be projected
 * to a mortar (the `child_mesh`) from both adjacent element faces (the
 * `parent_mesh`) without losing accuracy. Similarly, functions in a
 * mesh-refinement hierarchy don't lose accuracy when an element is split
 * (h-refined). For this reason, the `projection_matrix_child_to_parent` is
 * sometimes referred to as a "restriction operator" and the
 * `projection_matrix_parent_to_child` as a "prolongation operator".
 *
 * \par Massive quantities
 * If the quantity that should be projected is not a function over the
 * computational grid but a "massive" residual, i.e. a quantity
 * \f$\int_{\Omega_k} f(x) \psi_p(x) \mathrm{d}V\f$ where \f$\psi_p\f$ are the
 * basis functions on the mesh, then pass `true` for the parameter
 * `operand_is_massive` (default is `false`). The restriction operator for this
 * case is just the transpose of the prolongation operator, i.e. just an
 * interpolation matrix transpose. Note that the "massive" residual already
 * takes the difference in element size between parent and children into account
 * by including a Jacobian in the volume element of the integral.
 *
 * \par Implementation details
 * The half-interval projections are based on an equation derived by
 * Saul.  This shows that the projection from the spectral basis for
 * the entire interval to the spectral basis for the upper half
 * interval is
 * \f{equation*}
 * T_{jk} = \frac{2 j + 1}{2} 2^j \sum_{n=0}^{j-k} \binom{j}{k+n}
 * \binom{(j + k + n - 1)/2}{j} \frac{(k + n)!^2}{(2 k + n + 1)! n!}
 * \f}
 */
const Matrix& projection_matrix_child_to_parent(
    const Mesh<1>& child_mesh, const Mesh<1>& parent_mesh, ChildSize size,
    bool operand_is_massive = false) noexcept;

/// The projection matrix from a child mesh to its parent, in `Dim` dimensions.
template <size_t Dim>
std::array<std::reference_wrapper<const Matrix>, Dim>
projection_matrix_child_to_parent(const Mesh<Dim>& child_mesh,
                                  const Mesh<Dim>& parent_mesh,
                                  const std::array<ChildSize, Dim>& child_sizes,
                                  bool operand_is_massive = false) noexcept;

/// The projection matrix from a parent mesh to one of its children.
///
/// \see projection_matrix_child_to_parent()
const Matrix& projection_matrix_parent_to_child(const Mesh<1>& parent_mesh,
                                                const Mesh<1>& child_mesh,
                                                ChildSize size) noexcept;

/// The projection matrix from a parent mesh to one of its children, in `Dim`
/// dimensions
template <size_t Dim>
std::array<std::reference_wrapper<const Matrix>, Dim>
projection_matrix_parent_to_child(
    const Mesh<Dim>& parent_mesh, const Mesh<Dim>& child_mesh,
    const std::array<ChildSize, Dim>& child_sizes) noexcept;

}  // namespace Spectral
