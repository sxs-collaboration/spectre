// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <functional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/SegmentSize.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace Spectral {
/// Determine whether data needs to be projected between a child mesh and its
/// parent mesh. If no projection is necessary the data may be used as-is.
/// Projection is necessary if the child is either p-refined or h-refined
/// relative to its parent, or both. This operation is symmetric, i.e. it is
/// irrelevant in which order the child and the parent mesh are passed in.
template <size_t Dim>
bool needs_projection(const Mesh<Dim>& mesh1, const Mesh<Dim>& mesh2,
                      const std::array<SegmentSize, Dim>& child_sizes);

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
 *
 * \note This and the other matrix-returning projection functions in this file
 * (projection_matrix_parent_to_child(), projection_matrices()) only support
 * tensor-product bases (Legendre, Fourier, ZernikeB2). They cannot be used for
 * spherical-shell meshes that use the `SphericalHarmonic` basis, because the
 * angular projection couples the \f$\theta\f$ and \f$\phi\f$ directions and so
 * cannot be expressed as the per-dimension matrices these functions return. Use
 * Spectral::project() instead whenever spherical-shell meshes must be
 * supported; it dispatches to these matrices for tensor-product meshes and to a
 * Spherepack-based angular projection for spherical-shell meshes.
 */
const Matrix& projection_matrix_child_to_parent(
    const Mesh<1>& child_mesh, const Mesh<1>& parent_mesh, SegmentSize size,
    bool operand_is_massive = false);

/// The projection matrix from a child mesh to its parent, in `Dim` dimensions.
template <size_t Dim>
std::array<std::reference_wrapper<const Matrix>, Dim>
projection_matrix_child_to_parent(
    const Mesh<Dim>& child_mesh, const Mesh<Dim>& parent_mesh,
    const std::array<SegmentSize, Dim>& child_sizes,
    bool operand_is_massive = false);

/// The projection matrix from a parent mesh to one of its children.
///
/// \see projection_matrix_child_to_parent()
const Matrix& projection_matrix_parent_to_child(const Mesh<1>& parent_mesh,
                                                const Mesh<1>& child_mesh,
                                                SegmentSize size);

/// The projection matrix from a parent mesh to one of its children, in `Dim`
/// dimensions
template <size_t Dim>
std::array<std::reference_wrapper<const Matrix>, Dim>
projection_matrix_parent_to_child(
    const Mesh<Dim>& parent_mesh, const Mesh<Dim>& child_mesh,
    const std::array<SegmentSize, Dim>& child_sizes);

/// The projection matrices from a source mesh to a target mesh
/// covering given portions of an element
template <size_t Dim>
std::array<std::reference_wrapper<const Matrix>, Dim> projection_matrices(
    const Mesh<Dim>& source_mesh, const Mesh<Dim>& target_mesh,
    const std::array<SegmentSize, Dim>& source_sizes,
    const std::array<SegmentSize, Dim>& target_sizes,
    bool operand_is_massive = false);

/// Change the angular resolution (`l_max`, with `m_max == l_max`) of volume
/// data on a spherical shell from `source_data` to `result_data`, for
/// `num_components` components each laid out with the radial dimension varying
/// fastest. Uses Spherepack prolong/restrict. \see Spectral::project() for the
/// higher-level interface to call this.
///
/// For non-massive operands this is the L2 (Galerkin) projection,
/// $P_\mathrm{L2}$. For massive operands (`operand_is_massive == true`) the
/// restriction is the transpose of the prolongation (interpolation) operator,
/// $I^T$, which is the L2 projection conjugated by the diagonal matrices $W$ of
/// angular quadrature weights:
/// $I^T = W_\mathrm{target} P_\mathrm{L2} W_\mathrm{source}^{-1}$.
/// This means we divide by the source weights before projecting and multiply by
/// the target weights afterwards. See projection_matrix_child_to_parent() for
/// details on massive operands.
void project_spherical_harmonics(gsl::not_null<double*> result_data,
                                 const double* source_data,
                                 size_t num_components,
                                 size_t num_radial_points, size_t l_max_source,
                                 size_t l_max_target, bool operand_is_massive);

/// @{
/*!
 * \brief Project volume data from `source_mesh` to `target_mesh`, writing the
 * (resized) result into `*result`.
 *
 * This is the unified entry point for projection regardless of basis. It
 * handles both tensor-product meshes and spherical-shell meshes (Legendre
 * radial dimension plus `SphericalHarmonic` angular dimensions with
 * `m_max == l_max`):
 * - For tensor-product meshes it delegates to the per-dimension projection
 *   matrices above.
 * - For spherical-shell meshes it projects the radial dimension with the 1D
 *   projection matrix and changes the angular `l_max` with a Spherepack
 *   prolong/restrict. The angular dimensions cannot be h-refined, so their
 *   segment sizes must be `Full`.
 *
 * `source_sizes` and `target_sizes` carry h-refinement information (which
 * portion of an element the source/target cover); pass `Full` in every
 * dimension for pure p-refinement.
 *
 * \warning `*result` must not point to `source` (the result is resized to the
 * target number of grid points).
 */
template <typename VectorType, size_t Dim>
void project(gsl::not_null<VectorType*> result, const VectorType& source,
             const Mesh<Dim>& source_mesh, const Mesh<Dim>& target_mesh,
             const std::array<SegmentSize, Dim>& source_sizes,
             const std::array<SegmentSize, Dim>& target_sizes,
             bool operand_is_massive = false);

template <size_t Dim, typename TagList>
void project(const gsl::not_null<Variables<TagList>*> result,
             const Variables<TagList>& source, const Mesh<Dim>& source_mesh,
             const Mesh<Dim>& target_mesh,
             const std::array<SegmentSize, Dim>& source_sizes,
             const std::array<SegmentSize, Dim>& target_sizes,
             const bool operand_is_massive = false) {
  // Type-erase to the vector implementation with multiple components
  using VectorType = typename Variables<TagList>::vector_type;
  using ValueType = typename Variables<TagList>::value_type;
  result->initialize(target_mesh.number_of_grid_points());
  VectorType result_view(result->data(), result->size());
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  const VectorType source_view(const_cast<ValueType*>(source.data()),
                               source.size());
  project(make_not_null(&result_view), source_view, source_mesh, target_mesh,
          source_sizes, target_sizes, operand_is_massive);
}

template <size_t Dim, typename T>
T project(const T& source, const Mesh<Dim>& source_mesh,
          const Mesh<Dim>& target_mesh,
          const std::array<SegmentSize, Dim>& source_sizes,
          const std::array<SegmentSize, Dim>& target_sizes,
          const bool operand_is_massive = false) {
  T result{};
  project(make_not_null(&result), source, source_mesh, target_mesh,
          source_sizes, target_sizes, operand_is_massive);
  return result;
}
/// @}

/// @{
/// \brief Performs a perfect hash of the mortars into $2^{d-1}$ slots on the
/// range $[0, 2^{d-1})$.
///
/// This is particularly useful when hashing into statically-sized maps based
/// on the number of dimensions.
template <size_t DimMinusOne>
size_t hash(const std::array<Spectral::SegmentSize, DimMinusOne>& mortar_size);

template <size_t Dim>
struct MortarSizeHash {
  template <size_t MaxSize>
  static constexpr bool is_perfect = MaxSize == two_to_the(Dim);

  size_t operator()(
      const std::array<Spectral::SegmentSize, Dim - 1>& mortar_size);
};
/// @}
}  // namespace Spectral
