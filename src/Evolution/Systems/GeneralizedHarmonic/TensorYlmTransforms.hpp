// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ApplyTensorYlmFilter.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
namespace ylm {
class Spherepack;
}  // namespace ylm
/// \endcond

namespace ylm::TensorYlm {

/*!
 * \brief Transform Generalized Harmonic variables on a spherical shell to
 * tensor-Ylm coefficients of their spatial pieces.
 *
 * The `gh_vars` are nodal values on either a spherical slice
 * (`radial_extents == 1`) or a shell with `radial_extents` radial points and
 * S2 angular collocation points described by `spherepack`. The output
 * `gh_spatial_tensor_ylm_coefficients` has the GH spacetime tensors broken
 * into spatial pieces, transformed to the grid frame using
 * `jac_inertial_to_grid`, converted from nodal to modal S2 coefficients, and
 * finally transformed from Cartesian components to the TensorYlm
 * \f$(\ell, m, \bar m)\f$ basis.
 *
 * The output uses Spherepack spectral storage with the radial offsets
 * interleaved in the same layout as `Spherepack::phys_to_spec_all_offsets`.
 * Its tensor component labels are not Cartesian component labels. They are
 * TensorYlm basis labels: component index 0 denotes an \f$\ell\f$ basis index,
 * component index 1 denotes an \f$m\f$ basis index, and component index 2
 * denotes an \f$\bar m\f$ basis index.
 *
 * Scalar quantities are transformed by the nodal-to-modal spherical-harmonic
 * transform. Non-scalar quantities are then transformed from Cartesian
 * components to the TensorYlm basis with the supplied sparse matrices.
 *
 * For performance, the function does not allocate large buffers and does not
 * build the cartesian-to-spherical matrices. The caller supplies
 * `temp_storage` and precomputed matrices. `temp_storage` has the same type as
 * the output so the implementation can reuse one contiguous buffer as inertial
 * spatial pieces at the collocation points and as spectral coefficients. It
 * must have at least `radial_extents * spherepack.spectral_size()` grid points.
 * The `spherepack` must have `m_max == l_max`, matching the
 * cartesian-to-spherical matrices.
 *
 * \param gh_spatial_tensor_ylm_coefficients Output coefficients for the GH
 *   variables broken into spatial pieces. The output is overwritten and must
 *   have `radial_extents * spherepack.spectral_size()` grid points.
 * \param temp_storage Temporary storage allocated by the caller. It is
 *   overwritten and must not alias `gh_spatial_tensor_ylm_coefficients`.
 * \param gh_vars Generalized Harmonic variables at collocation points. Must
 *   have `radial_extents * spherepack.physical_size()` grid points.
 * \param jac_inertial_to_grid Jacobian taking spatial tensor components from
 *   the inertial frame to the grid frame.
 * \param cart_to_sphere_matrix_i Cartesian-to-TensorYlm matrix for rank-1
 *   spatial tensors.
 * \param cart_to_sphere_matrix_ii Cartesian-to-TensorYlm matrix for symmetric
 *   rank-2 spatial tensors.
 * \param cart_to_sphere_matrix_ij Cartesian-to-TensorYlm matrix for rank-2
 *   spatial tensors with no symmetry.
 * \param cart_to_sphere_matrix_ijj Cartesian-to-TensorYlm matrix for rank-3
 *   spatial tensors symmetric on the last two indices.
 * \param spherepack The spherical-harmonic transform object describing the S2
 *   grid and spectral storage.
 * \param radial_extents The number of radial grid points, or 1 for a single
 *   spherical slice.
 */
void gh_variables_to_tensor_ylm_coefficients(
    gsl::not_null<Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>*>
        gh_spatial_tensor_ylm_coefficients,
    gsl::not_null<Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>*>
        temp_storage,
    const Variables<filter_detail::gh_spacetime_vars_list>& gh_vars,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid,
    const SimpleSparseMatrix& cart_to_sphere_matrix_i,
    const SimpleSparseMatrix& cart_to_sphere_matrix_ii,
    const SimpleSparseMatrix& cart_to_sphere_matrix_ij,
    const SimpleSparseMatrix& cart_to_sphere_matrix_ijj,
    const Spherepack& spherepack, size_t radial_extents);

}  // namespace ylm::TensorYlm
