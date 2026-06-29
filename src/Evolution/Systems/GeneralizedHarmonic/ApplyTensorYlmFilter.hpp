// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/TensorYlm/ApplyFilter.hpp"
#include "NumericalAlgorithms/TensorYlm/Filter.hpp"
#include "NumericalAlgorithms/TensorYlm/TensorYlm.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
namespace ylm {
class Spherepack;
}  // namespace ylm
/// \endcond

namespace ylm::TensorYlm {

/// Defines tags and functions used internally in filtering, but
/// tested independently in the unit tests.
namespace filter_detail {

/// Defines tags for internal use in filtering.
namespace Tags {

/// The time-time part of the metric.
template <typename DataType>
struct Metric00 : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The time-time part of the generalized harmonic variable \f$Pi\f$
template <typename DataType>
struct Pi00 : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The time-space part of the metric.
template <typename DataType, size_t Dim, typename Frame>
struct Metrick0 : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/// The time-space part of the generalized harmonic variable \f$Pi\f$
template <typename DataType, size_t Dim, typename Frame>
struct Pik0 : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/// The space-space part of the metric.
template <typename DataType, size_t Dim, typename Frame>
struct Metrickj : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

/// The space-space part of the generalized harmonic variable \f$Pi\f$
template <typename DataType, size_t Dim, typename Frame>
struct Pikj : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

/// \f$\Phi_{k00}, where \f$\Phi\f$ is the generalized harmonic variable.
template <typename DataType, size_t Dim, typename Frame>
struct Phik00 : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/// \f$\Phi_{ki0}, where \f$\Phi\f$ is the generalized harmonic variable.
template <typename DataType, size_t Dim, typename Frame>
struct Phiki0 : db::SimpleTag {
  using type = tnsr::ij<DataType, Dim, Frame>;
};

/// \f$\Phi_{kij}, where \f$\Phi\f$ is the generalized harmonic variable.
template <typename DataType, size_t Dim, typename Frame>
struct Phikij : db::SimpleTag {
  using type = tnsr::ijj<DataType, Dim, Frame>;
};

}  // namespace Tags

using gh_spacetime_vars_list =
    tmpl::list<::gr::Tags::SpacetimeMetric<DataVector, 3, Frame::Inertial>,
               ::gh::Tags::Pi<DataVector, 3, Frame::Inertial>,
               ::gh::Tags::Phi<DataVector, 3, Frame::Inertial>>;

template <typename Frame>
using gh_spatial_vars_list = tmpl::list<
    Tags::Metric00<DataVector>, Tags::Metrick0<DataVector, 3, Frame>,
    Tags::Metrickj<DataVector, 3, Frame>, Tags::Pi00<DataVector>,
    Tags::Pik0<DataVector, 3, Frame>, Tags::Pikj<DataVector, 3, Frame>,
    Tags::Phik00<DataVector, 3, Frame>, Tags::Phiki0<DataVector, 3, Frame>,
    Tags::Phikij<DataVector, 3, Frame>>;

/*!
 * \brief Copies spacetime variables into their spatial pieces.
 *
 * For example, if one of the spacetime variables is the metric
 * $g_{ab}$, then the corresponding spatial variables are a spatial
 * scalar $g_{00}$, a spatial vector $g_{i0}$, and a spatial symmetric
 * 2-tensor $g_{ij}$.
 *
 * The arguments must already be allocated to their correct sizes; no
 * memory allocation is done.
 *
 * \param spatial_vars Points to a Variables containing spatial pieces.
 * \param spacetime_vars A Variables containing the spacetime variables.
 */
void break_spacetime_vars_into_spatial_pieces(
    gsl::not_null<Variables<gh_spatial_vars_list<Frame::Inertial>>*>
        spatial_vars,
    const Variables<gh_spacetime_vars_list>& spacetime_vars);

/*!
 * \brief Copies spatial pieces into the corresponding spacetime variables.
 *
 * This is the inverse of break_spacetime_vars_into_spatial_pieces.
 *
 * \param spatial_vars A Variables containing spatial pieces.
 * \param spacetime_vars A Variables containing the spacetime variables.
 */
void assemble_spacetime_vars_from_spatial_pieces(
    gsl::not_null<Variables<gh_spacetime_vars_list>*> spacetime_vars,
    const Variables<gh_spatial_vars_list<Frame::Inertial>>& spatial_vars);

/*!
 * \brief Transforms spatial tensors into a different frame, ignoring hessians.
 *
 * This is done for filtering, where having the correct (i.e. with hessians)
 * transformation doesn't matter; all that matters is that the tensor
 * indices correspond to the coordinates (or in other words, no dual frame).
 *
 * Assumes that all the variables have lower indices.
 *
 * Takes special care to re-use memory. The Variables arguments must
 * already be allocated to their correct sizes; no memory allocation
 * is done.
 *
 * \tparam SrcFrame Source frame.
 * \tparam DestFrame Destination frame.
 * \param dest A Variables for the destination spatial variables.
 * \param src A Variables containing the source spatial variables.
 * \param jac The jacobian dx^src/dx^dest
 */
template <typename SrcFrame, typename DestFrame>
void transform_spatial_tensors_to_different_frame_without_hessians(
    gsl::not_null<Variables<gh_spatial_vars_list<DestFrame>>*> dest,
    const Variables<gh_spatial_vars_list<SrcFrame>>& src,
    const InverseJacobian<DataVector, 3, SrcFrame, DestFrame>& jac);

}  // namespace filter_detail

/*!
 * \brief Applies TensorYlm filter in place to Generalized Harmonic variables.
 *
 * When radial_extents is 1, gh_vars and temp_storage are assumed to
 * be defined on a spherical slice, with number of grid points
 * corresponding to a spherical-harmonic grid of ell_max, and the
 * filter happens only on that slice.
 *
 * When radial_extents is > 1, gh_vars and temp_storage are assumed to
 * be defined on a spherical shell of topology I1 x S2. The filter
 * happens in the entire volume, internally iterating over each
 * spherical slice at a time.
 *
 * For performance reasons, apply_tensor_ylm_filter does not allocate
 * or deallocate memory, but it does take a temp_storage buffer.  The
 * size of temp_storage should at least
 * radial_extents*spectral_size*num_components, where num_components
 * is the total number of independent components in the GH variable
 * list (i.e. 30), and spectral_size is the size of the S2 Spherepack
 * spectral coefficient array for ell_max, as obtained from the member
 * function ylm::Spherepack::spectral_size().  Note that for S2 on
 * Spherepack, the number of collocation points is different than the
 * number of spectral coefficients, and both are different than the
 * size of the Spherepack storage array.
 *
 * \param gh_vars Generalized Harmonic variables at collocation points.
 * \param temp_storage Temporary storage for generalized harmonic variables,
 *   allocated outside apply_tensor_ylm_filter. See above for size requirements.
 * \param jac_inertial_to_grid Jacobian taking V_x from inertial to grid.
 * \param jac_grid_to_inertial Jacobian taking V_x from grid to inertial.
 * \param filter_matrices The bundle of filter matrices computed by
 *   fill_filter (scalar, rank-1 `i`, rank-2 symmetric `ii`, rank-2 `ij`,
 *   and rank-3 `kii`).
 * \param ell_max The maximum ylm ell.
 * \param radial_extents The number of radial grid points, can be 1 for slices.
 */
template <>
void apply_tensor_ylm_filter(
    gsl::not_null<Variables<filter_detail::gh_spacetime_vars_list>*> gh_vars,
    gsl::not_null<Variables<filter_detail::gh_spacetime_vars_list>*>
        temp_storage,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
        jac_grid_to_inertial,
    const ylm::TensorYlm::FilterMatrixHolder& filter_matrices, size_t ell_max,
    size_t radial_extents);
}  // namespace ylm::TensorYlm
