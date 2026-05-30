// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/ApplyTensorYlmFilter.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlm.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
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

template <typename Frame>
using sw_vars_list =
    tmpl::list<::CurvedScalarWave::Tags::Psi, ::CurvedScalarWave::Tags::Pi,
               ::CurvedScalarWave::Tags::Phi<3, Frame>>;

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
    gsl::not_null<Variables<sw_vars_list<DestFrame>>*> dest,
    const Variables<sw_vars_list<SrcFrame>>& src,
    const InverseJacobian<DataVector, 3, SrcFrame, DestFrame>& jac);

}  // namespace filter_detail

/*!
 * \brief Applies TensorYlm filter in place to Curved Scalar Wave variables.
 *
 * When radial_extents is 1, sw_vars and temp_storage are assumed to
 * be defined on a spherical slice, with number of grid points
 * corresponding to a spherical-harmonic grid of ell_max, and the
 * filter happens only on that slice.
 *
 * When radial_extents is > 1, sw_vars and temp_storage are assumed to
 * be defined on a spherical shell of topology I1 x S2. The filter
 * happens in the entire volume, internally iterating over each
 * spherical slice at a time.
 *
 * For performance reasons, apply_tensor_ylm_filter does not allocate
 * or deallocate memory, but it does take a temp_storage buffer.  The
 * size of temp_storage should at least
 * radial_extents*spectral_size*num_components, where num_components
 * is the total number of independent components in the SW variable
 * list (i.e. 5), and spectral_size is the size of the S2 Spherepack
 * spectral coefficient array for ell_max, as obtained from the member
 * function ylm::Spherepack::spectral_size().  Note that for S2 on
 * Spherepack, the number of collocation points is different than the
 * number of spectral coefficients, and both are different than the
 * size of the Spherepack storage array.
 *
 * \param sw_vars Scalar wave variables at collocation points.
 * \param temp_storage Temporary storage for scalar wave variables,
 *   allocated outside apply_tensor_ylm_filter. See above for size requirements.
 * \param jac_inertial_to_grid Jacobian taking V_x from inertial to grid.
 * \param jac_grid_to_inertial Jacobian taking V_x from grid to inertial.
 * \param filter_matrices The bundle of filter matrices computed by
 *   fill_filter. Only the `scalar` and `i` members are used here.
 * \param ell_max The maximum ylm ell.
 * \param radial_extents The number of radial grid points, can be 1 for slices.
 */
template <>
void apply_tensor_ylm_filter(
    gsl::not_null<Variables<filter_detail::sw_vars_list<Frame::Inertial>>*>
        sw_vars,
    gsl::not_null<Variables<filter_detail::sw_vars_list<Frame::Inertial>>*>
        temp_storage,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
        jac_grid_to_inertial,
    const FilterMatrixHolder& filter_matrices, size_t ell_max,
    size_t radial_extents);

}  // namespace ylm::TensorYlm
