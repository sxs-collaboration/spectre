// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlm.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace ylm {
class Spherepack;
}  // namespace ylm
/// \endcond

namespace ylm::TensorYlm::filter_detail {

/*!
 * \brief Does a nodal to modal transform on I1xS2, only in the S2 direction.
 *
 * \tparam VarsList A tmpl::list of tags to transform.
 * \param nodal A Variables containing the input nodal values.
 * \param modal A Variables containing the output modal values.
 * \param ylm A Spherepack object with the correct ell_max.
 * \param radial_extents Number of grid points in I1, or 1 if just an S2.
 */
template <typename VarsList>
void nodal_to_modal_ylm(gsl::not_null<Variables<VarsList>*> modal,
                        const Variables<VarsList>& nodal,
                        const ::ylm::Spherepack& ylm, size_t radial_extents);

/*!
 * \brief Does a modal to nodal transform on I1xS2, only in the S2 direction.
 *
 * \tparam VarsList A tmpl::list of tags to transform.
 * \param modal A Variables containing the input modal values.
 * \param nodal A Variables containing the output nodal values.
 * \param ylm A Spherepack object with the correct ell_max.
 * \param radial_extents Number of grid points in I1, or 1 if just an S2.
 */
template <typename VarsList>
void modal_to_nodal_ylm(gsl::not_null<Variables<VarsList>*> nodal,
                        const Variables<VarsList>& modal,
                        const ::ylm::Spherepack& ylm, size_t radial_extents);
}  // namespace ylm::TensorYlm::filter_detail
