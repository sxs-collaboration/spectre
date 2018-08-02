// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/EquationsOfState/EquationOfState.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;
/// \endcond

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor

namespace grmhd {
namespace ValenciaDivClean {
/*!
 * \brief Compute the primitive variables from the conservative variables
 *
 * For the Valencia formulation of the GRMHD system with divergence cleaning,
 * the conversion of the evolved conserved variables to the primitive variables
 * cannot be expressed in closed analytic form and requires a root find.
 *
 * [Siegel {\em et al}, The Astrophysical Journal 859:71(2018)]
 * (http://iopscience.iop.org/article/10.3847/1538-4357/aabcc5/meta)
 * compares several inversion methods.
 */
template <typename PrimitiveRecoveryScheme, size_t ThermodynamicDim>
void primitive_from_conservative(
    gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> spatial_velocity,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> magnetic_field,
    gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
    gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    gsl::not_null<Scalar<DataVector>*> pressure,
    gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) noexcept;
}  // namespace ValenciaDivClean
}  // namespace grmhd
