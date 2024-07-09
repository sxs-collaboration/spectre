// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

/*!
 * \brief Items related to solving for irrotational bns initial data
 * See e.g. \cite BaumgarteShapiro Ch. 15 (P. 523)
 */
namespace hydro::initial_data::irrotational_bns {
/// @{
/// \brief Compute the  shift plus a spatial vector \f$ k^i\f$ representing
/// the local binary rotation \f$B^i = \beta^i + k^i\f$
///
template <typename DataType>
void rotational_shift(
    gsl::not_null<tnsr::I<DataType, 3>*> result,
    const tnsr::I<DataType, 3>& shift,
    const tnsr::I<DataType, 3>& spatial_rotational_killing_vector);

template <typename DataType>
tnsr::I<DataType, 3> rotational_shift(
    const tnsr::I<DataType, 3>& shift,
    const tnsr::I<DataType, 3>& spatial_rotational_killing_vector);
/// @}

/// @{
/// \brief Compute the  stress-energy corresponding to the rotation shift.
/// (this has no corresponding equation number in \cite BaumgarteShapiro, it
/// is defined for convenience in evaluating fluxes and sources for the DG
/// scheme.)
///
///
/// \f[\Sigma^i_j = \frac{1}{2}\frac{B^iB_j}{\alpha^2}\f]
///
template <typename DataType>
void rotational_shift_stress(gsl::not_null<tnsr::II<DataType, 3>*> result,
                             const tnsr::I<DataType, 3>& rotational_shift,
                             const Scalar<DataType>& lapse);
template <typename DataType>
tnsr::II<DataType, 3> rotational_shift_stress(
    const tnsr::I<DataType, 3>& rotational_shift,
    const Scalar<DataType>& lapse);
/// @}

/// @{
/// \brief  Compute derivative  \f$ \partial_i (B^j / \alpha^2) \f$
///
/// Here \f$ \partial_i \f$ is the spatial partial derivative, \f$ \alpha\f$ is
/// the lapse and \f$B^i\f$ the rotational shift).  The derivatives passed as
/// arguments should be spatial partial derivatives.
template <typename DataType>
void derivative_rotational_shift_over_lapse_squared(
    gsl::not_null<tnsr::iJ<DataType, 3>*> result,
    const tnsr::I<DataType, 3>& rotational_shift,
    const tnsr::iJ<DataType, 3>& deriv_of_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType, 3>& deriv_of_lapse,
    const tnsr::iJ<DataType, 3>& deriv_of_spatial_rotational_killing_vector);
template <typename DataType>
tnsr::iJ<DataType, 3> derivative_rotational_shift_over_lapse_squared(
    const tnsr::I<DataType, 3>& rotational_shift,
    const tnsr::iJ<DataType, 3>& deriv_of_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType, 3>& deriv_of_lapse,
    const tnsr::iJ<DataType, 3>& deriv_of_spatial_rotational_killing_vector);
/// @}

/// @{
/// \brief Compute the specific enthalpy squared from other hydro variables and
/// the spacetime
///
/// The eqn. is identical in content to \cite BaumgarteShapiro 15.76, it
/// computes the specific enthalpy \f$ h \f$
/*!
   \f[
   h^2 = \frac{1}{\alpha^2} \left(C + B^i D_i \Phi\right)^2 - D_i
   \Phi D^i \Phi
   \f]
*/
/// Where \f$\Phi \f$ is the velocity potential, and \f$C\f$ is the
/// Euler-constant, which in a slowly rotating, slowly orbiting configuration
/// becomes the central specific enthalpy times the central lapse
template <typename DataType>
void specific_enthalpy_squared(
    gsl::not_null<Scalar<DataType>*> result,
    const tnsr::I<DataType, 3>& rotational_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType, 3>& velocity_potential_gradient,
    const tnsr::II<DataType, 3>& inverse_spatial_metric,
    double euler_enthalpy_constant);
template <typename DataType>
Scalar<DataType> specific_enthalpy_squared(
    const tnsr::I<DataType, 3>& rotational_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType, 3>& velocity_potential_gradient,
    const tnsr::II<DataType, 3>& inverse_spatial_metric,
    double euler_enthalpy_constant);
/// @}

/// @{
/// \brief Compute the spatial rotational killing vector associated with uniform
/// rotation around the z-axis.
///
/// Taking \f$\Omega_j\f$ to be the uniform rotation axis (assumed in the
/// z-direction) and \f$ \epsilon^{ijk}\f$ to be the Levi-Civita tensor
/// (\f$\epsilon_{ijk} = \sqrt{\gamma} e_{ijk}\f$, with \f$e_{ijk}\f$ totally
/// antisymmetric with \f$ e_{123} = 1\f$) , then
/// the killing vector is given by  (\cite BaumgarteShapiro 15.13) :
/*!
  \f[ k^i = \epsilon^{ijk}\Omega_j x_k \f]
 */
template <typename DataType>
void spatial_rotational_killing_vector(
    gsl::not_null<tnsr::I<DataType, 3>*> result, const tnsr::I<DataType, 3>& x,
    double orbital_angular_velocity,
    const Scalar<DataType>& sqrt_det_spatial_metric);
template <typename DataType>
tnsr::I<DataType, 3> spatial_rotational_killing_vector(
    const tnsr::I<DataType, 3>& x, double orbital_angular_velocity,
    const Scalar<DataType>& sqrt_det_spatial_metric);
/// @}

/// @{
/// \brief The spatial derivative of the spatial rotational killing vector
///
/// As for `spatial_rotational_killing_vector`, assumes uniform rotation around
/// the z-axis
template <typename DataType>
void divergence_spatial_rotational_killing_vector(
    gsl::not_null<Scalar<DataType>*> result, const tnsr::I<DataType, 3>& x,
    double orbital_angular_velocity,
    const Scalar<DataType>& sqrt_det_spatial_metric);
template <typename DataType>
Scalar<DataType> divergence_spatial_rotational_killing_vector(
    const tnsr::I<DataType, 3>& x, double orbital_angular_velocity,
    const Scalar<DataType>& sqrt_det_spatial_metric);
}  // namespace hydro::initial_data::irrotational_bns
