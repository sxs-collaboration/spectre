// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

/*!
 * \ingroup HydroInitialDataGroup
 * \brief Items related to solving for irrotational bns initial data
 * See e.g. Baumgarte and Shapiro, Ch. 15 (P. 523)
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
/// @}
template <typename DataType>
tnsr::I<DataType, 3> rotational_shift(
    const tnsr::I<DataType, 3>& shift,
    const tnsr::I<DataType, 3>& spatial_rotational_killing_vector);
/// @{
/// \brief Compute the  stress-energy corresponding to the rotation shift
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
/// \brief  Compute derivative  \f$D_i B^j / \alpha \f$
///
/// Here D_i is the spatial covariant derivative, \f$ \alpha\f$ is the lapse
/// and \f$B^i\f$ the rotational shift)
template <typename DataType>
void derivative_rotational_shift_over_lapse(
    gsl::not_null<tnsr::iJ<DataType, 3>*> result,
    const tnsr::I<DataType, 3>& rotational_shift,
    const tnsr::iJ<DataType, 3>& deriv_of_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType, 3>& deriv_of_lapse,
    const tnsr::iJ<DataType, 3>& deriv_of_spatial_rotational_killing_vector);
template <typename DataType>
tnsr::iJ<DataType, 3> derivative_rotational_shift_over_lapse(
    const tnsr::I<DataType, 3>& rotational_shift,
    const tnsr::iJ<DataType, 3>& deriv_of_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType, 3>& deriv_of_lapse,
    const tnsr::iJ<DataType, 3>& deriv_of_spatial_rotational_killing_vector);
/// @}

/// @{
/// \brief Compute the enthalpy density squared from other hydro variables and
/// the spacetime
///
/// The eqn. is identical to B&S 15.76
/// \f[
///   h^2 \rho^2 = \frac{1}{\alpha^2} \left(C + B^i D_i \Phi\right)^2 - D_i
/// \Phi D^i \Phi
/// /f\
/// Where \f$\Phi \f$ is the velocity potential, andC is the Euler-constant,
/// which in a slowly rotating, slowly orbiting configuration becomes the
/// central enthalpy squared over the central lapse squared
template <typename DataType>
void enthalpy_density_squared(
    gsl::not_null<Scalar<DataType>*> result,
    const tnsr::I<DataType, 3>& rotational_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType, 3>& velocity_potential_gradient,
    const tnsr::II<DataType, 3>& inverse_spatial_metric,
    double euler_enthalpy_constant);
template <typename DataType>
Scalar<DataType> enthalpy_density_squared(
    const tnsr::I<DataType, 3>& rotational_shift, const Scalar<DataType>& lapse,
    const tnsr::i<DataType, 3>& velocity_potential_gradient,
    const tnsr::II<DataType, 3>& inverse_spatial_metric,
    double euler_enthalpy_constant);
/// @}

/// @{
/// \brief Compute the spatial rotational killing vector associated with uniform
/// rotation around the z-axis.
template <typename DataType>
void spatial_rotational_killing_vector(
    gsl::not_null<tnsr::I<DataType, 3>*> result, const tnsr::I<DataType, 3>& x,
    const Scalar<DataType>& local_angular_velocity_around_z,
    const Scalar<DataType>& determinant_spatial_metric);
template <typename DataType>
tnsr::I<DataType, 3> spatial_rotational_killing_vector(
    const tnsr::I<DataType, 3>& x,
    const Scalar<DataType>& local_angular_velocity_around_z,
    const Scalar<DataType>& sqrt_det_spatial_metric);
/// @}

/// @{
/// \brief The spatial derivative of the spatial rotational killing vector
///
/// As above, assumes uniform rotation around the z-axis
template <typename DataType>
void derivative_spatial_rotational_killing_vector(
    gsl::not_null<tnsr::iJ<DataType, 3>*> result, const tnsr::I<DataType, 3>& x,
    const Scalar<DataType>& local_angular_velocity_around_z,
    const Scalar<DataType>& determinant_spatial_metric);
template <typename DataType>
tnsr::iJ<DataType, 3> derivative_spatial_rotational_killing_vector(
    const tnsr::I<DataType, 3>& x,
    const Scalar<DataType>& local_angular_velocity_around_z,
    const Scalar<DataType>& sqrt_det_spatial_metric);
}  // namespace hydro::initial_data::irrotational_bns
