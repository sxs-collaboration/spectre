// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

namespace RelativisticEuler {
namespace Valencia {

/*!
 * \brief Compute the conservative variables from primitive variables
 *
 * \f{align*}
 * {\tilde D} = & \sqrt{\gamma} \rho W \\
 * {\tilde S}_i = & \sqrt{\gamma} \rho h W^2 v_i \\
 * {\tilde \tau} = & \sqrt{\gamma} \left( \rho h W^2 - p - \rho W \right)
 * \f}
 * where \f${\tilde D}\f$, \f${\tilde S}_i\f$, and \f${\tilde \tau}\f$ are a
 * generalized mass-energy density, momentum density, and specific internal
 * energy density as measured by an Eulerian observer, \f$\gamma\f$ is the
 * determinant of the spatial metric, \f$\rho\f$ is the rest mass density,
 * \f$W = 1/\sqrt{1-v_i v^i}\f$ is the Lorentz factor, \f$h = 1 + \epsilon +
 * \frac{p}{\rho}\f$ is the specific enthalpy, \f$v_i\f$ is the spatial
 * velocity, \f$\epsilon\f$ is the specific internal energy, and \f$p\f$ is the
 * pressure.
 *
 * Using the definitions of the Lorentz factor and the specific enthalpy, the
 * last equation can be rewritten in a form that has a well-behaved Newtonian
 * limit: \f[
 * {\tilde \tau} = \sqrt{\gamma} W^2 \left[ \rho \left( \epsilon + v^2
 * \frac{W}{W + 1} \right) + p v^2 \right] .\f]
 */
template <typename DataType, size_t Dim>
void conservative_from_primitive(
    gsl::not_null<Scalar<DataType>*> tilde_d,
    gsl::not_null<Scalar<DataType>*> tilde_tau,
    gsl::not_null<tnsr::i<DataType, Dim, Frame::Inertial>*> tilde_s,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const tnsr::i<DataType, Dim, Frame::Inertial>& spatial_velocity_oneform,
    const Scalar<DataType>& spatial_velocity_squared,
    const Scalar<DataType>& lorentz_factor,
    const Scalar<DataType>& specific_enthalpy, const Scalar<DataType>& pressure,
    const Scalar<DataType>& sqrt_det_spatial_metric) noexcept;
}  // namespace Valencia
}  // namespace RelativisticEuler
