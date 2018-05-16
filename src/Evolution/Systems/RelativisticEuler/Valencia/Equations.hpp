// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;
/// \endcond

namespace RelativisticEuler {
namespace Valencia {

/*!
 * \brief Compute the source terms for the flux-conservative Valencia
 * formulation of the relativistic Euler system.
 *
 *
 * A flux-conservative system has the generic form:
 * \f[
 * \partial_t U_i + \partial_m F^m(U_i) = S(U_i)
 * \f]
 *
 * where \f$F^a()\f$ denotes the flux of a conserved variable \f$U_i\f$ and
 * \f$S()\f$ denotes the source term for the conserved variable.
 *
 * For the Valencia formulation:
 * \f{align*}
 * S({\tilde D}) = & 0\\
 * S({\tilde S}_i) = & \frac{1}{2} \alpha {\tilde S}^{mn} \partial_i \gamma_{mn}
 * + {\tilde S}_m \partial_i \beta^m - ({\tilde D} + {\tilde \tau}) \partial_i
 * \alpha \\ S({\tilde \tau}) = & \alpha {\tilde S}^{mn} K_{mn}
 * - {\tilde S}^m \partial_m \alpha
 * \f}
 *
 * where
 * \f{align*}
 * {\tilde S}^i = & {\tilde S}_m \gamma^{im} \\
 * {\tilde S}^{ij} = & {\tilde S}^i v^j + \sqrt{\gamma} p \gamma^{ij}
 * \f}
 * where \f${\tilde D}\f$, \f${\tilde S}_i\f$, and \f${\tilde \tau}\f$ are a
 * generalized mass-energy density, momentum density, and specific internal
 * energy density as measured by an Eulerian observer, \f$\gamma\f$ is the
 * determinant of the spatial metric \f$\gamma_{ij}\f$, \f$\rho\f$ is the rest
 * mass density, \f$W\f$ is the Lorentz factor, \f$h\f$ is the specific
 * enthalpy, \f$v^i\f$ is the spatial velocity, \f$p\f$ is the pressure,
 * \f$\alpha\f$ is the lapse, \f$\beta^i\f$ is the shift, and \f$K_{ij}\f$ is
 * the extrinsic curvature.
 */
template <size_t Dim>
void compute_source_terms_of_u(
    gsl::not_null<Scalar<DataVector>*> source_tilde_d,
    gsl::not_null<Scalar<DataVector>*> source_tilde_tau,
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> source_tilde_s,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& pressure, const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, Dim, Frame::Inertial>& d_shift,
    const tnsr::ijj<DataVector, Dim, Frame::Inertial>& d_spatial_metric,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>&
        extrinsic_curvature) noexcept;
}  // namespace Valencia
}  // namespace RelativisticEuler
