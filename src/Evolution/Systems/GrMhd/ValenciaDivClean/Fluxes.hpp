// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"    // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
// IWYU pragma: no_include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
// IWYU pragma: no_include "PointwiseFunctions/Hydro/Tags.hpp"

// IWYU pragma: no_forward_declare Tags::Flux

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace grmhd {
namespace ValenciaDivClean {

/*!
 * \brief The fluxes of the conservative variables
 *
 * \f{align*}
 * F^i({\tilde D}) = &~ {\tilde D} v^i_{tr} \\
 * F^i({\tilde S}_j) = &~  {\tilde S}_j v^i_{tr} + \sqrt{\gamma} \alpha \left( p
 * + p_m \right) \delta^i_j - \frac{B_j {\tilde B}^i}{W^2} - v_j {\tilde B}^i
 * B^m v_m\\
 * F^i({\tilde \tau}) = &~  {\tilde \tau} v^i_{tr} + \sqrt{\gamma} \alpha \left(
 * p + p_m \right) v^i - \alpha {\tilde B}^i B^m v_m \\
 * F^i({\tilde B}^j) = &~  {\tilde B}^j v^i_{tr} - \alpha v^j {\tilde B}^i +
 * \alpha \gamma^{ij} {\tilde \Phi} \\
 * F^i({\tilde \Phi}) = &~ \alpha {\tilde B^i} - \beta^i {\tilde \Phi}
 * \f}
 *
 * where the conserved variables \f${\tilde D}\f$, \f${\tilde S}_i\f$,
 * \f${\tilde \tau}\f$, \f${\tilde B}^i\f$, and \f${\tilde \Phi}\f$ are a
 * generalized mass-energy density, momentum density, specific internal energy
 * density, magnetic field, and divergence cleaning field.  Furthermore,
 * \f$v^i_{tr} = \alpha v^i - \beta^i\f$ is the transport velocity, \f$\alpha\f$
 * is the lapse, \f$\beta^i\f$ is the shift, \f$\gamma\f$ is the determinant of
 * the spatial metric \f$\gamma_{ij}\f$,  \f$v^i\f$ is the spatial velocity,
 * \f$B^i\f$ is the spatial magnetic field measured by an Eulerian observer,
 * \f$p\f$ is the fluid pressure, and \f$p_m = \frac{1}{2} \left[ \left( B^n v_n
 * \right)^2 + B^n B_n / W^2 \right]\f$ is the magnetic pressure.
 */
struct ComputeFluxes {
  using return_tags =
      tmpl::list<::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeD,
                              tmpl::size_t<3>, Frame::Inertial>,
                 ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeTau,
                              tmpl::size_t<3>, Frame::Inertial>,
                 ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeS<>,
                              tmpl::size_t<3>, Frame::Inertial>,
                 ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeB<>,
                              tmpl::size_t<3>, Frame::Inertial>,
                 ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildePhi,
                              tmpl::size_t<3>, Frame::Inertial>>;

  using argument_tags =
      tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                 grmhd::ValenciaDivClean::Tags::TildeTau,
                 grmhd::ValenciaDivClean::Tags::TildeS<>,
                 grmhd::ValenciaDivClean::Tags::TildeB<>,
                 grmhd::ValenciaDivClean::Tags::TildePhi, gr::Tags::Lapse<>,
                 gr::Tags::Shift<3>, gr::Tags::SqrtDetSpatialMetric<>,
                 gr::Tags::SpatialMetric<3>, gr::Tags::InverseSpatialMetric<3>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::MagneticField<DataVector, 3>>;

  static void apply(
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_tau_flux,
      gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
      gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_phi_flux,
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const Scalar<DataVector>& tilde_phi, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& pressure,
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
      const Scalar<DataVector>& lorentz_factor,
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field) noexcept;
};
}  // namespace ValenciaDivClean
}  // namespace grmhd
