// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
// IWYU pragma: no_include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
// IWYU pragma: no_include "PointwiseFunctions/Hydro/Tags.hpp"

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
 * \brief Compute the conservative variables from primitive variables
 *
 * \f{align*}
 * {\tilde D} = & \sqrt{\gamma} \rho W \\
 * {\tilde S}_i = & \sqrt{\gamma} \left( \rho h W^2 v_i + B^m B_m v_i - B^m v_m
 * B_i \right) \\
 * {\tilde \tau} = & \sqrt{\gamma} \left[ \rho h W^2 - p - \rho W - \frac{1}{2}
 * (B^m v_m)^2 + \frac{1}{2} B^m B_m \left( 1 + v^m v_m \right) \right] \\
 * {\tilde B}^i = & \sqrt{\gamma} B^i \\
 * {\tilde \Phi} = & \sqrt{\gamma} \Phi
 * \f}
 *
 * where the conserved variables \f${\tilde D}\f$, \f${\tilde S}_i\f$,
 * \f${\tilde \tau}\f$, \f${\tilde B}^i\f$, and \f${\tilde \Phi}\f$ are a
 * generalized mass-energy density, momentum density, specific internal energy
 * density, magnetic field, and divergence cleaning field.  Furthermore
 * \f$\gamma\f$ is the determinant of the spatial metric, \f$\rho\f$ is the rest
 * mass density, \f$W = 1/\sqrt{1-v_i v^i}\f$ is the Lorentz factor, \f$h = 1 +
 * \epsilon + \frac{p}{\rho}\f$ is the specific enthalpy, \f$v^i\f$ is the
 * spatial velocity, \f$\epsilon\f$ is the specific internal energy, \f$p\f$ is
 * the pressure, \f$B^i\f$ is the spatial magnetic field measured by an Eulerian
 * observer, and \f$\Phi\f$ is a divergence cleaning field.
 *
 * The quantity \f${\tilde \tau}\f$ is rewritten as in `RelativisticEuler`
 * to avoid cancellation error in the non-relativistic limit:
 * \f[
 * \left( \rho h W^2 - p - \rho W \right) \longrightarrow
 *  W^2 \left[ \rho \left( \epsilon + v^2
 * \frac{W}{W + 1} \right) + p v^2 \right] .\f]
 */
struct ConservativeFromPrimitive {
  using return_tags = tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                                 grmhd::ValenciaDivClean::Tags::TildeTau,
                                 grmhd::ValenciaDivClean::Tags::TildeS<>,
                                 grmhd::ValenciaDivClean::Tags::TildeB<>,
                                 grmhd::ValenciaDivClean::Tags::TildePhi>;

  using argument_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::SpecificEnthalpy<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 gr::Tags::SqrtDetSpatialMetric<>, gr::Tags::SpatialMetric<3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> tilde_d,
      gsl::not_null<Scalar<DataVector>*> tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
      gsl::not_null<Scalar<DataVector>*> tilde_phi,
      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& specific_internal_energy,
      const Scalar<DataVector>& specific_enthalpy,
      const Scalar<DataVector>& pressure,
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
      const Scalar<DataVector>& lorentz_factor,
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const Scalar<DataVector>& divergence_cleaning_field) noexcept;
};
}  // namespace ValenciaDivClean
}  // namespace grmhd
