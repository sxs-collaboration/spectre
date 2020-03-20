// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

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
template <size_t Dim>
struct ConservativeFromPrimitive {
  using return_tags =
      tmpl::list<RelativisticEuler::Valencia::Tags::TildeD,
                 RelativisticEuler::Valencia::Tags::TildeTau,
                 RelativisticEuler::Valencia::Tags::TildeS<Dim>>;

  using argument_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::SpecificEnthalpy<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 gr::Tags::SqrtDetSpatialMetric<>,
                 gr::Tags::SpatialMetric<Dim>>;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> tilde_d,
      gsl::not_null<Scalar<DataVector>*> tilde_tau,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> tilde_s,
      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& specific_internal_energy,
      const Scalar<DataVector>& specific_enthalpy,
      const Scalar<DataVector>& pressure,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& spatial_velocity,
      const Scalar<DataVector>& lorentz_factor,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>&
          spatial_metric) noexcept;
};
}  // namespace Valencia
}  // namespace RelativisticEuler
