// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/RelativisticEuler/Valencia/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace RelativisticEuler {
namespace Valencia {

/*!
 * \brief The fluxes of the conservative variables
 *
 * \f{align*}
 * F^i({\tilde D}) = &~ {\tilde D} v^i_{tr} \\
 * F^i({\tilde S}_j) = &~  {\tilde S}_j v^i_{tr} + \sqrt{\gamma} \alpha p
 * \delta^i_j \\
 * F^i({\tilde \tau}) = &~  {\tilde \tau} v^i_{tr} + \sqrt{\gamma} \alpha p v^i
 * \f}
 * where the conservative variables \f${\tilde D}\f$, \f${\tilde S}_i\f$, and
 * \f${\tilde \tau}\f$ are a generalized mass-energy density, momentum density,
 * and specific internal energy density as measured by an Eulerian observer,
 * \f$v^i_{tr} = \alpha v^i - \beta^i\f$ is the transport velocity, \f$\alpha\f$
 * is the lapse, \f$\beta^i\f$ is the shift, \f$v^i\f$ is the spatial velocity,
 * \f$\gamma\f$ is the determinant of the spatial metric, and \f$p\f$ is the
 * pressure.
 */
template <size_t Dim>
struct ComputeFluxes {
  using return_tags =
      tmpl::list<::Tags::Flux<RelativisticEuler::Valencia::Tags::TildeD,
                              tmpl::size_t<Dim>, Frame::Inertial>,
                 ::Tags::Flux<RelativisticEuler::Valencia::Tags::TildeTau,
                              tmpl::size_t<Dim>, Frame::Inertial>,
                 ::Tags::Flux<RelativisticEuler::Valencia::Tags::TildeS<Dim>,
                              tmpl::size_t<Dim>, Frame::Inertial>>;

  using argument_tags =
      tmpl::list<RelativisticEuler::Valencia::Tags::TildeD,
                 RelativisticEuler::Valencia::Tags::TildeTau,
                 RelativisticEuler::Valencia::Tags::TildeS<Dim>,
                 gr::Tags::Lapse<>, gr::Tags::Shift<Dim>,
                 gr::Tags::SqrtDetSpatialMetric<>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>>;

  static void apply(
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> tilde_d_flux,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> tilde_tau_flux,
      gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Inertial>*> tilde_s_flux,
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& tilde_s,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const Scalar<DataVector>& pressure,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& spatial_velocity);
};
}  // namespace Valencia
}  // namespace RelativisticEuler
