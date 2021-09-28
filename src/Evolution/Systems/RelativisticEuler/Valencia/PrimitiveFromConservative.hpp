// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState

namespace RelativisticEuler {
namespace Valencia {

/*!
 * \brief Compute the primitive variables from the conservative variables
 *
 * For the Valencia formulation of the Relativistic Euler system, the conversion
 * of the evolved conserved variables to the primitive variables cannot
 * be expressed in closed analytic form and requires a root find.  We use
 * the method from [Appendix C of Galeazzi \em et \em al, PRD 88, 064009 (2013)]
 * (https://journals.aps.org/prd/abstract/10.1103/PhysRevD.88.064009).  The
 * method finds the root of \f[ f(z) = z - \frac{r}{h(z)} \f] where
 * \f{align*}
 * r = & \frac{\sqrt{\gamma^{mn} {\tilde S}_m {\tilde S}_n}}{\tilde D} \\
 * h(z) = & [1+ \epsilon(z)][1 + a(z)] \\
 * \epsilon(z) = & W(z) q - z r + \frac{z^2}{1 + W(z)} \\
 * W(z) = & \sqrt{1 + z^2} \\
 * q = & \frac{\tilde \tau}{\tilde D} \\
 * a(z) = & \frac{p}{\rho(z) [1 + \epsilon(z)]} \\
 * \rho(z) = & \frac{\tilde D}{\sqrt{\gamma} W(z)}
 * \f}
 * and where the conserved variables \f${\tilde D}\f$, \f${\tilde S}_i\f$, and
 * \f${\tilde \tau}\f$ are a generalized mass-energy density, momentum density,
 * and specific internal energy density as measured by an Eulerian observer,
 * \f$\gamma\f$ and \f$\gamma^{mn}\f$ are the determinant and inverse of the
 * spatial metric \f$\gamma_{mn}\f$, \f$\rho\f$ is the rest mass density, \f$W =
 * 1/\sqrt{1 - \gamma_{mn} v^m v^n}\f$ is the Lorentz factor, \f$h = 1 +
 * \epsilon + \frac{p}{\rho}\f$ is the specific enthalpy, \f$v^i\f$ is the
 * spatial velocity, \f$\epsilon\f$ is the specific internal energy, and \f$p\f$
 * is the pressure.  The pressure is determined from the equation of state.
 * Finally, once \f$z\f$ is found, the spatial velocity is given by \f[ v^i =
 * \frac{{\tilde S}^i}{{\tilde D} W(z) h(z)} \f]
 *
 * \todo The method also will make corrections if physical bounds are violated,
 * see the paper for details.
 */
template <size_t Dim>
struct PrimitiveFromConservative {
  using return_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::SpecificEnthalpy<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>>;

  using argument_tags =
      tmpl::list<RelativisticEuler::Valencia::Tags::TildeD,
                 RelativisticEuler::Valencia::Tags::TildeTau,
                 RelativisticEuler::Valencia::Tags::TildeS<Dim>,
                 gr::Tags::InverseSpatialMetric<Dim>,
                 gr::Tags::SqrtDetSpatialMetric<>,
                 hydro::Tags::EquationOfStateBase>;

  template <size_t ThermodynamicDim>
  static void apply(
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<Scalar<DataVector>*> lorentz_factor,
      gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          spatial_velocity,
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& tilde_s,
      const tnsr::II<DataVector, Dim, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
          equation_of_state);
};
}  // namespace Valencia
}  // namespace RelativisticEuler
