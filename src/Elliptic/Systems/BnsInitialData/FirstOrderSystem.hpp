
// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/BnsInitialData/Equations.hpp"
#include "Elliptic/Systems/BnsInitialData/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace BnsInitialData {

/*!
 * \brief The Irrotational Bns equations From Baumgarte and Shapiro Chapter 15
 *  formulated as a set of coupled first-order PDEs.
 *
 * \details This system formulates the Irrotational Bns Hydrostatic Equilibrium
 * equations for the velocity potential \f$\Phi\f$. For a background matter
 * distribution (given by the specific enthalpy h) and a background metric
 * \f$\gamma_{ij}\f$. The velocity potential is defined by \f$D_i \Phi = h
 * u_i\f$ with \f$u_i\f$ (the spatial part of) the four velocity and where
 * \f$\Gamma^i_{jk}=\frac{1}{2}\gamma^{il}\left(\partial_j\gamma_{kl}
 * +\partial_k\gamma_{jl}-\partial_l\gamma_{jk}\right)\f$ are the Christoffel
 * symbols of the second kind of the background (spatial) metric
 * \f$\gamma_{ij}\f$. The
 * background metric \f$\gamma_{ij}\f$ and the Christoffel symbols derived from
 * it are assumed to be independent of the variables \f$\Phi\f$ and \f$u_i\f$,
 * i.e.
 * constant throughout an iterative elliptic solve.  Additionally a background
 * lapse (\f$\alpha\f$) and
 * shift (\f$\beta\f$) must be provided.  Finally, a ``rotational killing
 * vector" \f$k^i\f$ (with magnitude
 * proportional to the angular velocity of the orbital motion) is provided.  The
 * rotational shift is defined as \f$B^i = \beta^i + k^i\f$ which is
 * heuristically the background
 * motion of the spacetime.
 *
 * The system can be formulated in terms of these fluxes and sources (see
 * `elliptic::protocols::FirstOrderSystem`):
 *
 *
 * \f{align*}
 * -\partial_i F^i + S = f
 * \f}
 *
 * \f{align*}
 * F^i &=  D_i \phi - \frac{B^j D_j \phi}{\alpha^2}B^i  \\
 * S &= -F^iD_i \left( \ln \frac{\alpha \rho}{h}\right) -\Gamma^i_{ij}F^j \\
 * f &= -D_i \left(\frac{C B^i}{\alpha^2}\right) -
 * \frac{C}{\alpha^2}B^iD_i\left(
 * \ln \frac{\alpha \rho}{h}\right)\\
 * \f}
 */
struct FirstOrderSystem
    : tt::ConformsTo<elliptic::protocols::FirstOrderSystem> {
 private:
  using velocity_potential = Tags::VelocityPotential<DataVector>;

 public:
  static constexpr size_t volume_dim = 3;

  using primal_fields = tmpl::list<velocity_potential>;

  // We just use the standard `Flux` prefix because the fluxes don't have
  // symmetries and we don't need to give them a particular meaning.
  using primal_fluxes = tmpl::list<
      ::Tags::Flux<velocity_potential, tmpl::size_t<3>, Frame::Inertial>>;

  using background_fields = tmpl::list<
      gr::Tags::InverseSpatialMetric<DataVector, 3>,
      gr::Tags::SpatialChristoffelSecondKindContracted<DataVector, 3>,
      gr::Tags::Lapse<DataVector>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>,
                    tmpl::integral_constant<size_t, 3>, Frame::Inertial>,
      gr::Tags::Shift<DataVector, 3>,
      ::Tags::deriv<gr::Tags::Shift<DataVector, 3>,
                    tmpl::integral_constant<size_t, 3>, Frame::Inertial>,
      Tags::RotationalShift<DataVector>,
      Tags::DerivLogLapseTimesDensityOverSpecificEnthalpy<DataVector>,
      Tags::RotationalShiftStress<DataVector>>;
  using inv_metric_tag = gr::Tags::InverseSpatialMetric<DataVector, 3>;

  using fluxes_computer = Fluxes;
  using sources_computer = Sources;

  using boundary_conditions_base =
      elliptic::BoundaryConditions::BoundaryCondition<3>;
};
}  // namespace BnsInitialData
