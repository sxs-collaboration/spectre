// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Poisson::FirstOrderSystem

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/IrrotationalBns/Equations.hpp"
#include "Elliptic/Systems/IrrotationalBns/Geometry.hpp"
#include "Elliptic/Systems/IrrotationalBns/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace IrrotationalBns {

/*!
 * \brief The Irrotational Bns equations From Baumgarte and Shapiro Chapter 15
 *  formulated as a set of coupled first-order PDEs.
 *
 * \details This system formulates the Poisson equation \f$-\Delta_\gamma u(x) =
 * f(x)\f$ on a background metric \f$\gamma_{ij}\f$ as the set of coupled
 * first-order PDEs
 *
 * \f{align*}
 * -\partial_i \gamma^{ij} v_j(x) - \Gamma^i_{ij}\gamma^{jk}v_k &= f(x) \\
 * -\partial_i u(x) + v_i(x) &= 0
 * \f}
 *
 * where we have chosen the field gradient as an auxiliary variable \f$v_i\f$
 * and where \f$\Gamma^i_{jk}=\frac{1}{2}\gamma^{il}\left(\partial_j\gamma_{kl}
 * +\partial_k\gamma_{jl}-\partial_l\gamma_{jk}\right)\f$ are the Christoffel
 * symbols of the second kind of the background metric \f$\gamma_{ij}\f$. The
 * background metric \f$\gamma_{ij}\f$ and the Christoffel symbols derived from
 * it are assumed to be independent of the variables \f$u\f$ and \f$v_i\f$, i.e.
 * constant throughout an iterative elliptic solve.
 *
 * The system can be formulated in terms of these fluxes and sources (see
 * `elliptic::protocols::FirstOrderSystem`):
 *
 * \f{align*}
 * F^i_u &= \gamma^{ij} v_j(x) \\
 * S_u &= -\Gamma^i_{ij}\gamma^{jk}v_k \\
 * f_u &= f(x) \\
 * F^i_{v_j} &= u \delta^i_j \\
 * S_{v_j} &= v_j \\
 * f_{v_j} &= 0 \text{.}
 * \f}
 *
 * The fluxes and sources simplify significantly when the background metric is
 * flat and we employ Cartesian coordinates so \f$\gamma_{ij} = \delta_{ij}\f$
 * and \f$\Gamma^i_{jk} = 0\f$. Set the template parameter `BackgroundGeometry`
 * to `Poisson::Geometry::FlatCartesian` to specialise the system for this case.
 * Set it to `Poisson::Geometry::Curved` for the general case.
 */
template <Geometry BackgroundGeometry>
struct FirstOrderSystem
    : tt::ConformsTo<elliptic::protocols::FirstOrderSystem> {
 private:
  using velocity_potential = Tags::VelocityPotential;
  using auxiliary_velocity = Tags::AuxiliaryVelocity;

 public:
  static constexpr size_t volume_dim = 3;

  using primal_fields = tmpl::list<velocity_potential>;
  using auxiliary_fields = tmpl::list<auxiliary_velocity>;

  // We just use the standard `Flux` prefix because the fluxes don't have
  // symmetries and we don't need to give them a particular meaning.
  using primal_fluxes = tmpl::list<
      ::Tags::Flux<velocity_potential, tmpl::size_t<3>, Frame::Inertial>>;
  using auxiliary_fluxes = tmpl::list<
      ::Tags::Flux<auxiliary_velocity, tmpl::size_t<3>, Frame::Inertial>>;

  using background_fields = tmpl::list<
      gr::Tags::InverseSpatialMetric<DataVector, 3>,
      gr::Tags::SpatialChristoffelSecondKindContracted<DataVector, 3>,
      gr::Tags::Lapse<DataVector>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::int_<3>,
                    Frame::Inertial>,
      gr::Tags::Shift<DataVector, 3>,
      ::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::int_<3>,
                    Frame::Inertial>,
      Tags::EulerEnthalpyConstant, Tags::RotationalShift,
      Tags::DerivLogLapseOverSpecificEnthalpy, Tags::FixedSources,
      Tags::RotationalShiftStress, Tags::DivergenceRotationalShiftStress,
      Tags::SpatialRotationalKillingVector,
      Tags::DerivSpatialRotationalKillingVector>;
  using inv_metric_tag =
      tmpl::conditional_t<BackgroundGeometry == Geometry::FlatCartesian, void,
                          gr::Tags::InverseSpatialMetric<DataVector, 3>>;

  using fluxes_computer = Fluxes<BackgroundGeometry>;
  using sources_computer = Sources<BackgroundGeometry>;

  using boundary_conditions_base =
      elliptic::BoundaryConditions::BoundaryCondition<3>;
};
}  // namespace IrrotationalBns
