// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/ScalarGaussBonnet/Equations.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace sgb {

/*!
 * \brief The scalar equation in sGB theories of gravity in the decoupled limit,
 * assuming quasi-stationarity. See \ref sgb for details on the
 * explicit equation.
 */
struct FirstOrderSystem
    : tt::ConformsTo<elliptic::protocols::FirstOrderSystem> {
 private:
  using field = ::sgb::Tags::Psi;

 public:
  static constexpr size_t volume_dim = 3;

  using primal_fields = tmpl::list<field>;
  using primal_fluxes =
      tmpl::list<::Tags::Flux<field, tmpl::size_t<3>, Frame::Inertial>>;
  // Note that there are many more background fields required for the elliptic
  // solve, however these are numerically imported and so are not listed here.
  using background_fields = tmpl::list<
      Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
      Xcts::Tags::ConformalChristoffelContracted<DataVector, 3,
                                                 Frame::Inertial>>;
  using inv_metric_tag =
      Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>;

  using fluxes_computer = Fluxes;
  using sources_computer = Sources;
  using sources_computer_linearized = LinearizedSources;

  using boundary_conditions_base =
      elliptic::BoundaryConditions::BoundaryCondition<3>;
};

}  // namespace sgb
