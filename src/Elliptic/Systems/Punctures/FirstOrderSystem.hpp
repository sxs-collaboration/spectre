// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Elliptic/Systems/Punctures/Sources.hpp"
#include "Elliptic/Systems/Punctures/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Punctures {

/*!
 * \brief The puncture equation, formulated as a set of coupled first-order
 * partial differential equations
 *
 * See \ref Punctures for details on the puncture equation. Since it is just a
 * flat-space Poisson equation with nonlinear sources, we can reuse the
 * Euclidean Poisson fluxes.
 */
struct FirstOrderSystem
    : tt::ConformsTo<elliptic::protocols::FirstOrderSystem> {
 private:
  using field = Tags::Field;
  using field_gradient = ::Tags::deriv<field, tmpl::size_t<3>, Frame::Inertial>;

 public:
  static constexpr size_t volume_dim = 3;

  using primal_fields = tmpl::list<field>;
  using auxiliary_fields = tmpl::list<field_gradient>;

  using primal_fluxes =
      tmpl::list<::Tags::Flux<field, tmpl::size_t<3>, Frame::Inertial>>;
  using auxiliary_fluxes = tmpl::list<
      ::Tags::Flux<field_gradient, tmpl::size_t<3>, Frame::Inertial>>;

  using background_fields = tmpl::list<Tags::Alpha, Tags::Beta>;
  using inv_metric_tag = void;

  using fluxes_computer = Poisson::Fluxes<3, Poisson::Geometry::FlatCartesian>;
  using sources_computer = Sources;
  using sources_computer_linearized = LinearizedSources;

  using boundary_conditions_base =
      elliptic::BoundaryConditions::BoundaryCondition<3>;
};

}  // namespace Punctures
