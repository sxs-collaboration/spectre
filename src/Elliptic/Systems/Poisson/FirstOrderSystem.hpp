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
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Poisson {

/*!
 * \brief The Poisson equation formulated as a set of coupled first-order PDEs.
 *
 * \details This system formulates the Poisson equation \f$-\Delta_\gamma u(x) =
 * f(x)\f$ on a background metric \f$\gamma_{ij}\f$ as the set of coupled
 * first-order PDEs
 *
 * \f{align*}
 * -\partial_i v^i - \Gamma^i_{ij} v^j = f(x) \\
 * v^i = \gamma^{ij} \partial_j u(x)
 * \f}
 *
 * where \f$\Gamma^i_{jk}=\frac{1}{2}\gamma^{il}\left(\partial_j\gamma_{kl}
 * +\partial_k\gamma_{jl}-\partial_l\gamma_{jk}\right)\f$ are the Christoffel
 * symbols of the second kind of the background metric \f$\gamma_{ij}\f$. The
 * background metric \f$\gamma_{ij}\f$ and the Christoffel symbols derived from
 * it are assumed to be independent of the variables \f$u\f$, i.e.
 * constant throughout an iterative elliptic solve.
 *
 * The system can be formulated in terms of these fluxes and sources (see
 * `elliptic::protocols::FirstOrderSystem`):
 *
 * \f{align*}
 * F^i &= v^i = \gamma^{ij} \partial_j u \\
 * S &= -\Gamma^i_{ij} v^j \\
 * f &= f(x) \text{.}
 * \f}
 *
 * The fluxes and sources simplify significantly when the background metric is
 * flat and we employ Cartesian coordinates so \f$\gamma_{ij} = \delta_{ij}\f$
 * and \f$\Gamma^i_{jk} = 0\f$. Set the template parameter `BackgroundGeometry`
 * to `Poisson::Geometry::FlatCartesian` to specialise the system for this case.
 * Set it to `Poisson::Geometry::Curved` for the general case.
 */
template <size_t Dim, Geometry BackgroundGeometry>
struct FirstOrderSystem
    : tt::ConformsTo<elliptic::protocols::FirstOrderSystem> {
  static constexpr size_t volume_dim = Dim;

  using primal_fields = tmpl::list<Tags::Field<DataVector>>;
  // We just use the standard `Flux` prefix because the fluxes don't have
  // symmetries and we don't need to give them a particular meaning.
  using primal_fluxes =
      tmpl::list<::Tags::Flux<Tags::Field<DataVector>, tmpl::size_t<Dim>,
                              Frame::Inertial>>;

  using background_fields = tmpl::conditional_t<
      BackgroundGeometry == Geometry::FlatCartesian, tmpl::list<>,
      tmpl::list<
          gr::Tags::InverseSpatialMetric<DataVector, Dim>,
          gr::Tags::SpatialChristoffelSecondKindContracted<DataVector, Dim>>>;
  using inv_metric_tag =
      tmpl::conditional_t<BackgroundGeometry == Geometry::FlatCartesian, void,
                          gr::Tags::InverseSpatialMetric<DataVector, Dim>>;

  using fluxes_computer = Fluxes<Dim, BackgroundGeometry>;
  using sources_computer =
      tmpl::conditional_t<BackgroundGeometry == Geometry::FlatCartesian, void,
                          Sources<Dim, BackgroundGeometry>>;

  using boundary_conditions_base =
      elliptic::BoundaryConditions::BoundaryCondition<Dim>;
};
}  // namespace Poisson
