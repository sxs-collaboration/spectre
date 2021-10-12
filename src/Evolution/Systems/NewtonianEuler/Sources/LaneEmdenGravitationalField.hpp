// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/NewtonianEuler/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace NewtonianEuler {
namespace Solutions {
struct LaneEmdenStar;
}  // namespace Solutions
}  // namespace NewtonianEuler

namespace PUP {
class er;
}  // namespace PUP

namespace Tags {
template <typename SolutionType>
struct AnalyticSolution;
}  // namespace Tags

namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace Tags
}  // namespace domain

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

// IWYU pragma: no_include "Evolution/Systems/NewtonianEuler/Tags.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace NewtonianEuler {
namespace Sources {

/*!
 * \brief Source giving the acceleration due to gravity in the spherical,
 * Newtonian Lane-Emden star solution.
 *
 * The gravitational field \f$g^i\f$ enters the NewtonianEuler system as source
 * terms for the conserved momentum and energy:
 *
 * \f{align*}
 * \partial_t S^i + \partial_j F^{j}(S^i) &= S(S^i) = \rho g^i
 * \partial_t e + \partial_j F^{j}(e) &= S(e) = S_i g^i,
 * \f}
 *
 * where \f$S^i\f$ is the conserved momentum density, \f$e\f$ is the conserved
 * energy, \f$F^{j}(u)\f$ is the flux of the conserved quantity \f$u\f$, and
 * \f$\rho\f$ is the fluid mass density.
 *
 * \note This source is specialized to the Lane-Emden solution because it
 * queries a LaneEmdenStar analytic solution for the gravitational field that
 * generates the fluid acceleration. This source does *not* integrate the fluid
 * density to compute a self-consistent gravitational field (i.e., as if one
 * were solving a coupled Euler + Poisson system).
 */
struct LaneEmdenGravitationalField {
  LaneEmdenGravitationalField() = default;
  LaneEmdenGravitationalField(const LaneEmdenGravitationalField& /*rhs*/) =
      default;
  LaneEmdenGravitationalField& operator=(
      const LaneEmdenGravitationalField& /*rhs*/) = default;
  LaneEmdenGravitationalField(LaneEmdenGravitationalField&& /*rhs*/) = default;
  LaneEmdenGravitationalField& operator=(
      LaneEmdenGravitationalField&& /*rhs*/) = default;
  ~LaneEmdenGravitationalField() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}

  using sourced_variables =
      tmpl::list<Tags::MomentumDensity<3>, Tags::EnergyDensity>;

  using argument_tags = tmpl::list<
      Tags::MassDensityCons, Tags::MomentumDensity<3>,
      ::Tags::AnalyticSolution<NewtonianEuler::Solutions::LaneEmdenStar>,
      domain::Tags::Coordinates<3, Frame::Inertial>>;

  static void apply(
      gsl::not_null<tnsr::I<DataVector, 3>*> source_momentum_density,
      gsl::not_null<Scalar<DataVector>*> source_energy_density,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, 3>& momentum_density,
      const NewtonianEuler::Solutions::LaneEmdenStar& star,
      const tnsr::I<DataVector, 3>& x);
};

}  // namespace Sources
}  // namespace NewtonianEuler
