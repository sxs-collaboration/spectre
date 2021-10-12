// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/NewtonianEuler/TagsDeclarations.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
class DataVector;

namespace NewtonianEuler {
namespace Solutions {
template <size_t Dim>
struct IsentropicVortex;
}  // namespace Solutions
}  // namespace NewtonianEuler

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace NewtonianEuler {
namespace Sources {

/*!
 * \brief Source generating a modified isentropic vortex.
 *
 * If Solutions::IsentropicVortex is modifed so that the flow velocity along
 * the \f$z-\f$axis is not a constant but a function of \f$z\f$, the new vortex
 * will be a solution to the 3-D Newtonian Euler equations with a source term,
 *
 * \f{align*}
 * \partial_t\rho + \partial_i F^i(\rho) &= S(\rho)\\
 * \partial_t S^i + \partial_j F^{j}(S^i) &= S(S^i)\\
 * \partial_t e + \partial_i F^i(e) &= S(e),
 * \f}
 *
 * where \f$F^i(u)\f$ is the volume flux of the conserved quantity \f$u\f$
 * (see ComputeFluxes), and
 *
 * \f{align*}
 * S(\rho) &= \rho \dfrac{dv_z}{dz}\\
 * S(S_x) &= S_x \dfrac{dv_z}{dz}\\
 * S(S_y) &= S_y \dfrac{dv_z}{dz}\\
 * S(S_z) &= 2S_z \dfrac{dv_z}{dz}\\
 * S(e) &= \left(e + p + v_z S_z\right)\dfrac{dv_z}{dz},
 * \f}
 *
 * where \f$\rho\f$ is the mass density of the vortex, \f$S_i\f$ is
 * its momentum density, \f$e\f$ is its energy density,
 * \f$v_z = v_z(z)\f$ is the \f$z-\f$component of its velocity,
 * and \f$p\f$ is its pressure. These quantities are readily obtained
 * from the primitive variables, whose expressions are those in
 * Solutions::IsentropicVortex
 */
template <size_t Dim>
struct VortexPerturbation {
  VortexPerturbation() = default;
  VortexPerturbation(const VortexPerturbation& /*rhs*/) = default;
  VortexPerturbation& operator=(const VortexPerturbation& /*rhs*/) = default;
  VortexPerturbation(VortexPerturbation&& /*rhs*/) = default;
  VortexPerturbation& operator=(VortexPerturbation&& /*rhs*/) = default;
  ~VortexPerturbation() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}

  using sourced_variables = tmpl::conditional_t<
      Dim == 3,
      tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<Dim>,
                 Tags::EnergyDensity>,
      tmpl::list<>>;

  using argument_tags = tmpl::conditional_t<
      Dim == 3,
      tmpl::list<::Tags::AnalyticSolution<
                     NewtonianEuler::Solutions::IsentropicVortex<Dim>>,
                 domain::Tags::Coordinates<3, Frame::Inertial>, ::Tags::Time>,
      tmpl::list<>>;

  // Overload required for 2d simulations, where no variable is sourced.
  void apply() const;

  // Function to be used in 3d.
  void apply(gsl::not_null<Scalar<DataVector>*> source_mass_density_cons,
             gsl::not_null<tnsr::I<DataVector, Dim>*> source_momentum_density,
             gsl::not_null<Scalar<DataVector>*> source_energy_density,
             const NewtonianEuler::Solutions::IsentropicVortex<Dim>& vortex,
             const tnsr::I<DataVector, Dim>& x, double time) const;
};
}  // namespace Sources
}  // namespace NewtonianEuler
