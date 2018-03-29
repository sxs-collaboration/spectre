// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;

namespace Tags {
template <typename>
struct Fluxes;
}  // namespace Tags

namespace NewtonianEuler {
struct MassDensity;
template <size_t Dim>
struct MomentumDensity;
struct EnergyDensity;
template <size_t Dim>
struct Velocity;
struct Pressure;
}  // namespace NewtonianEuler
/// \endcond

namespace NewtonianEuler {

/*!
 * \brief Compute the fluxes of the conservative variables of the
 * Newtonian Euler system
 *
 * The fluxes are \f$(\text{Dim} + 2)\f$ vectors of
 * dimension \f$\text{Dim}\f$. Denoting the flux of the conservative
 * variable \f$u\f$ as \f$F(u)\f$, one has
 *
 * \f{align*}
 * F^i(\rho) &= S^i\\
 * F^i(S^j) &= S^i v^j + \delta^{ij}p\\
 * F^i(e) &= (e + p)v^i
 * \f}
 *
 * where \f$S^i\f$ is the momentum density, \f$e\f$ is the energy density,
 * \f$v^i\f$ is the velocity, \f$p\f$ is the pressure, and \f$\delta^{ij}\f$
 * is the Kronecker delta. This form of the fluxes combines conservative and
 * primitive variables (while the velocity appears explicitly, the pressure
 * implicitly depends, for instance, on the mass density and the specific
 * internal energy), so the conversion from one variable set to the other
 * must be known prior to calling this function.
 */
template <size_t Dim>
struct ComputeFluxes {
  using return_tags =
      tmpl::list<Tags::Fluxes<MassDensity>, Tags::Fluxes<MomentumDensity<Dim>>,
                 Tags::Fluxes<EnergyDensity>>;

  using argument_tags =
      tmpl::list<MomentumDensity<Dim>, EnergyDensity, Velocity<Dim>, Pressure>;

  static void apply(
      gsl::not_null<tnsr::I<DataVector, Dim>*> mass_density_flux,
      gsl::not_null<tnsr::IJ<DataVector, Dim>*> momentum_density_flux,
      gsl::not_null<tnsr::I<DataVector, Dim>*> energy_density_flux,
      const tnsr::I<DataVector, Dim>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const tnsr::I<DataVector, Dim>& velocity,
      const Scalar<DataVector>& pressure) noexcept;
};

}  // namespace NewtonianEuler
