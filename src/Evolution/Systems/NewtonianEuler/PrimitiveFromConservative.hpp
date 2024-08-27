// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;
/// \endcond

namespace NewtonianEuler {

/*!
 * \brief Compute the primitive variables from the conservative variables.
 *
 * \f{align*}
 * v^i &= \frac{S^i}{\rho} \\
 * \epsilon &= \frac{e}{\rho} - \frac{1}{2}\frac{S^2}{\rho^2}
 * \f}
 *
 * where \f$v^i\f$ is the velocity, \f$\epsilon\f$ is the specific
 * internal energy, \f$e\f$ is the energy density, \f$\rho\f$
 * is the mass density, \f$S^i\f$ is the momentum density, and
 * \f$S^2\f$ is the momentum density squared.
 *
 * This routine also returns the mass density as a primitive, and the pressure
 * from a generic equation of state \f$p = p(\rho, \epsilon)\f$.
 */
template <size_t Dim>
struct PrimitiveFromConservative {
  using return_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::Pressure<DataVector>>;

  using argument_tags =
      tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<Dim>,
                 Tags::EnergyDensity, hydro::Tags::EquationOfState<false, 2>>;

  template <size_t ThermodynamicDim>
  static void apply(
      gsl::not_null<Scalar<DataVector>*> mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim>*> velocity,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<Scalar<DataVector>*> pressure,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, Dim>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state);
};
}  // namespace NewtonianEuler
