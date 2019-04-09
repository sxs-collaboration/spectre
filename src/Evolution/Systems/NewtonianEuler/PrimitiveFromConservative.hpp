// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::EnergyDensity
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::MassDensity
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::MassDensityCons
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::MomentumDensity
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::Pressure
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::SpecificInternalEnergy
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::Velocity
// IWYU pragma: no_forward_declare hydro::Tags::EquationOfStateBase

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;
/// \endcond

// IWYU pragma: no_forward_declare Tensor

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
template <size_t Dim, size_t ThermodynamicDim>
struct PrimitiveFromConservative {
  using return_tags =
      tmpl::list<Tags::MassDensity<DataVector>, Tags::Velocity<DataVector, Dim>,
                 Tags::SpecificInternalEnergy<DataVector>,
                 Tags::Pressure<DataVector>>;

  using argument_tags = tmpl::list<
      Tags::MassDensityCons<DataVector>, Tags::MomentumDensity<DataVector, Dim>,
      Tags::EnergyDensity<DataVector>, hydro::Tags::EquationOfStateBase>;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim>*> velocity,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<Scalar<DataVector>*> pressure,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, Dim>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state) noexcept;
};
}  // namespace NewtonianEuler
