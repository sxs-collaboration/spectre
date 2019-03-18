// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare NewtonianEuler::Tags::EnergyDensity
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::MassDensity
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::MassDensityCons
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::MomentumDensity
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::SpecificInternalEnergy
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::Velocity

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
 * \brief Compute the conservative variables from the primitive variables.
 *
 * \f{align*}
 * S^i &= \rho v^i \\
 * e &= \dfrac{1}{2}\rho v^2 + \rho\epsilon
 * \f}
 *
 * where \f$S^i\f$ is the momentum density, \f$e\f$ is the energy density,
 * \f$\rho\f$ is the mass density, \f$v^i\f$ is the velocity, \f$v^2\f$ is its
 * magnitude squared, and \f$\epsilon\f$ is the specific internal energy.
 * In addition, this method returns the mass density as a conservative.
 */
template <size_t Dim>
struct ConservativeFromPrimitive {
  using return_tags = tmpl::list<Tags::MassDensityCons<DataVector>,
                                 Tags::MomentumDensity<DataVector, Dim>,
                                 Tags::EnergyDensity<DataVector>>;

  using argument_tags =
      tmpl::list<Tags::MassDensity<DataVector>, Tags::Velocity<DataVector, Dim>,
                 Tags::SpecificInternalEnergy<DataVector>>;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> mass_density_cons,
      gsl::not_null<tnsr::I<DataVector, Dim>*> momentum_density,
      gsl::not_null<Scalar<DataVector>*> energy_density,
      const Scalar<DataVector>& mass_density,
      const tnsr::I<DataVector, Dim>& velocity,
      const Scalar<DataVector>& specific_internal_energy) noexcept;
};
}  // namespace NewtonianEuler
