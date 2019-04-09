// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare hydro::Tags::EquationOfState

namespace NewtonianEuler {
/// Computes the Newtonian sound speed squared
/// \f$c_s^2 = \chi + p\kappa / \rho^2\f$, where
/// \f$p\f$ is the fluid pressure, \f$\rho\f$ is the mass density,
/// \f$\chi = (\partial p/\partial\rho)_\epsilon\f$ and
/// \f$\kappa = (\partial p/ \partial \epsilon)_\rho\f$, where
/// \f$\epsilon\f$ is the specific internal energy.
template <typename DataType, size_t ThermodynamicDim>
Scalar<DataType> sound_speed_squared(
    const Scalar<DataType>& mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) noexcept;

namespace Tags {
/// Compute item for the sound speed squared \f$c_s^2\f$.
/// \see NewtonianEuler::sound_speed_squared
///
/// Can be retrieved using `NewtonianEuler::Tags::SoundSpeedSquared`
template <typename DataType>
struct SoundSpeedSquaredCompute : SoundSpeedSquared<DataType>, db::ComputeTag {
  template <typename EquationOfStateType>
  static Scalar<DataType> function(
      const Scalar<DataType>& mass_density,
      const Scalar<DataType>& specific_internal_energy,
      const EquationOfStateType& equation_of_state) noexcept {
    return sound_speed_squared(mass_density, specific_internal_energy,
                               equation_of_state);
  }
  using argument_tags =
      tmpl::list<MassDensity<DataType>, SpecificInternalEnergy<DataType>,
                 hydro::Tags::EquationOfStateBase>;
};

/// Compute item for the sound speed \f$c_s\f$.
///
/// Can be retrieved using `NewtonianEuler::Tags::SoundSpeed`
template <typename DataType>
struct SoundSpeedCompute : SoundSpeed<DataType>, db::ComputeTag {
  static Scalar<DataType> function(
      const Scalar<DataType>& sound_speed_squared) noexcept {
    return Scalar<DataType>{sqrt(get(sound_speed_squared))};
  }
  using argument_tags = tmpl::list<SoundSpeedSquared<DataType>>;
};
}  // namespace Tags
}  // namespace NewtonianEuler
