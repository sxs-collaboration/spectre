// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
/// \endcond

namespace hydro {
/*!
 * \ingroup EquationsOfStateGroup
 * \brief Computes the relativistic sound speed squared
 *
 * The relativistic sound speed squared is given by
 * \f$c_s^2 = \left(\chi + p\kappa / \rho^2\right)/h\f$, where
 * \f$p\f$ is the fluid pressure, \f$\rho\f$ is the rest mass density,
 * \f$h = 1 + \epsilon + p / \rho\f$ is the specific enthalpy
 * \f$\chi = (\partial p/\partial\rho)_\epsilon\f$ and
 * \f$\kappa = (\partial p/ \partial \epsilon)_\rho\f$, where
 * \f$\epsilon\f$ is the specific internal energy.
 */
template <typename DataType, size_t ThermodynamicDim>
Scalar<DataType> sound_speed_squared(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& specific_enthalpy,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) noexcept;

namespace Tags {
/// Compute item for the sound speed squared \f$c_s^2\f$.
/// \see hydro::sound_speed_squared
///
/// Can be retrieved using `hydro::Tags::SoundSpeedSquared`
template <typename DataType>
struct SoundSpeedSquaredCompute : SoundSpeedSquared<DataType>, db::ComputeTag {
  template <typename EquationOfStateType>
  static Scalar<DataType> function(
      const Scalar<DataType>& rest_mass_density,
      const Scalar<DataType>& specific_internal_energy,
      const Scalar<DataType>& specific_enthalpy,
      const EquationOfStateType& equation_of_state) noexcept {
    return sound_speed_squared(rest_mass_density, specific_internal_energy,
                               specific_enthalpy, equation_of_state);
  }
  using argument_tags =
      tmpl::list<RestMassDensity<DataType>, SpecificInternalEnergy<DataType>,
                 SpecificEnthalpy<DataType>, hydro::Tags::EquationOfStateBase>;
  using base = SoundSpeedSquared<DataType>;
};
}  // namespace Tags
}  // namespace hydro
