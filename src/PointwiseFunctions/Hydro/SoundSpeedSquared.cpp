// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/SoundSpeedSquared.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Overloader.hpp"

/// \cond
namespace hydro {

template <typename DataType, size_t ThermodynamicDim>
Scalar<DataType> sound_speed_squared(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& specific_enthalpy,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) noexcept {
  DataType sound_speed_squared{};
  make_overloader(
      [&rest_mass_density,
       &sound_speed_squared ](const EquationsOfState::EquationOfState<true, 1>&
                                  the_equation_of_state) noexcept {
        sound_speed_squared =
            get(the_equation_of_state.chi_from_density(rest_mass_density)) +
            get(the_equation_of_state
                    .kappa_times_p_over_rho_squared_from_density(
                        rest_mass_density));
      },
      [&rest_mass_density, &specific_internal_energy,
       &sound_speed_squared ](const EquationsOfState::EquationOfState<true, 2>&
                                  the_equation_of_state) noexcept {
        sound_speed_squared =
            get(the_equation_of_state.chi_from_density_and_energy(
                rest_mass_density, specific_internal_energy)) +
            get(the_equation_of_state
                    .kappa_times_p_over_rho_squared_from_density_and_energy(
                        rest_mass_density, specific_internal_energy));
      })(equation_of_state);
  sound_speed_squared /= get(specific_enthalpy);
  return Scalar<DataType>{sound_speed_squared};
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                           \
  template Scalar<DTYPE(data)> sound_speed_squared(                    \
      const Scalar<DTYPE(data)>& rest_mass_density,                    \
      const Scalar<DTYPE(data)>& specific_internal_energy,             \
      const Scalar<DTYPE(data)>& specific_enthalpy,                    \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>& \
          equation_of_state) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector), (1, 2))

#undef DTYPE
#undef THERMO_DIM
#undef INSTANTIATE
}  // namespace hydro
/// \endcond
