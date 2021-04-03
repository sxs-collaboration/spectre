// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/SoundSpeedSquared.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace hydro {

template <typename DataType, size_t ThermodynamicDim>
void sound_speed_squared(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& specific_enthalpy,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) noexcept {
  destructive_resize_components(result, get_size(get(rest_mass_density)));
  if constexpr (ThermodynamicDim == 1) {
    get(*result) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
  } else if constexpr (ThermodynamicDim == 2) {
    get(*result) =
        get(equation_of_state.chi_from_density_and_energy(
            rest_mass_density, specific_internal_energy)) +
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
  }
  get(*result) /= get(specific_enthalpy);
}

template <typename DataType, size_t ThermodynamicDim>
Scalar<DataType> sound_speed_squared(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& specific_enthalpy,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) noexcept {
  Scalar<DataType> result{};
  sound_speed_squared(make_not_null(&result), rest_mass_density,
                      specific_internal_energy, specific_enthalpy,
                      equation_of_state);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                           \
  template void sound_speed_squared(                                   \
      const gsl::not_null<Scalar<DTYPE(data)>*> result,                \
      const Scalar<DTYPE(data)>& rest_mass_density,                    \
      const Scalar<DTYPE(data)>& specific_internal_energy,             \
      const Scalar<DTYPE(data)>& specific_enthalpy,                    \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>& \
          equation_of_state) noexcept;                                 \
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
