// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianEuler {
template <typename DataType, size_t ThermodynamicDim>
void sound_speed_squared(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) {
  destructive_resize_components(result, get_size(get(mass_density)));
  if constexpr (ThermodynamicDim == 1) {
    get(*result) =
        get(equation_of_state.chi_from_density(mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            mass_density));
  } else if constexpr (ThermodynamicDim == 2) {
    get(*result) =
        get(equation_of_state.chi_from_density_and_energy(
            mass_density, specific_internal_energy)) +
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    mass_density, specific_internal_energy));
  }
}

template <typename DataType, size_t ThermodynamicDim>
Scalar<DataType> sound_speed_squared(
    const Scalar<DataType>& mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) {
  Scalar<DataType> result{};
  sound_speed_squared(make_not_null(&result), mass_density,
                      specific_internal_energy, equation_of_state);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                            \
  template void sound_speed_squared(                                    \
      const gsl::not_null<Scalar<DTYPE(data)>*> result,                 \
      const Scalar<DTYPE(data)>& mass_density,                          \
      const Scalar<DTYPE(data)>& specific_internal_energy,              \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>& \
          equation_of_state);                                           \
  template Scalar<DTYPE(data)> sound_speed_squared(                     \
      const Scalar<DTYPE(data)>& mass_density,                          \
      const Scalar<DTYPE(data)>& specific_internal_energy,              \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>& \
          equation_of_state);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector), (1, 2))

#undef INSTANTIATE
#undef THERMO_DIM
#undef DTYPE
}  // namespace NewtonianEuler
