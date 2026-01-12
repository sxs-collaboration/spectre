// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/SpecificEntropy.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace hydro {

template <typename DataType, size_t ThermodynamicDim>
void specific_entropy(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& temperature,
    const Scalar<DataType>& electron_fraction,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  if constexpr (ThermodynamicDim == 1) {
    get(*result) =
        get(equation_of_state.specific_entropy_from_density(rest_mass_density));
  } else if constexpr (ThermodynamicDim == 2) {
    get(*result) =
        get(equation_of_state.specific_entropy_from_density_and_temperature(
            rest_mass_density, temperature));
  } else if constexpr (ThermodynamicDim == 3) {
    get(*result) =
        get(equation_of_state.specific_entropy_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction));
  }
}

template <typename DataType, size_t ThermodynamicDim>
Scalar<DataType> specific_entropy(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& temperature,
    const Scalar<DataType>& electron_fraction,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  Scalar<DataType> result{};
  specific_entropy(make_not_null(&result), rest_mass_density, temperature,
                   electron_fraction, equation_of_state);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                           \
  template void specific_entropy(                                      \
      const gsl::not_null<Scalar<DTYPE(data)>*> result,                \
      const Scalar<DTYPE(data)>& rest_mass_density,                    \
      const Scalar<DTYPE(data)>& temperature,                          \
      const Scalar<DTYPE(data)>& electron_fraction,                    \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>& \
          equation_of_state);                                          \
  template Scalar<DTYPE(data)> specific_entropy(                       \
      const Scalar<DTYPE(data)>& rest_mass_density,                    \
      const Scalar<DTYPE(data)>& temperature,                          \
      const Scalar<DTYPE(data)>& electron_fraction,                    \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>& \
          equation_of_state);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector), (1, 2, 3))

#undef DTYPE
#undef THERMO_DIM
#undef INSTANTIATE
}  // namespace hydro
