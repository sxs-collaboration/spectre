// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace hydro {
template <typename DataType>
void relativistic_specific_enthalpy(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& pressure) noexcept {
  destructive_resize_components(result, get_size(get(rest_mass_density)));
  get(*result) = 1.0 + get(specific_internal_energy) +
                 get(pressure) / get(rest_mass_density);
}

template <typename DataType>
Scalar<DataType> relativistic_specific_enthalpy(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const Scalar<DataType>& pressure) noexcept {
  Scalar<DataType> result{};
  relativistic_specific_enthalpy(make_not_null(&result), rest_mass_density,
                                 specific_internal_energy, pressure);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                   \
  template void relativistic_specific_enthalpy(                \
      const gsl::not_null<Scalar<DTYPE(data)>*> result,        \
      const Scalar<DTYPE(data)>& rest_mass_density,            \
      const Scalar<DTYPE(data)>& specific_internal_energy,     \
      const Scalar<DTYPE(data)>& pressure) noexcept;           \
  template Scalar<DTYPE(data)> relativistic_specific_enthalpy( \
      const Scalar<DTYPE(data)>& rest_mass_density,            \
      const Scalar<DTYPE(data)>& specific_internal_energy,     \
      const Scalar<DTYPE(data)>& pressure) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
}  // namespace hydro
