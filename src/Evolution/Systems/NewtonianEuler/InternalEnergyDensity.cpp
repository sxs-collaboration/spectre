// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/InternalEnergyDensity.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianEuler {
template <typename DataType>
void internal_energy_density(const gsl::not_null<Scalar<DataType>*> result,
                             const Scalar<DataType>& mass_density,
                             const Scalar<DataType>& specific_internal_energy) {
  destructive_resize_components(result, get_size(get(mass_density)));
  get(*result) = get(mass_density) * get(specific_internal_energy);
}

template <typename DataType>
Scalar<DataType> internal_energy_density(
    const Scalar<DataType>& mass_density,
    const Scalar<DataType>& specific_internal_energy) {
  Scalar<DataType> result{};
  internal_energy_density(make_not_null(&result), mass_density,
                          specific_internal_energy);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                \
  template void internal_energy_density(                    \
      const gsl::not_null<Scalar<DTYPE(data)>*> result,     \
      const Scalar<DTYPE(data)>& mass_density,              \
      const Scalar<DTYPE(data)>& specific_internal_energy); \
  template Scalar<DTYPE(data)> internal_energy_density(     \
      const Scalar<DTYPE(data)>& mass_density,              \
      const Scalar<DTYPE(data)>& specific_internal_energy);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef INSTANTIATE
#undef DTYPE
}  // namespace NewtonianEuler
