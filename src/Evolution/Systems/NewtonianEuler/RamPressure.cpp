// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/RamPressure.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianEuler {
template <typename DataType, size_t Dim, typename Fr>
void ram_pressure(const gsl::not_null<tnsr::II<DataType, Dim, Fr>*> result,
                  const Scalar<DataType>& mass_density,
                  const tnsr::I<DataType, Dim, Fr>& velocity) {
  destructive_resize_components(result, get_size(get(mass_density)));
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      result->get(i, j) = get(mass_density) * velocity.get(i) * velocity.get(j);
    }
  }
}

template <typename DataType, size_t Dim, typename Fr>
tnsr::II<DataType, Dim, Fr> ram_pressure(
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim, Fr>& velocity) {
  tnsr::II<DataType, Dim, Fr> result{};
  ram_pressure(make_not_null(&result), mass_density, velocity);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                         \
  template void ram_pressure(                                        \
      const gsl::not_null<tnsr::II<DTYPE(data), DIM(data)>*> result, \
      const Scalar<DTYPE(data)>& mass_density,                       \
      const tnsr::I<DTYPE(data), DIM(data)>& velocity);              \
  template tnsr::II<DTYPE(data), DIM(data)> ram_pressure(            \
      const Scalar<DTYPE(data)>& mass_density,                       \
      const tnsr::I<DTYPE(data), DIM(data)>& velocity);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector), (1, 2, 3))

#undef INSTANTIATE
#undef DIM
#undef DTYPE
}  // namespace NewtonianEuler
