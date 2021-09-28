// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/SpecificKineticEnergy.hpp"

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace NewtonianEuler {
template <typename DataType, size_t Dim, typename Fr>
void specific_kinetic_energy(const gsl::not_null<Scalar<DataType>*> result,
                             const tnsr::I<DataType, Dim, Fr>& velocity) {
  destructive_resize_components(result, get_size(get<0>(velocity)));
  get(*result) = 0.5 * get(dot_product(velocity, velocity));
}

template <typename DataType, size_t Dim, typename Fr>
Scalar<DataType> specific_kinetic_energy(
    const tnsr::I<DataType, Dim, Fr>& velocity) {
  Scalar<DataType> result{};
  specific_kinetic_energy(make_not_null(&result), velocity);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                            \
  template void specific_kinetic_energy(                \
      const gsl::not_null<Scalar<DTYPE(data)>*> result, \
      const tnsr::I<DTYPE(data), DIM(data)>& velocity); \
  template Scalar<DTYPE(data)> specific_kinetic_energy( \
      const tnsr::I<DTYPE(data), DIM(data)>& velocity);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector), (1, 2, 3))

#undef INSTANTIATE
#undef DIM
#undef DTYPE
}  // namespace NewtonianEuler
