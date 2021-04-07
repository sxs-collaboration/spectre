// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/KineticEnergyDensity.hpp"

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianEuler {
template <typename DataType, size_t Dim, typename Fr>
void kinetic_energy_density(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim, Fr>& velocity) noexcept {
  destructive_resize_components(result, get_size(get(mass_density)));
  get(*result) = 0.5 * get(mass_density) * get(dot_product(velocity, velocity));
}

template <typename DataType, size_t Dim, typename Fr>
Scalar<DataType> kinetic_energy_density(
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim, Fr>& velocity) noexcept {
  Scalar<DataType> result{};
  kinetic_energy_density(make_not_null(&result), mass_density, velocity);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                     \
  template void kinetic_energy_density(                          \
      const gsl::not_null<Scalar<DTYPE(data)>*> result,          \
      const Scalar<DTYPE(data)>& mass_density,                   \
      const tnsr::I<DTYPE(data), DIM(data)>& velocity) noexcept; \
  template Scalar<DTYPE(data)> kinetic_energy_density(           \
      const Scalar<DTYPE(data)>& mass_density,                   \
      const tnsr::I<DTYPE(data), DIM(data)>& velocity) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector), (1, 2, 3))

#undef INSTANTIATE
#undef DIM
#undef DTYPE
}  // namespace NewtonianEuler
