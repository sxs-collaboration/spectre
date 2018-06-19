// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace NewtonianEuler {

template <size_t Dim, typename DataType>
void conservative_from_primitive(
    const gsl::not_null<tnsr::I<DataType, Dim>*> momentum_density,
    const gsl::not_null<Scalar<DataType>*> energy_density,
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim>& velocity,
    const Scalar<DataType>& specific_internal_energy) noexcept {
  for (size_t i = 0; i < Dim; ++i) {
    momentum_density->get(i) = get(mass_density) * velocity.get(i);
  }

  get(*energy_density) =
      get(mass_density) * (0.5 * get(dot_product(velocity, velocity)) +
                           get(specific_internal_energy));
}

}  // namespace NewtonianEuler

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                  \
  template void NewtonianEuler::conservative_from_primitive(                  \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data)>*> momentum_density, \
      const gsl::not_null<Scalar<DTYPE(data)>*> energy_density,               \
      const Scalar<DTYPE(data)>& mass_density,                                \
      const tnsr::I<DTYPE(data), DIM(data)>& velocity,                        \
      const Scalar<DTYPE(data)>& specific_internal_energy) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector))

#undef DIM
#undef DTYPE
#undef INSTANTIATE
/// \endcond
