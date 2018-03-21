// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/PrimitiveFromConservative.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace NewtonianEuler {

template <size_t Dim, typename DataType>
void velocity(const gsl::not_null<tnsr::I<DataType, Dim>*> velocity,
              const Scalar<DataType>& mass_density,
              const tnsr::I<DataType, Dim>& momentum_density) noexcept {
  const DataType one_over_mass_density = 1.0 / get(mass_density);
  for (size_t i = 0; i < Dim; ++i) {
    velocity->get(i) = momentum_density.get(i) * one_over_mass_density;
  }
}

template <size_t Dim, typename DataType>
tnsr::I<DataType, Dim> velocity(
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim>& momentum_density) noexcept {
  auto velocity = make_with_value<tnsr::I<DataType, Dim>>(mass_density, 0.0);
  NewtonianEuler::velocity(make_not_null(&velocity), mass_density,
                           momentum_density);
  return velocity;
}

template <size_t Dim, typename DataType>
void primitive_from_conservative(
    const gsl::not_null<tnsr::I<DataType, Dim>*> velocity,
    const gsl::not_null<Scalar<DataType>*> specific_internal_energy,
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim>& momentum_density,
    const Scalar<DataType>& energy_density) noexcept {
  NewtonianEuler::velocity(velocity, mass_density, momentum_density);
  get(*specific_internal_energy) = get(energy_density) / get(mass_density) -
                                   0.5 * get(dot_product(*velocity, *velocity));
}

}  // namespace NewtonianEuler

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                              \
  template void NewtonianEuler::velocity(                                 \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data)>*> velocity,     \
      const Scalar<DTYPE(data)>& mass_density,                            \
      const tnsr::I<DTYPE(data), DIM(data)>& momentum_density) noexcept;  \
  template tnsr::I<DTYPE(data), DIM(data)> NewtonianEuler::velocity(      \
      const Scalar<DTYPE(data)>& mass_density,                            \
      const tnsr::I<DTYPE(data), DIM(data)>& momentum_density) noexcept;  \
  template void NewtonianEuler::primitive_from_conservative(              \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data)>*> velocity,     \
      const gsl::not_null<Scalar<DTYPE(data)>*> specific_internal_energy, \
      const Scalar<DTYPE(data)>& mass_density,                            \
      const tnsr::I<DTYPE(data), DIM(data)>& momentum_density,            \
      const Scalar<DTYPE(data)>& energy_density) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector))

#undef DIM
#undef DTYPE
#undef INSTANTIATE
/// \endcond
