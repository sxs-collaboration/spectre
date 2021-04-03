// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/MachNumber.hpp"

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace NewtonianEuler {
template <typename DataType, size_t Dim, typename Fr>
void mach_number(const gsl::not_null<Scalar<DataType>*> result,
                 const tnsr::I<DataType, Dim, Fr>& velocity,
                 const Scalar<DataType>& sound_speed) noexcept {
  destructive_resize_components(result, get_size(get(sound_speed)));
  get(*result) = get(magnitude(velocity)) / get(sound_speed);
}

template <typename DataType, size_t Dim, typename Fr>
Scalar<DataType> mach_number(const tnsr::I<DataType, Dim, Fr>& velocity,
                             const Scalar<DataType>& sound_speed) noexcept {
  Scalar<DataType> result{};
  mach_number(make_not_null(&result), velocity, sound_speed);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                  \
  template void mach_number(const gsl::not_null<Scalar<DTYPE(data)>*> result, \
                            const tnsr::I<DTYPE(data), DIM(data)>& velocity,  \
                            const Scalar<DTYPE(data)>& sound_speed) noexcept; \
  template Scalar<DTYPE(data)> mach_number(                                   \
      const tnsr::I<DTYPE(data), DIM(data)>& velocity,                        \
      const Scalar<DTYPE(data)>& sound_speed) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector), (1, 2, 3))

#undef INSTANTIATE
#undef DIM
#undef DTYPE
}  // namespace NewtonianEuler
