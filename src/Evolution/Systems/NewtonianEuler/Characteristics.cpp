// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace NewtonianEuler {

template <size_t Dim>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, Dim + 2>*> char_speeds,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& sound_speed,
    const tnsr::i<DataVector, Dim>& normal) noexcept {
  auto& characteristic_speeds = *char_speeds;
  characteristic_speeds =
      make_array<Dim + 2>(DataVector(get(dot_product(velocity, normal))));

  characteristic_speeds[0] -= get(sound_speed);
  characteristic_speeds[Dim + 1] += get(sound_speed);
}

template <size_t Dim>
std::array<DataVector, Dim + 2> characteristic_speeds(
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& sound_speed,
    const tnsr::i<DataVector, Dim>& normal) noexcept {
  std::array<DataVector, Dim + 2> char_speeds{};
  characteristic_speeds(make_not_null(&char_speeds), velocity, sound_speed,
                        normal);
  return char_speeds;
}

}  // namespace NewtonianEuler

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void NewtonianEuler::characteristic_speeds(                         \
      const gsl::not_null<std::array<DataVector, DIM(data) + 2>*> char_speeds, \
      const tnsr::I<DataVector, DIM(data)>& velocity,                          \
      const Scalar<DataVector>& sound_speed,                                   \
      const tnsr::i<DataVector, DIM(data)>& normal) noexcept;                  \
  template std::array<DataVector, DIM(data) + 2>                               \
  NewtonianEuler::characteristic_speeds(                                       \
      const tnsr::I<DataVector, DIM(data)>& velocity,                          \
      const Scalar<DataVector>& sound_speed,                                   \
      const tnsr::i<DataVector, DIM(data)>& normal) noexcept;                  \
  template struct NewtonianEuler::ComputeLargestCharacteristicSpeed<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond
