// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeArray.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace NewtonianEuler {

template <size_t Dim>
std::array<DataVector, Dim + 2> characteristic_speeds(
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::i<DataVector, Dim>& normal) noexcept {
  const DataVector sound_speed = sqrt(get(sound_speed_squared));

  auto characteristic_speeds =
      make_array<Dim + 2>(DataVector(get(dot_product(velocity, normal))));
  characteristic_speeds[0] -= sound_speed;
  characteristic_speeds[Dim + 1] += sound_speed;

  return characteristic_speeds;
}

}  // namespace NewtonianEuler

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                          \
  template std::array<DataVector, DIM(data) + 2>      \
  NewtonianEuler::characteristic_speeds(              \
      const tnsr::I<DataVector, DIM(data)>& velocity, \
      const Scalar<DataVector>& sound_speed_squared,  \
      const tnsr::i<DataVector, DIM(data)>& normal) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond
