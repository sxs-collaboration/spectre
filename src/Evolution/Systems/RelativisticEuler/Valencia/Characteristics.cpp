// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RelativisticEuler/Valencia/Characteristics.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace RelativisticEuler {
namespace Valencia {

template <size_t Dim>
std::array<DataVector, Dim + 2> characteristic_speeds(
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::I<DataVector, Dim>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::i<DataVector, Dim>& normal) noexcept {
  auto characteristic_speeds =
      make_array<Dim + 2>(DataVector(-1.0 * get(dot_product(normal, shift))));

  // Dim-fold degenerate eigenvalue
  const DataVector normal_velocity = get(dot_product(normal, spatial_velocity));
  DataVector lapse_times_normal_velocity = get(lapse) * normal_velocity;
  for (size_t i = 1; i < Dim + 1; ++i) {
    gsl::at(characteristic_speeds, i) += lapse_times_normal_velocity;
  }

  const DataVector one_minus_v_sqrd_cs_sqrd =
      1.0 - get(spatial_velocity_squared) * get(sound_speed_squared);
  const DataVector vn_times_one_minus_cs_sqrd =
      normal_velocity * (1.0 - get(sound_speed_squared));

  DataVector first_term = get(lapse) / one_minus_v_sqrd_cs_sqrd;
  DataVector& second_term = lapse_times_normal_velocity;
  second_term = first_term * sqrt(get(sound_speed_squared)) *
                sqrt((1.0 - get(spatial_velocity_squared)) *
                     (one_minus_v_sqrd_cs_sqrd -
                      normal_velocity * vn_times_one_minus_cs_sqrd));
  first_term *= vn_times_one_minus_cs_sqrd;

  characteristic_speeds[0] += first_term - second_term;
  characteristic_speeds[Dim + 1] += first_term + second_term;

  return characteristic_speeds;
}

}  // namespace Valencia
}  // namespace RelativisticEuler

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                  \
  template std::array<DataVector, DIM(data) + 2>              \
  RelativisticEuler::Valencia::characteristic_speeds(         \
      const Scalar<DataVector>& lapse,                        \
      const tnsr::I<DataVector, DIM(data)>& shift,            \
      const tnsr::I<DataVector, DIM(data)>& spatial_velocity, \
      const Scalar<DataVector>& spatial_velocity_squared,     \
      const Scalar<DataVector>& sound_speed_squared,          \
      const tnsr::i<DataVector, DIM(data)>& normal) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond
