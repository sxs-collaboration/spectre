// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RelativisticEuler/Valencia/Characteristics.hpp"

#include <cstdlib>
#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace RelativisticEuler {
namespace Valencia {

template <size_t Dim>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, Dim + 2>*> char_speeds,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::I<DataVector, Dim>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::i<DataVector, Dim>& normal) noexcept {
  const size_t num_grid_points = get<0>(shift).size();
  constexpr size_t num_vectors = 5;
  // Allocating a single large buffer is much faster than many small buffers
  std::unique_ptr<double[], decltype(&free)> temp_vectors(
      static_cast<double*>(
          malloc(num_grid_points * sizeof(double) * num_vectors)),
      &free);

  // Because we don't require char_speeds to be of the correct size we use a
  // temp buffer for the dot product, then multiply by -1 assigning the result
  // to char_speeds.
  Scalar<DataVector> normal_velocity{temp_vectors.get(), num_grid_points};
  dot_product(make_not_null(&normal_velocity), normal, shift);
  (*char_speeds)[0] = -1.0 * get(normal_velocity);
  // Dim-fold degenerate eigenvalue
  dot_product(make_not_null(&normal_velocity), normal, spatial_velocity);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  DataVector lapse_times_normal_velocity(temp_vectors.get() + num_grid_points,
                                         num_grid_points);
  lapse_times_normal_velocity = get(lapse) * get(normal_velocity);
  (*char_speeds)[1] = (*char_speeds)[0] + lapse_times_normal_velocity;
  for (size_t i = 2; i < Dim + 1; ++i) {
    gsl::at(*char_speeds, i) = (*char_speeds)[1];
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  DataVector one_minus_v_sqrd_cs_sqrd(temp_vectors.get() + 2 * num_grid_points,
                                      num_grid_points);
  one_minus_v_sqrd_cs_sqrd =
      1.0 - get(spatial_velocity_squared) * get(sound_speed_squared);
  DataVector vn_times_one_minus_cs_sqrd(
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      temp_vectors.get() + 3 * num_grid_points, num_grid_points);
  vn_times_one_minus_cs_sqrd =
      get(normal_velocity) * (1.0 - get(sound_speed_squared));

  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  DataVector first_term(temp_vectors.get() + 4 * num_grid_points,
                        num_grid_points);
  first_term = get(lapse) / one_minus_v_sqrd_cs_sqrd;
  // Reuse allocation
  DataVector& second_term = lapse_times_normal_velocity;
  second_term = first_term * sqrt(get(sound_speed_squared)) *
                sqrt((1.0 - get(spatial_velocity_squared)) *
                     (one_minus_v_sqrd_cs_sqrd -
                      get(normal_velocity) * vn_times_one_minus_cs_sqrd));
  first_term *= vn_times_one_minus_cs_sqrd;

  (*char_speeds)[Dim + 1] = (*char_speeds)[0] + first_term + second_term;
  (*char_speeds)[0] += first_term - second_term;
}

template <size_t Dim>
std::array<DataVector, Dim + 2> characteristic_speeds(
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::I<DataVector, Dim>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::i<DataVector, Dim>& normal) noexcept {
  std::array<DataVector, Dim + 2> char_speeds{};
  characteristic_speeds(make_not_null(&char_speeds), lapse, shift,
                        spatial_velocity, spatial_velocity_squared,
                        sound_speed_squared, normal);
  return char_speeds;
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
