// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RelativisticEuler/Valencia/Characteristics.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

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
  // Allocating a single large buffer is much faster than many small buffers
  Variables<
      tmpl::list<Tags::TempScalar<0>, Tags::TempScalar<1>, Tags::TempScalar<2>,
                 Tags::TempScalar<3>, Tags::TempScalar<4>>>
      temp_tensors{num_grid_points};

  // Because we don't require char_speeds to be of the correct size we use a
  // temp buffer for the dot product, then multiply by -1 assigning the result
  // to char_speeds.
  {
    Scalar<DataVector>& normal_shift =
        get<Tags::TempScalar<0>>(temp_tensors);
    dot_product(make_not_null(&normal_shift), normal, shift);
    (*char_speeds)[0] = -1.0 * get(normal_shift);
  }
  // Dim-fold degenerate eigenvalue, reuse normal_shift allocation
  Scalar<DataVector>& normal_velocity =
      get<Tags::TempScalar<0>>(temp_tensors);
  dot_product(make_not_null(&normal_velocity), normal, spatial_velocity);
  (*char_speeds)[1] = (*char_speeds)[0] + get(lapse) * get(normal_velocity);
  for (size_t i = 2; i < Dim + 1; ++i) {
    gsl::at(*char_speeds, i) = (*char_speeds)[1];
  }

  Scalar<DataVector>& one_minus_v_sqrd_cs_sqrd =
      get<Tags::TempScalar<1>>(temp_tensors);
  get(one_minus_v_sqrd_cs_sqrd) =
      1.0 - get(spatial_velocity_squared) * get(sound_speed_squared);
  Scalar<DataVector>& vn_times_one_minus_cs_sqrd =
      get<Tags::TempScalar<2>>(temp_tensors);
  get(vn_times_one_minus_cs_sqrd) =
      get(normal_velocity) * (1.0 - get(sound_speed_squared));

  Scalar<DataVector>& first_term = get<Tags::TempScalar<3>>(temp_tensors);
  get(first_term) = get(lapse) / get(one_minus_v_sqrd_cs_sqrd);
  Scalar<DataVector>& second_term = get<Tags::TempScalar<4>>(temp_tensors);
  get(second_term) =
      get(first_term) * sqrt(get(sound_speed_squared)) *
      sqrt((1.0 - get(spatial_velocity_squared)) *
           (get(one_minus_v_sqrd_cs_sqrd) -
            get(normal_velocity) * get(vn_times_one_minus_cs_sqrd)));
  get(first_term) *= get(vn_times_one_minus_cs_sqrd);

  (*char_speeds)[Dim + 1] =
      (*char_speeds)[0] + get(first_term) + get(second_term);
  (*char_speeds)[0] += get(first_term) - get(second_term);
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
