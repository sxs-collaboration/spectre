// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"

#include <cmath>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/SpinWeighted.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"

// IWYU pragma: no_forward_declare SpinWeighted

namespace Spectral {
namespace Swsh {

namespace detail {
template <ComplexRepresentation Representation>
void append_libsharp_collocation_pointers(
    const gsl::not_null<std::vector<double*>*> collocation_data,
    const gsl::not_null<std::vector<ComplexDataView<Representation>>*>
        collocation_views,
    const gsl::not_null<ComplexDataVector*> vector, const size_t l_max,
    const bool positive_spin) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_radial_points =
      vector->size() / number_of_angular_points;

  for (size_t i = 0; i < number_of_radial_points; ++i) {
    collocation_views->push_back(detail::ComplexDataView<Representation>{
        vector, number_of_angular_points, i * number_of_angular_points});
    collocation_data->push_back(collocation_views->back().real_data());
    // alteration needed because libsharp doesn't support negative spins
    if (not positive_spin) {
      collocation_views->back().conjugate();
    }
    collocation_data->push_back(collocation_views->back().imag_data());
  }
}

void append_libsharp_coefficient_pointers(
    const gsl::not_null<std::vector<std::complex<double>*>*> coefficient_data,
    const gsl::not_null<ComplexModalVector*> vector,
    const size_t l_max) noexcept {
  const size_t number_of_coefficients =
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max);
  const size_t number_of_radial_points =
      vector->size() / number_of_coefficients;

  for (size_t i = 0; i < number_of_radial_points; ++i) {
    // coefficients associated with the real part
    coefficient_data->push_back(vector->data() + i * number_of_coefficients);
    // coefficients associated with the imaginary part
    coefficient_data->push_back(vector->data() +
                                (2 * i + 1) * (number_of_coefficients / 2));
  }
}

template <ComplexRepresentation Representation>
void execute_libsharp_transform_set(
    const sharp_jobtype& jobtype, const int spin,
    const gsl::not_null<std::vector<std::complex<double>*>*> coefficient_data,
    const gsl::not_null<std::vector<double*>*> collocation_data,
    const gsl::not_null<const CollocationMetadata<Representation>*>
        collocation_metadata,
    const sharp_alm_info* alm_info, const size_t num_transforms) noexcept {
  // libsharp considers two arrays per transform when spin is not zero.
  const size_t number_of_arrays_per_transform = (spin == 0 ? 1 : 2);
  // libsharp has an internal flag for the maximum number of transforms, so if
  // we have more than max_libsharp_transforms, we have to do them in chunks
  // of max_libsharp_transforms.
  for (size_t transform_block = 0;
       transform_block <
       (num_transforms + max_libsharp_transforms - 1) / max_libsharp_transforms;
       ++transform_block) {
    if (transform_block < (num_transforms / max_libsharp_transforms)) {
      // clang-tidy cppcoreguidelines-pro-bounds-pointer-arithmetic
      sharp_execute(jobtype, abs(spin),
                    coefficient_data->data() +  // NOLINT
                        number_of_arrays_per_transform *
                            max_libsharp_transforms * transform_block,
                    collocation_data->data() +  // NOLINT
                        number_of_arrays_per_transform *
                            max_libsharp_transforms * transform_block,
                    collocation_metadata->get_sharp_geom_info(), alm_info,
                    max_libsharp_transforms, SHARP_DP, nullptr, nullptr);
    } else {
      // clang-tidy cppcoreguidelines-pro-bounds-pointer-arithmetic
      sharp_execute(jobtype, abs(spin),
                    coefficient_data->data() +  // NOLINT
                        number_of_arrays_per_transform *
                            max_libsharp_transforms * transform_block,
                    collocation_data->data() +  // NOLINT
                        number_of_arrays_per_transform *
                            max_libsharp_transforms * transform_block,
                    collocation_metadata->get_sharp_geom_info(), alm_info,
                    num_transforms % max_libsharp_transforms, SHARP_DP, nullptr,
                    nullptr);
    }
  }
}
}  // namespace detail

template <ComplexRepresentation Representation, int Spin>
SpinWeighted<ComplexModalVector, Spin> swsh_transform(
    const size_t l_max, const size_t number_of_radial_points,
    const SpinWeighted<ComplexDataVector, Spin>& collocation) noexcept {
  SpinWeighted<ComplexModalVector, Spin> result_vector{};
  swsh_transform<Representation, Spin>(l_max, number_of_radial_points,
                                       make_not_null(&result_vector),
                                       collocation);
  return result_vector;
}

template <ComplexRepresentation Representation, int Spin>
SpinWeighted<ComplexDataVector, Spin> inverse_swsh_transform(
    const size_t l_max, const size_t number_of_radial_points,
    const SpinWeighted<ComplexModalVector, Spin>&
        libsharp_coefficients) noexcept {
  SpinWeighted<ComplexDataVector, Spin> result_vector{};
  inverse_swsh_transform<Representation, Spin>(l_max, number_of_radial_points,
                                               make_not_null(&result_vector),
                                               libsharp_coefficients);
  return result_vector;
}

#define GET_REPRESENTATION(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_SPIN(data) BOOST_PP_TUPLE_ELEM(1, data)

#define SWSH_TRANSFORM_INSTANTIATION(r, data)                       \
  template SpinWeighted<ComplexModalVector, GET_SPIN(data)>         \
  swsh_transform<GET_REPRESENTATION(data), GET_SPIN(data)>(         \
      const size_t l_max, const size_t number_of_radial_points,     \
      const SpinWeighted<ComplexDataVector, GET_SPIN(data)>&        \
          collocation) noexcept;                                    \
  template SpinWeighted<ComplexDataVector, GET_SPIN(data)>          \
  inverse_swsh_transform<GET_REPRESENTATION(data), GET_SPIN(data)>( \
      const size_t l_max, const size_t number_of_radial_points,     \
      const SpinWeighted<ComplexModalVector, GET_SPIN(data)>&       \
          coefficients) noexcept;

#define SWSH_TRANSFORM_UTILITIES_INSTANTIATION(r, data)                   \
  template void append_libsharp_collocation_pointers(                     \
      const gsl::not_null<std::vector<double*>*> collocation_data,        \
      const gsl::not_null<                                                \
          std::vector<ComplexDataView<GET_REPRESENTATION(data)>>*>        \
          collocation_views,                                              \
      const gsl::not_null<ComplexDataVector*> vector, const size_t l_max, \
      const bool positive_spin) noexcept;                                 \
  template void execute_libsharp_transform_set(                           \
      const sharp_jobtype& jobtype, const int spin,                       \
      const gsl::not_null<std::vector<std::complex<double>*>*>            \
          coefficient_data,                                               \
      const gsl::not_null<std::vector<double*>*> collocation_data,        \
      const gsl::not_null<                                                \
          const CollocationMetadata<GET_REPRESENTATION(data)>*>           \
          collocation_metadata,                                           \
      const sharp_alm_info* alm_info, const size_t num_transforms) noexcept;

namespace detail {
GENERATE_INSTANTIATIONS(SWSH_TRANSFORM_UTILITIES_INSTANTIATION,
                        (ComplexRepresentation::Interleaved,
                         ComplexRepresentation::RealsThenImags))
}  // namespace detail

GENERATE_INSTANTIATIONS(SWSH_TRANSFORM_INSTANTIATION,
                        (ComplexRepresentation::Interleaved,
                         ComplexRepresentation::RealsThenImags),
                        (-2, -1, 0, 1, 2))

#undef GET_REPRESENTATION
#undef GET_SPIN
#undef SWSH_TRANSFORM_INSTANTIATION
#undef SWSH_TRANSFORM_UTILITIES_INSTANTIATION

}  // namespace Swsh
}  // namespace Spectral
