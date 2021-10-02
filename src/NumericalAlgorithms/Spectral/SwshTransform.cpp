// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"

#include <cmath>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/SpinWeighted.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"

// IWYU pragma: no_forward_declare SpinWeighted

namespace Spectral::Swsh {

namespace detail {
template <ComplexRepresentation Representation>
void append_libsharp_collocation_pointers(
    const gsl::not_null<std::vector<double*>*> collocation_data,
    const gsl::not_null<std::vector<ComplexDataView<Representation>>*>
        collocation_views,
    const gsl::not_null<ComplexDataVector*> vector, const size_t l_max,
    const bool positive_spin) {
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
    const gsl::not_null<ComplexModalVector*> vector, const size_t l_max) {
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
    const sharp_alm_info* alm_info, const size_t num_transforms) {
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
    const SpinWeighted<ComplexDataVector, Spin>& collocation) {
  SpinWeighted<ComplexModalVector, Spin> result_vector{};
  swsh_transform<Representation, Spin>(l_max, number_of_radial_points,
                                       make_not_null(&result_vector),
                                       collocation);
  return result_vector;
}

template <ComplexRepresentation Representation, int Spin>
SpinWeighted<ComplexDataVector, Spin> inverse_swsh_transform(
    const size_t l_max, const size_t number_of_radial_points,
    const SpinWeighted<ComplexModalVector, Spin>& libsharp_coefficients) {
  SpinWeighted<ComplexDataVector, Spin> result_vector{};
  inverse_swsh_transform<Representation, Spin>(l_max, number_of_radial_points,
                                               make_not_null(&result_vector),
                                               libsharp_coefficients);
  return result_vector;
}

template <int Spin>
void interpolate_to_collocation(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> target,
    const SpinWeighted<ComplexDataVector, Spin>& source,
    const size_t target_l_max, const size_t source_l_max,
    const size_t number_of_radial_points) {
  const auto source_modes =
      swsh_transform(source_l_max, number_of_radial_points, source);
  SpinWeighted<ComplexModalVector, Spin> target_modes{
      size_of_libsharp_coefficient_vector(target_l_max) *
      number_of_radial_points};

  const auto& target_coefficient_metadata =
      cached_coefficients_metadata(target_l_max);
  for (size_t i = 0; i < number_of_radial_points; ++i) {
    auto source_coefficient_metadata_iterator =
        cached_coefficients_metadata(source_l_max).begin();
    for (const auto coefficient : target_coefficient_metadata) {
      // first, we advance `source_coefficient_metadata_iterator` to the same
      // mode that is represented by `coefficient`, or to the next mode
      // following `coefficient` that is present in the source resolution, or to
      // the iterator end if neither exists. Note: this assumes the libsharp
      // coefficient ordering for optimizations
      while (source_coefficient_metadata_iterator !=
                 cached_coefficients_metadata(source_l_max).end() and
             ((*source_coefficient_metadata_iterator).m < coefficient.m or
              ((*source_coefficient_metadata_iterator).l < coefficient.l and
               (*source_coefficient_metadata_iterator).m == coefficient.m))) {
        ++source_coefficient_metadata_iterator;
      }
      // assign the current coefficient if present in both the source and the
      // target
      if (source_coefficient_metadata_iterator !=
              cached_coefficients_metadata(source_l_max).end() and
          coefficient.l == (*source_coefficient_metadata_iterator).l and
          coefficient.m == (*source_coefficient_metadata_iterator).m) {
        target_modes
            .data()[i * size_of_libsharp_coefficient_vector(target_l_max) +
                    coefficient.transform_of_real_part_offset] =
            source_modes
                .data()[i * size_of_libsharp_coefficient_vector(source_l_max) +
                        (*source_coefficient_metadata_iterator)
                            .transform_of_real_part_offset];
        target_modes
            .data()[i * size_of_libsharp_coefficient_vector(target_l_max) +
                    coefficient.transform_of_imag_part_offset] =
            source_modes
                .data()[i * size_of_libsharp_coefficient_vector(source_l_max) +
                        (*source_coefficient_metadata_iterator)
                            .transform_of_imag_part_offset];
      } else {
        // assign 0.0 if the coefficient is present in the target and not in the
        // source representation
        target_modes
            .data()[i * size_of_libsharp_coefficient_vector(target_l_max) +
                    coefficient.transform_of_real_part_offset] =
            std::complex<double>(0.0, 0.0);
        target_modes
            .data()[i * size_of_libsharp_coefficient_vector(target_l_max) +
                    coefficient.transform_of_imag_part_offset] =
            std::complex<double>(0.0, 0.0);
      }
    }
  }
  inverse_swsh_transform(target_l_max, number_of_radial_points, target,
                         target_modes);
}

#define GET_REPRESENTATION(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_SPIN(data) BOOST_PP_TUPLE_ELEM(1, data)

#define SWSH_TRANSFORM_INSTANTIATION(r, data)                              \
  template SpinWeighted<ComplexModalVector, GET_SPIN(data)>                \
  swsh_transform<GET_REPRESENTATION(data), GET_SPIN(data)>(                \
      const size_t l_max, const size_t number_of_radial_points,            \
      const SpinWeighted<ComplexDataVector, GET_SPIN(data)>& collocation); \
  template SpinWeighted<ComplexDataVector, GET_SPIN(data)>                 \
  inverse_swsh_transform<GET_REPRESENTATION(data), GET_SPIN(data)>(        \
      const size_t l_max, const size_t number_of_radial_points,            \
      const SpinWeighted<ComplexModalVector, GET_SPIN(data)>& coefficients);

#define SWSH_TRANSFORM_UTILITIES_INSTANTIATION(r, data)                   \
  template void append_libsharp_collocation_pointers(                     \
      const gsl::not_null<std::vector<double*>*> collocation_data,        \
      const gsl::not_null<                                                \
          std::vector<ComplexDataView<GET_REPRESENTATION(data)>>*>        \
          collocation_views,                                              \
      const gsl::not_null<ComplexDataVector*> vector, const size_t l_max, \
      const bool positive_spin);                                          \
  template void execute_libsharp_transform_set(                           \
      const sharp_jobtype& jobtype, const int spin,                       \
      const gsl::not_null<std::vector<std::complex<double>*>*>            \
          coefficient_data,                                               \
      const gsl::not_null<std::vector<double*>*> collocation_data,        \
      const gsl::not_null<                                                \
          const CollocationMetadata<GET_REPRESENTATION(data)>*>           \
          collocation_metadata,                                           \
      const sharp_alm_info* alm_info, const size_t num_transforms);

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

#define GET_SPIN(data) BOOST_PP_TUPLE_ELEM(0, data)

#define SWSH_INTERPOLATION_INSTANTIATION(r, data)                           \
  template void interpolate_to_collocation<GET_SPIN(data)>(                 \
      const gsl::not_null<SpinWeighted<ComplexDataVector, GET_SPIN(data)>*> \
          target,                                                           \
      const SpinWeighted<ComplexDataVector, GET_SPIN(data)>& source,        \
      const size_t target_l_max, const size_t source_l_max,                 \
      const size_t number_of_radial_points);

GENERATE_INSTANTIATIONS(SWSH_INTERPOLATION_INSTANTIATION, (-2, -1, 0, 1, 2))

#undef GET_SPIN
#undef SWSH_INTERPOLATION_INSTANTIATION
#undef SWSH_TRANSFORM_INSTANTIATION
#undef SWSH_TRANSFORM_UTILITIES_INSTANTIATION

}  // namespace Spectral::Swsh
