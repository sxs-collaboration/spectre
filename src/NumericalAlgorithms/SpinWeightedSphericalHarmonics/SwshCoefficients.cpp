// Distributed under the MIT License.
// See LICENSE.txt for details

#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"

#include <array>
#include <cmath>
#include <ostream>
#include <sharp_cxx.h>
#include <utility>

#include "DataStructures/SpinWeighted.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTransform.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/StaticCache.hpp"

namespace Spectral::Swsh {

CoefficientsMetadata::CoefficientsMetadata(const size_t l_max) : l_max_(l_max) {
  sharp_alm_info* alm_to_initialize = nullptr;
  sharp_make_triangular_alm_info(l_max, l_max, 1, &alm_to_initialize);
  alm_info_.reset(alm_to_initialize);
}

const CoefficientsMetadata& cached_coefficients_metadata(const size_t l_max) {
  const static auto lazy_coefficients_cache =
      make_static_cache<CacheRange<0_st, detail::coefficients_maximum_l_max>>(
          [](const size_t generator_l_max) {
            return CoefficientsMetadata{generator_l_max};
          });
  return lazy_coefficients_cache(l_max);
}

template <int Spin>
std::complex<double> libsharp_mode_to_goldberg_plus_m(
    const LibsharpCoefficientInfo& coefficient_info,
    const SpinWeighted<ComplexModalVector, Spin>& libsharp_modes,
    const size_t radial_offset) {
  return sharp_swsh_sign(Spin, coefficient_info.m, true) *
             libsharp_modes
                 .data()[radial_offset * size_of_libsharp_coefficient_vector(
                                             coefficient_info.l_max) +
                         coefficient_info.transform_of_real_part_offset] +
         std::complex<double>(0.0, 1.0) *
             sharp_swsh_sign(Spin, coefficient_info.m, false) *
             libsharp_modes
                 .data()[radial_offset * size_of_libsharp_coefficient_vector(
                                             coefficient_info.l_max) +
                         coefficient_info.transform_of_imag_part_offset];
}

template <int Spin>
std::complex<double> libsharp_mode_to_goldberg_minus_m(
    const LibsharpCoefficientInfo& coefficient_info,
    const SpinWeighted<ComplexModalVector, Spin>& libsharp_modes,
    const size_t radial_offset) {
  return sharp_swsh_sign(Spin, -static_cast<int>(coefficient_info.m), true) *
             conj(libsharp_modes
                      .data()[radial_offset *
                                  size_of_libsharp_coefficient_vector(
                                      coefficient_info.l_max) +
                              coefficient_info.transform_of_real_part_offset]) +
         std::complex<double>(0.0, 1.0) *
             sharp_swsh_sign(Spin, -static_cast<int>((coefficient_info.m)),
                             false) *
             conj(libsharp_modes
                      .data()[radial_offset *
                                  size_of_libsharp_coefficient_vector(
                                      coefficient_info.l_max) +
                              coefficient_info.transform_of_imag_part_offset]);
}

template <int Spin>
std::complex<double> libsharp_mode_to_goldberg(
    const size_t l, const int m, const size_t l_max,
    const SpinWeighted<ComplexModalVector, Spin>& libsharp_modes,
    const size_t radial_offset) {
  const CoefficientsMetadata::CoefficientsIndexIterator coefficients_iterator{
      l_max, l, static_cast<size_t>(abs(m))};
  if (m >= 0) {
    return libsharp_mode_to_goldberg_plus_m(*coefficients_iterator,
                                            libsharp_modes, radial_offset);
  } else {
    return libsharp_mode_to_goldberg_minus_m(*coefficients_iterator,
                                             libsharp_modes, radial_offset);
  }
}

template <int Spin>
void goldberg_modes_to_libsharp_modes_single_pair(
    const LibsharpCoefficientInfo& coefficient_info,
    const gsl::not_null<SpinWeighted<ComplexModalVector, Spin>*> libsharp_modes,
    const size_t radial_offset,
    const std::complex<double> goldberg_plus_m_mode_value,
    std::complex<double> goldberg_minus_m_mode_value) {
  const auto sign_determinant = static_cast<double>(
      sharp_swsh_sign(Spin, coefficient_info.m, true) *
          sharp_swsh_sign(Spin, -static_cast<int>(coefficient_info.m), false) +
      sharp_swsh_sign(Spin, coefficient_info.m, false) *
          sharp_swsh_sign(Spin, -static_cast<int>(coefficient_info.m), true));
  const auto i = std::complex<double>(0.0, 1.0);
  if (coefficient_info.m == 0) {
    goldberg_minus_m_mode_value = goldberg_plus_m_mode_value;
  }
  libsharp_modes->data()[radial_offset * size_of_libsharp_coefficient_vector(
                                             coefficient_info.l_max) +
                         coefficient_info.transform_of_real_part_offset] =
      (sharp_swsh_sign(Spin, -static_cast<int>(coefficient_info.m), false) *
           goldberg_plus_m_mode_value +
       sharp_swsh_sign(Spin, coefficient_info.m, false) *
           conj(goldberg_minus_m_mode_value)) /
      sign_determinant;

  libsharp_modes->data()[radial_offset * size_of_libsharp_coefficient_vector(
                                             coefficient_info.l_max) +
                         coefficient_info.transform_of_imag_part_offset] =
      (-i * sharp_swsh_sign(Spin, -static_cast<int>(coefficient_info.m), true) *
           goldberg_plus_m_mode_value +
       i * sharp_swsh_sign(Spin, coefficient_info.m, true) *
           conj(goldberg_minus_m_mode_value)) /
      sign_determinant;
}

template <int Spin>
void goldberg_modes_to_libsharp_modes_single_pair(
    const size_t l, const int m, const size_t l_max,
    const gsl::not_null<SpinWeighted<ComplexModalVector, Spin>*> libsharp_modes,
    const size_t radial_offset,
    const std::complex<double> goldberg_plus_m_mode_value,
    const std::complex<double> goldberg_minus_m_mode_value) {
  CoefficientsMetadata::CoefficientsIndexIterator coefficients_iterator{
      l_max, l, static_cast<size_t>(abs(m))};
  goldberg_modes_to_libsharp_modes_single_pair(
      *coefficients_iterator, libsharp_modes, radial_offset,
      goldberg_plus_m_mode_value, goldberg_minus_m_mode_value);
}

template <int Spin>
void libsharp_to_goldberg_modes(
    const gsl::not_null<SpinWeighted<ComplexModalVector, Spin>*> goldberg_modes,
    const SpinWeighted<ComplexModalVector, Spin>& libsharp_modes,
    const size_t l_max) {
  const size_t number_of_radial_grid_points =
      libsharp_modes.data().size() / size_of_libsharp_coefficient_vector(l_max);

  goldberg_modes->destructive_resize(square(1 + l_max) *
                                     number_of_radial_grid_points);

  const auto& coefficients_metadata = cached_coefficients_metadata(l_max);
  for (size_t i = 0; i < number_of_radial_grid_points; ++i) {
    for (const auto coefficient_info : coefficients_metadata) {
      goldberg_modes->data()[goldberg_mode_index(
          coefficient_info.l_max, coefficient_info.l,
          static_cast<int>(coefficient_info.m), i)] =
          libsharp_mode_to_goldberg_plus_m(coefficient_info, libsharp_modes, i);
      goldberg_modes->data()[goldberg_mode_index(
          coefficient_info.l_max, coefficient_info.l,
          -static_cast<int>(coefficient_info.m), i)] =
          libsharp_mode_to_goldberg_minus_m(coefficient_info, libsharp_modes,
                                            i);
    }
  }
}

template <int Spin>
SpinWeighted<ComplexModalVector, Spin> libsharp_to_goldberg_modes(
    const SpinWeighted<ComplexModalVector, Spin>& libsharp_modes,
    const size_t l_max) {
  const size_t number_of_radial_grid_points =
      libsharp_modes.data().size() / size_of_libsharp_coefficient_vector(l_max);
  SpinWeighted<ComplexModalVector, Spin> result{square(1 + l_max) *
                                                number_of_radial_grid_points};
  libsharp_to_goldberg_modes(make_not_null(&result), libsharp_modes, l_max);
  return result;
}

template <int Spin>
void goldberg_to_libsharp_modes(
    const gsl::not_null<SpinWeighted<ComplexModalVector, Spin>*> libsharp_modes,
    const SpinWeighted<ComplexModalVector, Spin>& goldberg_modes,
    const size_t l_max) {
  const size_t number_of_radial_grid_points =
      goldberg_modes.data().size() / square(l_max + 1);
  for (size_t i = 0; i < number_of_radial_grid_points; ++i) {
    for (const auto mode : cached_coefficients_metadata(l_max)) {
      goldberg_modes_to_libsharp_modes_single_pair(
          mode, libsharp_modes, i,
          goldberg_modes.data()[goldberg_mode_index(
              l_max, mode.l, static_cast<int>(mode.m), i)],
          goldberg_modes.data()[goldberg_mode_index(
              l_max, mode.l, -static_cast<int>(mode.m), i)]);
    }
  }
}

template <int Spin>
SpinWeighted<ComplexModalVector, Spin> goldberg_to_libsharp_modes(
    const SpinWeighted<ComplexModalVector, Spin>& goldberg_modes,
    const size_t l_max) {
  const size_t number_of_radial_grid_points =
      goldberg_modes.data().size() / square(l_max + 1);
  SpinWeighted<ComplexModalVector, Spin> result{
      size_of_libsharp_coefficient_vector(l_max) *
      number_of_radial_grid_points};
  goldberg_to_libsharp_modes(make_not_null(&result), goldberg_modes, l_max);
  return result;
}

template <int Spin>
void goldberg_to_nodal(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> nodal_values,
    const SpinWeighted<ComplexModalVector, Spin>& goldberg_modes,
    const size_t l_max) {
  const size_t l_max_plus_one_squared = square(l_max + 1);
  const size_t number_of_radial_grid_points =
      goldberg_modes.size() / l_max_plus_one_squared;
  const size_t number_of_libsharp_modes =
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max);
  SpinWeighted<ComplexModalVector, Spin> modal_libsharp{
      number_of_radial_grid_points * number_of_libsharp_modes};
  // Here we convert the goldberg modal values libsharp modal values, this is
  // done for each radial collocation points using non-owning angular views
  for (size_t i = 0; i < number_of_radial_grid_points; i++) {
    SpinWeighted<ComplexModalVector, Spin> angular_view_goldberg;
    auto& complex_data_vector_view_goldberg = angular_view_goldberg.data();
    complex_data_vector_view_goldberg.set_data_ref(
        const_cast<std::complex<double>*>(
            goldberg_modes.data().data()) +  // NOLINT
            (l_max_plus_one_squared * i),
        l_max_plus_one_squared);
    SpinWeighted<ComplexModalVector, Spin> angular_view_libsharp;
    auto& complex_data_vector_view_libsharp = angular_view_libsharp.data();
    complex_data_vector_view_libsharp.set_data_ref(
        modal_libsharp.data().data() + (number_of_libsharp_modes * i),
        number_of_libsharp_modes);
    angular_view_libsharp = Spectral::Swsh::goldberg_to_libsharp_modes(
        angular_view_goldberg, l_max);
  }
  // The libsharp modal values are converted into grid point nodal values
  Spectral::Swsh::inverse_swsh_transform(l_max, number_of_radial_grid_points,
                                         nodal_values, modal_libsharp);
}

template <int Spin>
SpinWeighted<ComplexDataVector, Spin> goldberg_to_nodal(
    const SpinWeighted<ComplexModalVector, Spin>& goldberg_modes,
    const size_t l_max) {
  SpinWeighted<ComplexDataVector, Spin> result{};
  goldberg_to_nodal(make_not_null(&result), goldberg_modes, l_max);
  return result;
}

template <int Spin>
void nodal_to_goldberg(
    const gsl::not_null<SpinWeighted<ComplexModalVector, Spin>*> goldberg_modes,
    const SpinWeighted<ComplexDataVector, Spin>& nodal_values,
    const size_t l_max) {
  // The nodal values are converted into
  // libsharp compatible modes
  const size_t number_of_swsh_grid_points = (l_max + 1) * (2 * l_max + 1);
  const size_t number_of_radial_grid_points =
      nodal_values.size() / number_of_swsh_grid_points;
  auto modal_libsharp = Spectral::Swsh::swsh_transform(
      l_max, number_of_radial_grid_points, nodal_values);
  // Here we convert the libsharp modes to goldberg modes, this is
  // done for each radial collocation points using non-owning angular views
  const size_t l_max_plus_one_squared = square(l_max + 1);
  const size_t number_of_libsharp_modes =
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max);
  for (size_t i = 0; i < number_of_radial_grid_points; i++) {
    SpinWeighted<ComplexModalVector, Spin> angular_view_libsharp;
    auto& complex_data_vector_view_libsharp = angular_view_libsharp.data();
    complex_data_vector_view_libsharp.set_data_ref(
        modal_libsharp.data().data() + (number_of_libsharp_modes * i),
        number_of_libsharp_modes);
    SpinWeighted<ComplexModalVector, Spin> angular_view_goldberg;
    auto& complex_data_vector_view_goldberg = angular_view_goldberg.data();
    complex_data_vector_view_goldberg.set_data_ref(
        const_cast<std::complex<double>*>(
            (*goldberg_modes).data().data()) +  // NOLINT
            (l_max_plus_one_squared * i),
        l_max_plus_one_squared);
    angular_view_goldberg = Spectral::Swsh::libsharp_to_goldberg_modes(
        angular_view_libsharp, l_max);
  }
}

template <int Spin>
SpinWeighted<ComplexModalVector, Spin> nodal_to_goldberg(
    const SpinWeighted<ComplexDataVector, Spin>& nodal_values,
    const size_t l_max) {
  const size_t number_of_swsh_grid_points = (l_max + 1) * (2 * l_max + 1);
  const size_t number_of_radial_grid_points =
      nodal_values.size() / number_of_swsh_grid_points;
  const size_t l_max_plus_one_squared = square(l_max + 1);
  SpinWeighted<ComplexModalVector, Spin> result{l_max_plus_one_squared *
                                                number_of_radial_grid_points};
  nodal_to_goldberg(make_not_null(&result), nodal_values, l_max);
  return result;
}

#define GET_SPIN(data) BOOST_PP_TUPLE_ELEM(0, data)

#define LIBSHARP_TO_GOLDBERG_INSTANTIATION(r, data)                            \
  template std::complex<double> libsharp_mode_to_goldberg_plus_m(              \
      const LibsharpCoefficientInfo& coefficient_info,                         \
      const SpinWeighted<ComplexModalVector, GET_SPIN(data)>& libsharp_modes,  \
      const size_t radial_offset);                                             \
  template std::complex<double> libsharp_mode_to_goldberg_minus_m(             \
      const LibsharpCoefficientInfo& coefficient_info,                         \
      const SpinWeighted<ComplexModalVector, GET_SPIN(data)>& libsharp_modes,  \
      const size_t radial_offset);                                             \
  template std::complex<double> libsharp_mode_to_goldberg(                     \
      const size_t l, const int m, const size_t l_max,                         \
      const SpinWeighted<ComplexModalVector, GET_SPIN(data)>& libsharp_modes,  \
      const size_t radial_offset);                                             \
  template void goldberg_modes_to_libsharp_modes_single_pair(                  \
      const LibsharpCoefficientInfo& coefficient_info,                         \
      const gsl::not_null<SpinWeighted<ComplexModalVector, GET_SPIN(data)>*>   \
          libsharp_modes,                                                      \
      const size_t radial_offset,                                              \
      const std::complex<double> goldberg_plus_m_mode_value,                   \
      const std::complex<double> goldberg_minus_m_mode_value);                 \
  template void goldberg_modes_to_libsharp_modes_single_pair(                  \
      const size_t l, const int m, const size_t l_max,                         \
      const gsl::not_null<SpinWeighted<ComplexModalVector, GET_SPIN(data)>*>   \
          libsharp_modes,                                                      \
      const size_t radial_offset,                                              \
      const std::complex<double> goldberg_plus_m_mode_value,                   \
      const std::complex<double> goldberg_minus_m_mode_value);                 \
  template void libsharp_to_goldberg_modes(                                    \
      const gsl::not_null<SpinWeighted<ComplexModalVector, GET_SPIN(data)>*>   \
          goldberg_modes,                                                      \
      const SpinWeighted<ComplexModalVector, GET_SPIN(data)>& libsharp_modes,  \
      const size_t l_max);                                                     \
  template SpinWeighted<ComplexModalVector, GET_SPIN(data)>                    \
  libsharp_to_goldberg_modes(                                                  \
      const SpinWeighted<ComplexModalVector, GET_SPIN(data)>& libsharp_modes,  \
      const size_t l_max);                                                     \
  template SpinWeighted<ComplexModalVector, GET_SPIN(data)>                    \
  goldberg_to_libsharp_modes(                                                  \
      const SpinWeighted<ComplexModalVector, GET_SPIN(data)>& goldberg_modes,  \
      const size_t l_max);                                                     \
  template void goldberg_to_libsharp_modes(                                    \
      const gsl::not_null<SpinWeighted<ComplexModalVector, GET_SPIN(data)>*>   \
          libsharp_modes,                                                      \
      const SpinWeighted<ComplexModalVector, GET_SPIN(data)>& goldberg_modes,  \
      const size_t l_max);                                                     \
  template SpinWeighted<ComplexDataVector, GET_SPIN(data)> goldberg_to_nodal(  \
      const SpinWeighted<ComplexModalVector, GET_SPIN(data)>& goldberg_modes,  \
      const size_t l_max);                                                     \
  template void goldberg_to_nodal(                                             \
      const gsl::not_null<SpinWeighted<ComplexDataVector, GET_SPIN(data)>*>    \
          nodal_values,                                                        \
      const SpinWeighted<ComplexModalVector, GET_SPIN(data)>& goldberg_modes,  \
      const size_t l_max);                                                     \
  template SpinWeighted<ComplexModalVector, GET_SPIN(data)> nodal_to_goldberg( \
      const SpinWeighted<ComplexDataVector, GET_SPIN(data)>& nodal_values,     \
      const size_t l_max);                                                     \
  template void nodal_to_goldberg(                                             \
      const gsl::not_null<SpinWeighted<ComplexModalVector, GET_SPIN(data)>*>   \
          goldberg_modes,                                                      \
      const SpinWeighted<ComplexDataVector, GET_SPIN(data)>& nodal_values,     \
      const size_t l_max);

GENERATE_INSTANTIATIONS(LIBSHARP_TO_GOLDBERG_INSTANTIATION, (-2, -1, 0, 1, 2))

#undef LIBSHARP_TO_GOLDBERG_INSTANTIATION
#undef GET_SPIN

}  // namespace Spectral::Swsh
