// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeArray.hpp"

// IWYU pragma: no_forward_declare SpinWeighted

namespace Spectral::Swsh {

template <int Spin>
void filter_swsh_volume_quantity(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> to_filter,
    const size_t l_max, const size_t limit_l, const double exponential_alpha,
    const size_t exponential_half_power,
    const gsl::not_null<ComplexDataVector*> buffer,
    const gsl::not_null<SpinWeighted<ComplexModalVector, Spin>*>
        transform_buffer) noexcept {
  const size_t number_of_radial_grid_points =
      to_filter->size() / number_of_swsh_collocation_points(l_max);

  if (LIKELY(exponential_alpha != 0.0)) {
    buffer->destructive_resize(to_filter->size());
    // Filter the radial direction using the provided exponential parameters.
    apply_matrices(buffer,
                   make_array(Matrix{}, Matrix{},
                              Spectral::filtering::exponential_filter(
                                  Mesh<1>{number_of_radial_grid_points,
                                          Spectral::Basis::Legendre,
                                          Spectral::Quadrature::GaussLobatto},
                                  exponential_alpha, exponential_half_power)),
                   to_filter->data(),
                   Index<3>{number_of_swsh_phi_collocation_points(l_max),
                            number_of_swsh_theta_collocation_points(l_max),
                            number_of_radial_grid_points});
    to_filter->data() = *buffer;
  }
  if (LIKELY(limit_l < l_max)) {
    transform_buffer->destructive_resize(
        number_of_radial_grid_points *
        size_of_libsharp_coefficient_vector(l_max));
    // Filter the angular direction using a transform and `limit_l`
    swsh_transform(l_max, number_of_radial_grid_points, transform_buffer,
                   *to_filter);
    const auto& coefficients_metadata = cached_coefficients_metadata(l_max);
    for (size_t i = 0; i < number_of_radial_grid_points; ++i) {
      for (const auto mode : coefficients_metadata) {
        if (mode.l > limit_l) {
          transform_buffer->data()[mode.transform_of_real_part_offset +
                                   i * coefficients_metadata.size()] = 0.0;
          transform_buffer->data()[mode.transform_of_imag_part_offset +
                                   i * coefficients_metadata.size()] = 0.0;
        }
      }
    }
  inverse_swsh_transform(l_max, number_of_radial_grid_points, to_filter,
                         *transform_buffer);
  }
}

template <int Spin>
void filter_swsh_volume_quantity(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> to_filter,
    const size_t l_max, const size_t limit_l, const double exponential_alpha,
    const size_t exponential_half_power) noexcept {
  const size_t number_of_radial_grid_points =
      to_filter->data().size() / number_of_swsh_collocation_points(l_max);
  auto buffer = ComplexDataVector{to_filter->data().size()};
  auto transform_buffer = SpinWeighted<ComplexModalVector, Spin>{
      number_of_radial_grid_points *
      size_of_libsharp_coefficient_vector(l_max)};
  filter_swsh_volume_quantity(to_filter, l_max, limit_l, exponential_alpha,
                              exponential_half_power, make_not_null(&buffer),
                              make_not_null(&transform_buffer));
}

template <int Spin>
void filter_swsh_boundary_quantity(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> to_filter,
    const size_t l_max, const size_t limit_l,
    const gsl::not_null<SpinWeighted<ComplexModalVector, Spin>*>
        transform_buffer) noexcept {
  if (LIKELY(limit_l < l_max)) {
    transform_buffer->destructive_resize(
        size_of_libsharp_coefficient_vector(l_max));
    swsh_transform(l_max, 1, transform_buffer, *to_filter);
    const auto& coefficients_metadata = cached_coefficients_metadata(l_max);
    for (const auto mode : coefficients_metadata) {
      if (mode.l > limit_l) {
        transform_buffer->data()[mode.transform_of_real_part_offset] = 0.0;
        transform_buffer->data()[mode.transform_of_imag_part_offset] = 0.0;
      }
    }
    inverse_swsh_transform(l_max, 1, to_filter, *transform_buffer);
  }
}

template <int Spin>
void filter_swsh_boundary_quantity(
    const gsl::not_null<SpinWeighted<ComplexDataVector, Spin>*> to_filter,
    const size_t l_max, const size_t limit_l) noexcept {
  const size_t number_of_radial_grid_points =
      to_filter->data().size() / number_of_swsh_collocation_points(l_max);
  auto transform_buffer = SpinWeighted<ComplexModalVector, Spin>{
      number_of_radial_grid_points *
      size_of_libsharp_coefficient_vector(l_max)};
  filter_swsh_boundary_quantity(to_filter, l_max, limit_l,
                                make_not_null(&transform_buffer));
}

#define GET_SPIN(data) BOOST_PP_TUPLE_ELEM(0, data)

#define SWSH_FILTER_INSTANTIATION(r, data)                                   \
  template void filter_swsh_volume_quantity(                                 \
      const gsl::not_null<SpinWeighted<ComplexDataVector, GET_SPIN(data)>*>  \
          to_filter,                                                         \
      const size_t l_max, const size_t limit_l,                              \
      const double exponential_alpha, const size_t exponential_half_power,   \
      const gsl::not_null<ComplexDataVector*> buffer,                        \
      const gsl::not_null<SpinWeighted<ComplexModalVector, GET_SPIN(data)>*> \
          transform_buffer) noexcept;                                        \
  template void filter_swsh_volume_quantity(                                 \
      const gsl::not_null<SpinWeighted<ComplexDataVector, GET_SPIN(data)>*>  \
          to_filter,                                                         \
      const size_t l_max, const size_t limit_l,                              \
      const double exponential_alpha,                                        \
      const size_t exponential_half_power) noexcept;                         \
  template void filter_swsh_boundary_quantity(                               \
      const gsl::not_null<SpinWeighted<ComplexDataVector, GET_SPIN(data)>*>  \
          to_filter,                                                         \
      const size_t l_max, const size_t limit_l,                              \
      const gsl::not_null<SpinWeighted<ComplexModalVector, GET_SPIN(data)>*> \
          transform_buffer) noexcept;                                        \
  template void filter_swsh_boundary_quantity(                               \
      const gsl::not_null<SpinWeighted<ComplexDataVector, GET_SPIN(data)>*>  \
          to_filter,                                                         \
      const size_t l_max, const size_t limit_l) noexcept;

GENERATE_INSTANTIATIONS(SWSH_FILTER_INSTANTIATION, (-2, -1, 0, 1, 2))

#undef GET_SPIN
#undef SWSH_FILTER_INSTANTIATION
}  // namespace Spectral::Swsh
