// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/iterator/zip_iterator.hpp>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {
namespace TestHelpers {
template <typename... Structure>
Tensor<ComplexModalVector, Structure...> tensor_to_goldberg_coefficients(
    const Tensor<DataVector, Structure...>& nodal_data,
    const size_t l_max) noexcept {
  Tensor<ComplexModalVector, Structure...> goldberg_modal_data{
      square(l_max + 1)};
  SpinWeighted<ComplexDataVector, 0> transform_buffer{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  for (size_t i = 0; i < nodal_data.size(); ++i) {
    transform_buffer.data() = std::complex<double>(1.0, 0.0) * nodal_data[i];
    goldberg_modal_data[i] =
        Spectral::Swsh::libsharp_to_goldberg_modes(
            Spectral::Swsh::swsh_transform(l_max, 1, transform_buffer), l_max)
            .data();
  }
  return goldberg_modal_data;
}

template <typename... Structure>
Tensor<ComplexModalVector, Structure...> tensor_to_libsharp_coefficients(
    const Tensor<DataVector, Structure...>& nodal_data,
    const size_t l_max) noexcept {
  Tensor<ComplexModalVector, Structure...> libsharp_modal_data{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
  SpinWeighted<ComplexDataVector, 0> transform_buffer{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  for (size_t i = 0; i < nodal_data.size(); ++i) {
    transform_buffer.data() = std::complex<double>(1.0, 0.0) * nodal_data[i];
    libsharp_modal_data[i] =
        Spectral::Swsh::swsh_transform(l_max, 1, transform_buffer).data();
  }
  return libsharp_modal_data;
}

template <typename AnalyticSolution>
void create_fake_time_varying_modal_data(
    const gsl::not_null<tnsr::ii<ComplexModalVector, 3>*>
        spatial_metric_coefficients,
    const gsl::not_null<tnsr::ii<ComplexModalVector, 3>*>
        dt_spatial_metric_coefficients,
    const gsl::not_null<tnsr::ii<ComplexModalVector, 3>*>
        dr_spatial_metric_coefficients,
    const gsl::not_null<tnsr::I<ComplexModalVector, 3>*> shift_coefficients,
    const gsl::not_null<tnsr::I<ComplexModalVector, 3>*> dt_shift_coefficients,
    const gsl::not_null<tnsr::I<ComplexModalVector, 3>*> dr_shift_coefficients,
    const gsl::not_null<Scalar<ComplexModalVector>*> lapse_coefficients,
    const gsl::not_null<Scalar<ComplexModalVector>*> dt_lapse_coefficients,
    const gsl::not_null<Scalar<ComplexModalVector>*> dr_lapse_coefficients,
    const AnalyticSolution& solution, const double extraction_radius,
    const double amplitude, const double frequency, const double time,
    const size_t l_max, const bool convert_to_goldberg = true,
    const bool apply_normalization_bug = false) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  // create the vector of collocation points that we want to interpolate to

  tnsr::I<DataVector, 3> collocation_points{number_of_angular_points};
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation) {
    get<0>(collocation_points)[collocation_point.offset] =
        extraction_radius * (1.0 + amplitude * sin(frequency * time)) *
        sin(collocation_point.theta) * cos(collocation_point.phi);
    get<1>(collocation_points)[collocation_point.offset] =
        extraction_radius * (1.0 + amplitude * sin(frequency * time)) *
        sin(collocation_point.theta) * sin(collocation_point.phi);
    get<2>(collocation_points)[collocation_point.offset] =
        extraction_radius * (1.0 + amplitude * sin(frequency * time)) *
        cos(collocation_point.theta);
  }

  const auto kerr_schild_variables = solution.variables(
      collocation_points, 0.0, gr::Solutions::KerrSchild::tags<DataVector>{});

  const Scalar<DataVector>& lapse =
      get<gr::Tags::Lapse<DataVector>>(kerr_schild_variables);
  const Scalar<DataVector>& dt_lapse =
      get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(kerr_schild_variables);
  const auto& d_lapse = get<gr::Solutions::KerrSchild::DerivLapse<DataVector>>(
      kerr_schild_variables);

  const auto& shift = get<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>(
      kerr_schild_variables);
  const auto& dt_shift =
      get<::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>>(
          kerr_schild_variables);
  const auto& d_shift = get<gr::Solutions::KerrSchild::DerivShift<DataVector>>(
      kerr_schild_variables);

  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>(
          kerr_schild_variables);
  const auto& dt_spatial_metric = get<
      ::Tags::dt<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>>(
      kerr_schild_variables);
  const auto& d_spatial_metric =
      get<gr::Solutions::KerrSchild::DerivSpatialMetric<DataVector>>(
          kerr_schild_variables);

  DataVector normalization_factor{number_of_angular_points, 1.0};
  if (apply_normalization_bug) {
    normalization_factor = 0.0;
    const auto inverse_spatial_metric =
        determinant_and_inverse(spatial_metric).second;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        normalization_factor +=
            inverse_spatial_metric.get(i, j) * collocation_points.get(i) *
            collocation_points.get(j) /
            square(extraction_radius *
                   (1.0 + amplitude * sin(frequency * time)));
      }
    }
    normalization_factor = sqrt(normalization_factor);
  }

  Scalar<DataVector> dr_lapse{number_of_angular_points};
  get(dr_lapse) = (get<0>(collocation_points) * get<0>(d_lapse) +
                   get<1>(collocation_points) * get<1>(d_lapse) +
                   get<2>(collocation_points) * get<2>(d_lapse)) /
                  (extraction_radius * normalization_factor);
  tnsr::I<DataVector, 3> dr_shift{number_of_angular_points};
  for (size_t i = 0; i < 3; ++i) {
    dr_shift.get(i) = (get<0>(collocation_points) * d_shift.get(0, i) +
                       get<1>(collocation_points) * d_shift.get(1, i) +
                       get<2>(collocation_points) * d_shift.get(2, i)) /
                      (extraction_radius * normalization_factor);
  }
  tnsr::ii<DataVector, 3> dr_spatial_metric{number_of_angular_points};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      dr_spatial_metric.get(i, j) =
          (get<0>(collocation_points) * d_spatial_metric.get(0, i, j) +
           get<1>(collocation_points) * d_spatial_metric.get(1, i, j) +
           get<2>(collocation_points) * d_spatial_metric.get(2, i, j)) /
          (extraction_radius * normalization_factor);
    }
  }

  if (convert_to_goldberg) {
    *lapse_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(lapse, l_max);
    *dt_lapse_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(dt_lapse, l_max);
    *dr_lapse_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(dr_lapse, l_max);

    *shift_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(shift, l_max);
    *dt_shift_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(dt_shift, l_max);
    *dr_shift_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(dr_shift, l_max);

    *spatial_metric_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(spatial_metric, l_max);
    *dt_spatial_metric_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(dt_spatial_metric, l_max);
    *dr_spatial_metric_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(dr_spatial_metric, l_max);
  } else {
    *lapse_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(lapse, l_max);
    *dt_lapse_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(dt_lapse, l_max);
    *dr_lapse_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(dr_lapse, l_max);

    *shift_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(shift, l_max);
    *dt_shift_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(dt_shift, l_max);
    *dr_shift_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(dr_shift, l_max);

    *spatial_metric_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(spatial_metric, l_max);
    *dt_spatial_metric_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(dt_spatial_metric, l_max);
    *dr_spatial_metric_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(dr_spatial_metric, l_max);
  }
}
}  // namespace TestHelpers
}  // namespace Cce
