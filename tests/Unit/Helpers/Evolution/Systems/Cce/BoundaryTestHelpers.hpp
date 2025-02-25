// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/iterator/zip_iterator.hpp>
#include <cstddef>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "Evolution/Systems/Cce/WorldtubeModeRecorder.hpp"
#include "Helpers/Evolution/Systems/Cce/WriteToWorldtubeH5.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::TestHelpers {
template <typename... Structure>
Tensor<ComplexModalVector, Structure...> tensor_to_goldberg_coefficients(
    const Tensor<DataVector, Structure...>& nodal_data, size_t l_max) {
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
    const size_t l_max)  // NOLINT(readability-avoid-const-params-in-decls)
{
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
void create_fake_time_varying_gh_nodal_data(
    const gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
    const gsl::not_null<tnsr::iaa<DataVector, 3>*> phi,
    const gsl::not_null<tnsr::aa<DataVector, 3>*> pi,
    const AnalyticSolution& solution, const double extraction_radius,
    const double amplitude, const double frequency, const double time,
    const size_t l_max) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  // create the vector of collocation points that we want to interpolate to

  tnsr::I<DataVector, 3> collocation_points{number_of_angular_points};
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto collocation_point : collocation) {
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

  const auto& shift =
      get<gr::Tags::Shift<DataVector, 3>>(kerr_schild_variables);
  const auto& dt_shift =
      get<::Tags::dt<gr::Tags::Shift<DataVector, 3>>>(kerr_schild_variables);
  const auto& d_shift = get<gr::Solutions::KerrSchild::DerivShift<DataVector>>(
      kerr_schild_variables);

  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(kerr_schild_variables);
  const auto& dt_spatial_metric =
      get<::Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>(
          kerr_schild_variables);
  const auto& d_spatial_metric =
      get<gr::Solutions::KerrSchild::DerivSpatialMetric<DataVector>>(
          kerr_schild_variables);

  gr::spacetime_metric(spacetime_metric, lapse, shift, spatial_metric);
  gh::phi(phi, lapse, d_lapse, shift, d_shift, spatial_metric,
          d_spatial_metric);
  gh::pi(pi, lapse, dt_lapse, shift, dt_shift, spatial_metric,
         dt_spatial_metric, *phi);
}

// Overload for all quantities, including Adm and Z4c quantities
template <typename AnalyticSolution, typename T = ComplexModalVector>
void create_fake_time_varying_data(
    const gsl::not_null<tnsr::ii<T, 3>*> spatial_metric_coefficients,
    const gsl::not_null<tnsr::ii<T, 3>*> dt_spatial_metric_coefficients,
    const gsl::not_null<tnsr::ii<T, 3>*> dr_spatial_metric_coefficients,
    const gsl::not_null<tnsr::I<T, 3>*> shift_coefficients,
    const gsl::not_null<tnsr::I<T, 3>*> dt_shift_coefficients,
    const gsl::not_null<tnsr::I<T, 3>*> dr_shift_coefficients,
    const gsl::not_null<Scalar<T>*> lapse_coefficients,
    const gsl::not_null<Scalar<T>*> dt_lapse_coefficients,
    const gsl::not_null<Scalar<T>*> dr_lapse_coefficients,
    const gsl::not_null<tnsr::ii<T, 3>*> extrinsic_curvature_coefficients,
    const gsl::not_null<tnsr::I<T, 3>*> auxiliary_shift_coefficients,
    const gsl::not_null<tnsr::I<T, 3>*> conformal_christoffel_coefficients,
    const gsl::not_null<tnsr::ijj<T, 3>*> deriv_spatial_metric_coefficients,
    const gsl::not_null<tnsr::iJ<T, 3>*> deriv_shift_coefficients,
    const gsl::not_null<tnsr::i<T, 3>*> deriv_lapse_coefficients,
    const AnalyticSolution& solution, const double extraction_radius,
    const double amplitude, const double frequency, const double time,
    const size_t l_max, const bool convert_to_goldberg = true,
    const bool apply_normalization_bug = false) {
  static_assert(std::is_same_v<T, ComplexModalVector> or
                std::is_same_v<T, DataVector>);
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  // create the vector of collocation points that we want to interpolate to

  tnsr::I<DataVector, 3> collocation_points{number_of_angular_points};
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto collocation_point : collocation) {
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

  const auto& shift =
      get<gr::Tags::Shift<DataVector, 3>>(kerr_schild_variables);
  const auto& dt_shift =
      get<::Tags::dt<gr::Tags::Shift<DataVector, 3>>>(kerr_schild_variables);
  const auto& d_shift = get<gr::Solutions::KerrSchild::DerivShift<DataVector>>(
      kerr_schild_variables);

  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(kerr_schild_variables);
  const auto& dt_spatial_metric =
      get<::Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>(
          kerr_schild_variables);
  const auto& d_spatial_metric =
      get<gr::Solutions::KerrSchild::DerivSpatialMetric<DataVector>>(
          kerr_schild_variables);

  const auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(kerr_schild_variables);
  const auto auxiliary_shift = [&shift]() {
    auto result = shift;
    get<0>(result) = 0.5;
    get<1>(result) = 0.5;
    get<2>(result) = 0.5;
    return result;
  }();
  // The value in the KerrSchild vars isn't conformal, but this is just a test
  // so we don't care
  const auto& conformal_christoffel =
      get<gr::Tags::TraceSpatialChristoffelSecondKind<DataVector, 3>>(
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

  if constexpr (std::is_same_v<T, ComplexModalVector>) {
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
          TestHelpers::tensor_to_goldberg_coefficients(dt_spatial_metric,
                                                       l_max);
      *dr_spatial_metric_coefficients =
          TestHelpers::tensor_to_goldberg_coefficients(dr_spatial_metric,
                                                       l_max);
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
          TestHelpers::tensor_to_libsharp_coefficients(dt_spatial_metric,
                                                       l_max);
      *dr_spatial_metric_coefficients =
          TestHelpers::tensor_to_libsharp_coefficients(dr_spatial_metric,
                                                       l_max);
    }
    (void)extrinsic_curvature_coefficients;
    (void)auxiliary_shift_coefficients;
    (void)conformal_christoffel_coefficients;
    (void)deriv_spatial_metric_coefficients;
    (void)deriv_shift_coefficients;
    (void)deriv_lapse_coefficients;
  } else {
    get(*lapse_coefficients) = get(lapse);
    get(*dt_lapse_coefficients) = get(dt_lapse);
    get(*dr_lapse_coefficients) = get(dr_lapse);

    for (size_t i = 0; i < 3; i++) {
      shift_coefficients->get(i) = shift.get(i);
      dt_shift_coefficients->get(i) = dt_shift.get(i);
      dr_shift_coefficients->get(i) = dr_shift.get(i);
      auxiliary_shift_coefficients->get(i) = auxiliary_shift.get(i);
      conformal_christoffel_coefficients->get(i) = conformal_christoffel.get(i);
      deriv_lapse_coefficients->get(i) = d_lapse.get(i);

      for (size_t j = 0; j < 3; j++) {
        deriv_shift_coefficients->get(i, j) = d_shift.get(i, j);
        if (j < i) {
          continue;
        }
        spatial_metric_coefficients->get(i, j) = spatial_metric.get(i, j);
        dt_spatial_metric_coefficients->get(i, j) = dt_spatial_metric.get(i, j);
        dr_spatial_metric_coefficients->get(i, j) = dr_spatial_metric.get(i, j);
        extrinsic_curvature_coefficients->get(i, j) =
            extrinsic_curvature.get(i, j);

        for (size_t k = 0; k < 3; k++) {
          deriv_spatial_metric_coefficients->get(k, i, j) =
              d_spatial_metric.get(k, i, j);
        }
      }
    }
  }
}

// Overload for only the strictly necessary quantities (time and radial derivs)
template <typename AnalyticSolution, typename T = ComplexModalVector>
void create_fake_time_varying_data(
    const gsl::not_null<tnsr::ii<T, 3>*> spatial_metric_coefficients,
    const gsl::not_null<tnsr::ii<T, 3>*> dt_spatial_metric_coefficients,
    const gsl::not_null<tnsr::ii<T, 3>*> dr_spatial_metric_coefficients,
    const gsl::not_null<tnsr::I<T, 3>*> shift_coefficients,
    const gsl::not_null<tnsr::I<T, 3>*> dt_shift_coefficients,
    const gsl::not_null<tnsr::I<T, 3>*> dr_shift_coefficients,
    const gsl::not_null<Scalar<T>*> lapse_coefficients,
    const gsl::not_null<Scalar<T>*> dt_lapse_coefficients,
    const gsl::not_null<Scalar<T>*> dr_lapse_coefficients,
    const AnalyticSolution& solution, const double extraction_radius,
    const double amplitude, const double frequency, const double time,
    const size_t l_max, const bool convert_to_goldberg = true,
    const bool apply_normalization_bug = false) {
  tnsr::ii<T, 3> extrinsic_curvature_coefficients{};
  tnsr::I<T, 3> auxiliary_shift_coefficients{};
  tnsr::I<T, 3> conformal_christoffel_coefficients{};
  tnsr::ijj<T, 3> deriv_spatial_metric_coefficients{};
  tnsr::iJ<T, 3> deriv_shift_coefficients{};
  tnsr::i<T, 3> deriv_lapse_coefficients{};
  create_fake_time_varying_data(
      spatial_metric_coefficients, dt_spatial_metric_coefficients,
      dr_spatial_metric_coefficients, shift_coefficients, dt_shift_coefficients,
      dr_shift_coefficients, lapse_coefficients, dt_lapse_coefficients,
      dr_lapse_coefficients, make_not_null(&extrinsic_curvature_coefficients),
      make_not_null(&auxiliary_shift_coefficients),
      make_not_null(&conformal_christoffel_coefficients),
      make_not_null(&deriv_spatial_metric_coefficients),
      make_not_null(&deriv_shift_coefficients),
      make_not_null(&deriv_lapse_coefficients), solution, extraction_radius,
      amplitude, frequency, time, l_max, convert_to_goldberg,
      apply_normalization_bug);
}

template <typename T = ComplexModalVector, bool WriteBondi = true,
          typename AnalyticSolution>
void write_test_file(const AnalyticSolution& solution,
                     const std::string& filename, const double target_time,
                     const double extraction_radius, const double frequency,
                     const double amplitude, const size_t l_max,
                     const bool descending_m = true,
                     const bool write_extra_adm_vars = false) {
  const bool is_modal = std::is_same_v<T, ComplexModalVector>;
  const size_t size =
      is_modal ? square(l_max + 1)
               : Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  (void)number_of_angular_points;
  tnsr::ii<T, 3> spatial_metric_coefficients{size};
  tnsr::ii<T, 3> dt_spatial_metric_coefficients{size};
  tnsr::ii<T, 3> dr_spatial_metric_coefficients{size};
  tnsr::I<T, 3> shift_coefficients{size};
  tnsr::I<T, 3> dt_shift_coefficients{size};
  tnsr::I<T, 3> dr_shift_coefficients{size};
  Scalar<T> lapse_coefficients{size};
  Scalar<T> dt_lapse_coefficients{size};
  Scalar<T> dr_lapse_coefficients{size};
  tnsr::ii<T, 3> extrinsic_curvature_coefficients{size};
  tnsr::I<T, 3> auxiliary_shift_coefficients{size};
  tnsr::I<T, 3> conformal_christoffel_coefficients{size};
  tnsr::ijj<T, 3> deriv_spatial_metric_coefficients{size};
  tnsr::iJ<T, 3> deriv_shift_coefficients{size};
  tnsr::i<T, 3> deriv_lapse_coefficients{size};
  Variables<Cce::Tags::characteristic_worldtube_boundary_tags<
      Cce::Tags::BoundaryValue>>
      boundary_data_variables{};
  if constexpr (WriteBondi) {
    boundary_data_variables.initialize(number_of_angular_points);
  }

  // write times to file for several steps before and after the target time
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  // scoped to close the file
  {
    using RecorderType =
        tmpl::conditional_t<WriteBondi and is_modal, Cce::WorldtubeModeRecorder,
                            Cce::TestHelpers::WorldtubeModeRecorder>;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    RecorderType recorder{l_max, filename};
    for (size_t t = 0; t < 30; ++t) {
      const double time = 0.1 * static_cast<double>(t) + target_time - 1.5;
      TestHelpers::create_fake_time_varying_data(
          make_not_null(&spatial_metric_coefficients),
          make_not_null(&dt_spatial_metric_coefficients),
          make_not_null(&dr_spatial_metric_coefficients),
          make_not_null(&shift_coefficients),
          make_not_null(&dt_shift_coefficients),
          make_not_null(&dr_shift_coefficients),
          make_not_null(&lapse_coefficients),
          make_not_null(&dt_lapse_coefficients),
          make_not_null(&dr_lapse_coefficients),
          make_not_null(&extrinsic_curvature_coefficients),
          make_not_null(&auxiliary_shift_coefficients),
          make_not_null(&conformal_christoffel_coefficients),
          make_not_null(&deriv_spatial_metric_coefficients),
          make_not_null(&deriv_shift_coefficients),
          make_not_null(&deriv_lapse_coefficients), solution, extraction_radius,
          amplitude, frequency, time, l_max, not WriteBondi);

      if constexpr (WriteBondi) {
        (void)descending_m;
        Cce::create_bondi_boundary_data(
            make_not_null(&boundary_data_variables),
            spatial_metric_coefficients, dt_spatial_metric_coefficients,
            dr_spatial_metric_coefficients, shift_coefficients,
            dt_shift_coefficients, dr_shift_coefficients, lapse_coefficients,
            dt_lapse_coefficients, dr_lapse_coefficients, extraction_radius,
            l_max);

        // loop over the tags that we want to dump.
        tmpl::for_each<Cce::Tags::worldtube_boundary_tags_for_writing<>>(
            [&](auto tag_v) {
              (void)tag_v;
              using tag = typename decltype(tag_v)::type;

              const ComplexDataVector& nodal_data =
                  get(get<tag>(boundary_data_variables)).data();

              if constexpr (is_modal) {
                recorder.template append_modal_data<tag::type::type::spin>(
                    Cce::dataset_label_for_tag<typename tag::tag>(), time,
                    nodal_data, l_max);
              } else {
                recorder.append_worldtube_mode_data(
                    Cce::dataset_label_for_tag<typename tag::tag>(), time,
                    nodal_data, l_max);
              }
            });
      } else {
        const auto deriv_str = [](const std::string& var, const size_t index) {
          return "/D"s + (index == 0 ? "x"s : (index == 1 ? "y"s : "z"s)) + var;
        };
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = 0; j < 3; ++j) {
            if (write_extra_adm_vars) {
              recorder.append_worldtube_mode_data(
                  detail::dataset_name_for_component(deriv_str("Shift", i), j),
                  time, deriv_shift_coefficients.get(i, j), descending_m);
            }

            if (j < i) {
              continue;
            }
            recorder.append_worldtube_mode_data(
                detail::dataset_name_for_component("/g", i, j), time,
                spatial_metric_coefficients.get(i, j), descending_m);
            recorder.append_worldtube_mode_data(
                detail::dataset_name_for_component("/Drg", i, j), time,
                dr_spatial_metric_coefficients.get(i, j), descending_m);
            recorder.append_worldtube_mode_data(
                detail::dataset_name_for_component("/Dtg", i, j), time,
                dt_spatial_metric_coefficients.get(i, j), descending_m);

            if (write_extra_adm_vars) {
              recorder.append_worldtube_mode_data(
                  detail::dataset_name_for_component("/K", i, j), time,
                  extrinsic_curvature_coefficients.get(i, j), descending_m);

              for (size_t k = 0; k < 3; k++) {
                recorder.append_worldtube_mode_data(
                    detail::dataset_name_for_component(deriv_str("g", k), i, j),
                    time, deriv_spatial_metric_coefficients.get(k, i, j));
              }
            }
          }
          recorder.append_worldtube_mode_data(
              detail::dataset_name_for_component("/Shift", i), time,
              shift_coefficients.get(i), descending_m);
          recorder.append_worldtube_mode_data(
              detail::dataset_name_for_component("/DrShift", i), time,
              dr_shift_coefficients.get(i), descending_m);
          recorder.append_worldtube_mode_data(
              detail::dataset_name_for_component("/DtShift", i), time,
              dt_shift_coefficients.get(i), descending_m);

          if (write_extra_adm_vars) {
            recorder.append_worldtube_mode_data(
                detail::dataset_name_for_component("/AuxiliaryShift", i), time,
                auxiliary_shift_coefficients.get(i), descending_m);
            recorder.append_worldtube_mode_data(
                detail::dataset_name_for_component("/ConformalChristoffel", i),
                time, conformal_christoffel_coefficients.get(i), descending_m);
            recorder.append_worldtube_mode_data(
                detail::dataset_name_for_component(deriv_str("Lapse", i)), time,
                deriv_lapse_coefficients.get(i), descending_m);
          }
        }
        recorder.append_worldtube_mode_data(
            detail::dataset_name_for_component("/Lapse"), time,
            get(lapse_coefficients), descending_m);
        recorder.append_worldtube_mode_data(
            detail::dataset_name_for_component("/DrLapse"), time,
            get(dr_lapse_coefficients), descending_m);
        recorder.append_worldtube_mode_data(
            detail::dataset_name_for_component("/DtLapse"), time,
            get(dt_lapse_coefficients), descending_m);
      }
    }
  }
}
}  // namespace Cce::TestHelpers
