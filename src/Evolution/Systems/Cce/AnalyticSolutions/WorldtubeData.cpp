// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::Solutions {

void WorldtubeData::variables_impl(
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_coordinates,
    const size_t output_l_max, double /*time*/,
    tmpl::type_<Tags::CauchyCartesianCoords> /*meta*/) const {
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(output_l_max);
  for (const auto collocation_point : collocation) {
    get<0>(*cartesian_coordinates)[collocation_point.offset] =
        extraction_radius_ * cos(collocation_point.phi) *
        sin(collocation_point.theta);
    get<1>(*cartesian_coordinates)[collocation_point.offset] =
        extraction_radius_ * sin(collocation_point.phi) *
        sin(collocation_point.theta);
    get<2>(*cartesian_coordinates)[collocation_point.offset] =
        extraction_radius_ * cos(collocation_point.theta);
  }
}

void WorldtubeData::variables_impl(
    const gsl::not_null<tnsr::i<DataVector, 3>*> dr_cartesian_coordinates,
    const size_t output_l_max, const double /*time*/,
    tmpl::type_<Tags::Dr<Tags::CauchyCartesianCoords>> /*meta*/) const {
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(output_l_max);
  for (const auto collocation_point : collocation) {
    get<0>(*dr_cartesian_coordinates)[collocation_point.offset] =
        cos(collocation_point.phi) * sin(collocation_point.theta);
    get<1>(*dr_cartesian_coordinates)[collocation_point.offset] =
        sin(collocation_point.phi) * sin(collocation_point.theta);
    get<2>(*dr_cartesian_coordinates)[collocation_point.offset] =
        cos(collocation_point.theta);
  }
}

void WorldtubeData::variables_impl(
    const gsl::not_null<tnsr::aa<DataVector, 3>*> pi, const size_t output_l_max,
    const double time,
    tmpl::type_<gh::Tags::Pi<DataVector, 3>> /*meta*/) const {
  const auto& d_spacetime_metric =
      cache_or_compute<gh::Tags::Phi<DataVector, 3>>(output_l_max, time);
  const auto& dt_spacetime_metric =
      cache_or_compute<::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>>(
          output_l_max, time);
  const auto& lapse =
      cache_or_compute<gr::Tags::Lapse<DataVector>>(output_l_max, time);
  const auto& shift =
      cache_or_compute<gr::Tags::Shift<DataVector, 3>>(output_l_max, time);
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = a; b < 4; ++b) {
      pi->get(a, b) = -dt_spacetime_metric.get(a, b) / get(lapse);
      for (size_t i = 0; i < 3; ++i) {
        pi->get(a, b) +=
            shift.get(i) * d_spacetime_metric.get(i, a, b) / get(lapse);
      }
    }
  }
}

void WorldtubeData::variables_impl(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> spatial_metric,
    const size_t output_l_max, const double time,
    tmpl::type_<gr::Tags::SpatialMetric<DataVector, 3>> /*meta*/) const {
  gr::spatial_metric(spatial_metric,
                     cache_or_compute<gr::Tags::SpacetimeMetric<DataVector, 3>>(
                         output_l_max, time));
}

void WorldtubeData::variables_impl(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> dt_spatial_metric,
    const size_t output_l_max, const double time,
    tmpl::type_<::Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>> /*meta*/)
    const {
  const auto& dt_spacetime_metric =
      cache_or_compute<::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>>(
          output_l_max, time);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      dt_spatial_metric->get(i, j) = dt_spacetime_metric.get(i + 1, j + 1);
    }
  }
}

void WorldtubeData::variables_impl(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> dr_spatial_metric,
    const size_t output_l_max, const double time,
    tmpl::type_<Tags::Dr<gr::Tags::SpatialMetric<DataVector, 3>>> /*meta*/)
    const {
  const auto& dr_cartesian_coordinates =
      cache_or_compute<Tags::Dr<Tags::CauchyCartesianCoords>>(output_l_max,
                                                              time);
  const auto& d_spacetime_metric =
      cache_or_compute<gh::Tags::Phi<DataVector, 3>>(output_l_max, time);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      dr_spatial_metric->get(i, j) = get<0>(dr_cartesian_coordinates) *
                                     d_spacetime_metric.get(0, i + 1, j + 1);
      for (size_t k = 1; k < 3; ++k) {
        dr_spatial_metric->get(i, j) += dr_cartesian_coordinates.get(k) *
                                        d_spacetime_metric.get(k, i + 1, j + 1);
      }
    }
  }
}

void WorldtubeData::variables_impl(
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift,
    const size_t output_l_max, const double time,
    tmpl::type_<gr::Tags::Shift<DataVector, 3>> /*meta*/) const {
  const auto& spacetime_metric =
      cache_or_compute<gr::Tags::SpacetimeMetric<DataVector, 3>>(output_l_max,
                                                                 time);
  const auto& spatial_metric =
      cache_or_compute<gr::Tags::SpatialMetric<DataVector, 3>>(output_l_max,
                                                               time);
  gr::shift(shift, spacetime_metric,
            determinant_and_inverse(spatial_metric).second);
}

void WorldtubeData::variables_impl(
    const gsl::not_null<tnsr::I<DataVector, 3>*> dt_shift,
    const size_t output_l_max, const double time,
    tmpl::type_<::Tags::dt<gr::Tags::Shift<DataVector, 3>>> /*meta*/) const {
  const auto& lapse =
      cache_or_compute<gr::Tags::Lapse<DataVector>>(output_l_max, time);
  const auto& shift =
      cache_or_compute<gr::Tags::Shift<DataVector, 3>>(output_l_max, time);
  gh::time_deriv_of_shift(
      dt_shift, lapse, shift,
      determinant_and_inverse(
          cache_or_compute<gr::Tags::SpatialMetric<DataVector, 3>>(output_l_max,
                                                                   time))
          .second,
      gr::spacetime_normal_vector(lapse, shift),
      cache_or_compute<gh::Tags::Phi<DataVector, 3>>(output_l_max, time),
      cache_or_compute<gh::Tags::Pi<DataVector, 3>>(output_l_max, time));
}

void WorldtubeData::variables_impl(
    const gsl::not_null<tnsr::I<DataVector, 3>*> dr_shift,
    const size_t output_l_max, const double time,
    tmpl::type_<Tags::Dr<gr::Tags::Shift<DataVector, 3>>> /*meta*/) const {
  const auto& lapse =
      cache_or_compute<gr::Tags::Lapse<DataVector>>(output_l_max, time);
  const auto& shift =
      cache_or_compute<gr::Tags::Shift<DataVector, 3>>(output_l_max, time);
  const auto d_shift = gh::spatial_deriv_of_shift(
      lapse,
      gr::inverse_spacetime_metric(
          lapse, shift,
          determinant_and_inverse(
              cache_or_compute<gr::Tags::SpatialMetric<DataVector, 3>>(
                  output_l_max, time))
              .second),
      gr::spacetime_normal_vector(lapse, shift),
      cache_or_compute<gh::Tags::Phi<DataVector, 3>>(output_l_max, time));
  const auto& dr_cartesian_coordinates =
      cache_or_compute<Tags::Dr<Tags::CauchyCartesianCoords>>(output_l_max,
                                                              time);

  for (size_t i = 0; i < 3; ++i) {
    dr_shift->get(i) = get<0>(dr_cartesian_coordinates) * d_shift.get(0, i);
    for (size_t j = 1; j < 3; ++j) {
      dr_shift->get(i) += dr_cartesian_coordinates.get(j) * d_shift.get(j, i);
    }
  }
}

void WorldtubeData::variables_impl(
    const gsl::not_null<Scalar<DataVector>*> lapse, const size_t output_l_max,
    const double time,
    tmpl::type_<gr::Tags::Lapse<DataVector>> /*meta*/) const {
  gr::lapse(
      lapse,
      cache_or_compute<gr::Tags::Shift<DataVector, 3>>(output_l_max, time),
      cache_or_compute<gr::Tags::SpacetimeMetric<DataVector, 3>>(output_l_max,
                                                                 time));
}

void WorldtubeData::variables_impl(
    const gsl::not_null<Scalar<DataVector>*> dt_lapse,
    const size_t output_l_max, const double time,
    tmpl::type_<::Tags::dt<gr::Tags::Lapse<DataVector>>> /*meta*/) const {
  const auto& lapse =
      cache_or_compute<gr::Tags::Lapse<DataVector>>(output_l_max, time);
  const auto& shift =
      cache_or_compute<gr::Tags::Shift<DataVector, 3>>(output_l_max, time);
  gh::time_deriv_of_lapse(
      dt_lapse, lapse, shift, gr::spacetime_normal_vector(lapse, shift),
      cache_or_compute<gh::Tags::Phi<DataVector, 3>>(output_l_max, time),
      cache_or_compute<gh::Tags::Pi<DataVector, 3>>(output_l_max, time));
}

void WorldtubeData::variables_impl(
    const gsl::not_null<Scalar<DataVector>*> dr_lapse,
    const size_t output_l_max, const double time,
    tmpl::type_<Tags::Dr<gr::Tags::Lapse<DataVector>>> /*meta*/) const {
  const auto& lapse =
      cache_or_compute<gr::Tags::Lapse<DataVector>>(output_l_max, time);
  const auto& shift =
      cache_or_compute<gr::Tags::Shift<DataVector, 3>>(output_l_max, time);
  const auto d_lapse = gh::spatial_deriv_of_lapse(
      lapse, gr::spacetime_normal_vector(lapse, shift),
      cache_or_compute<gh::Tags::Phi<DataVector, 3>>(output_l_max, time));
  const auto& dr_cartesian_coordinates =
      cache_or_compute<Tags::Dr<Tags::CauchyCartesianCoords>>(output_l_max,
                                                              time);

  get(*dr_lapse) = get<0>(dr_cartesian_coordinates) * get<0>(d_lapse) +
                   get<1>(dr_cartesian_coordinates) * get<1>(d_lapse) +
                   get<2>(dr_cartesian_coordinates) * get<2>(d_lapse);
}

void WorldtubeData::pup(PUP::er& p) {
  p | extraction_radius_;
  if (p.isUnpacking()) {
    intermediate_cache_ = IntermediateCacheTuple{};
  }
}

void KleinGordonWorldtubeData::variables_impl(
    const gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_coordinates,
    const size_t output_l_max, double /*time*/,
    tmpl::type_<Tags::CauchyCartesianCoords> /*meta*/) const {
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(output_l_max);
  for (const auto collocation_point : collocation) {
    get<0>(*cartesian_coordinates)[collocation_point.offset] =
        extraction_radius_ * cos(collocation_point.phi) *
        sin(collocation_point.theta);
    get<1>(*cartesian_coordinates)[collocation_point.offset] =
        extraction_radius_ * sin(collocation_point.phi) *
        sin(collocation_point.theta);
    get<2>(*cartesian_coordinates)[collocation_point.offset] =
        extraction_radius_ * cos(collocation_point.theta);
  }
}

void KleinGordonWorldtubeData::pup(PUP::er& p) {
  p | extraction_radius_;
  if (p.isUnpacking()) {
    intermediate_cache_ = IntermediateCacheTuple{};
  }
}

}  // namespace Cce::Solutions
