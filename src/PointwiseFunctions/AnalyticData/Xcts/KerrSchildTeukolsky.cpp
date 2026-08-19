// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/KerrSchildTeukolsky.hpp"

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.tpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts::AnalyticData::detail {

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial> /*meta*/)
    const {
  *conformal_metric =
      get<gr::Tags::SpatialMetric<DataType, Dim>>(kerr_schild_vars.get());
  const auto& teukolsky_spatial_metric =
      get<gr::Tags::SpatialMetric<DataType, Dim, Frame::Inertial>>(
          teukolsky_vars.get());
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      conformal_metric->get(i, j) += teukolsky_spatial_metric.get(i, j);
    }
  }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif  // defined(__GNUC__) && !defined(__clang__)

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_conformal_metric,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  if constexpr (std::is_same_v<DataType, DataVector>) {
    if (not(this->mesh.has_value() and this->inv_jacobian.has_value())) {
      ERROR("Need a mesh and a Jacobian for numeric differentiation.");
    }
    const auto& conformal_metric = cache->get_var(
        *this, Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>{});
    partial_derivative(deriv_conformal_metric, conformal_metric,
                       this->mesh->get(), this->inv_jacobian->get());
  } else {
    (void)deriv_conformal_metric;
    (void)cache;
    ERROR(
        "The derivative of the perturbed conformal metric is computed "
        "numerically and requires DataVector grid data.");
  }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__)

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const {
  const auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataType, Dim>>(kerr_schild_vars.get());
  const auto& inv_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataType, Dim>>(
          kerr_schild_vars.get());
  trace(trace_extrinsic_curvature, extrinsic_curvature, inv_spatial_metric);
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const {
  get(*dt_trace_extrinsic_curvature) = 0.;
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_factor_minus_one,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ConformalFactorMinusOne<DataType> /*meta*/) const {
  get(*conformal_factor_minus_one) = 0.;
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::LapseTimesConformalFactorMinusOne<DataType> /*meta*/) const {
  *lapse_times_conformal_factor_minus_one =
      get<gr::Tags::Lapse<DataType>>(kerr_schild_vars.get());
  get(*lapse_times_conformal_factor_minus_one) -= 1.;
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/)
    const {
  std::fill(shift_background->begin(), shift_background->end(), 0.);
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<tnsr::iJ<DataType, Dim>*> deriv_shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  std::fill(deriv_shift_background->begin(), deriv_shift_background->end(), 0.);
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, Dim, Frame::Inertial>*>
        longitudinal_shift_background_minus_dt_conformal_metric,
    const gsl::not_null<Cache*> cache,
    Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, Dim, Frame::Inertial> /*meta*/) const {
  const auto& dt_teukolsky_metric =
      get<::Tags::dt<gr::Tags::SpatialMetric<DataType, Dim, Frame::Inertial>>>(
          teukolsky_vars.get());
  const auto& conformal_metric = cache->get_var(
      *this, Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      *this,
      Xcts::Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial>{});

  const auto trace_dt_metric = trace(dt_teukolsky_metric, inv_conformal_metric);

  tnsr::ii<DataType, Dim, Frame::Inertial> trace_free_dt_metric{
      get_size(*x.get().begin())};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      trace_free_dt_metric.get(i, j) =
          dt_teukolsky_metric.get(i, j) -
          conformal_metric.get(i, j) * get(trace_dt_metric) / 3.;
    }
  }

  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      longitudinal_shift_background_minus_dt_conformal_metric->get(i, j) = 0.;
      for (size_t k = 0; k < Dim; ++k) {
        for (size_t l = 0; l < Dim; ++l) {
          longitudinal_shift_background_minus_dt_conformal_metric->get(i, j) -=
              inv_conformal_metric.get(i, k) * inv_conformal_metric.get(j, l) *
              trace_free_dt_metric.get(k, l);
        }
      }
    }
  }
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/) const {
  *shift_excess = get<gr::Tags::Shift<DataType, Dim>>(kerr_schild_vars.get());
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> energy_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0> /*meta*/) const {
  get(*energy_density) = 0.;
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> stress_trace,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0> /*meta*/) const {
  get(*stress_trace) = 0.;
}

template <typename DataType>
void KerrSchildTeukolskyVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> momentum_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, Dim>, 0> /*meta*/)
    const {
  std::fill(momentum_density->begin(), momentum_density->end(), 0.);
}

template class KerrSchildTeukolskyVariables<double>;
template class KerrSchildTeukolskyVariables<DataVector>;

}  // namespace Xcts::AnalyticData::detail

namespace Xcts::AnalyticData {

KerrSchildTeukolsky::KerrSchildTeukolsky(
    gr::Solutions::KerrSchild kerr_schild,
    const gr::Solutions::TeukolskyWave& teukolsky_wave)
    : kerr_schild_(std::move(kerr_schild)),
      teukolsky_wave_(teukolsky_wave.amplitude(), teukolsky_wave.mode(),
                      teukolsky_wave.parity(), teukolsky_wave.direction(),
                      teukolsky_wave.center(), teukolsky_wave.radius(),
                      teukolsky_wave.width(), false) {}

void KerrSchildTeukolsky::pup(PUP::er& p) {
  elliptic::analytic_data::Background::pup(p);
  elliptic::analytic_data::InitialGuess::pup(p);
  p | kerr_schild_;
  p | teukolsky_wave_;
}

bool operator==(const KerrSchildTeukolsky& lhs,
                const KerrSchildTeukolsky& rhs) {
  return lhs.kerr_schild_ == rhs.kerr_schild_ and
         lhs.teukolsky_wave_ == rhs.teukolsky_wave_;
}

bool operator!=(const KerrSchildTeukolsky& lhs,
                const KerrSchildTeukolsky& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID KerrSchildTeukolsky::my_PUP_ID = 0;  // NOLINT

}  // namespace Xcts::AnalyticData

template class Xcts::AnalyticData::CommonVariables<
    double, typename Xcts::AnalyticData::detail::KerrSchildTeukolskyVariables<
                double>::Cache>;
template class Xcts::AnalyticData::CommonVariables<
    DataVector, typename Xcts::AnalyticData::detail::
                    KerrSchildTeukolskyVariables<DataVector>::Cache>;
