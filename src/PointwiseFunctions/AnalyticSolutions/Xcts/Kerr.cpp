// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Xcts/Kerr.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/CommonVariables.tpp"
#include "PointwiseFunctions/Elasticity/Strain.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts::Solutions::detail {

template <typename DataType>
void KerrVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalMetric<DataType, Dim, Frame::Inertial> /*meta*/) const {
  const auto& conformal_factor =
      cache->get_var(Xcts::Tags::ConformalFactor<DataType>{});
  *conformal_metric = kerr_schild.get_var(
      gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>{});
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      conformal_metric->get(i, j) /= pow<4>(get(conformal_factor));
    }
  }
}

template <typename DataType>
void KerrVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, Dim>*> inv_conformal_metric,
    const gsl::not_null<Cache*> cache,
    Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial> /*meta*/)
    const {
  const auto& conformal_factor =
      cache->get_var(Xcts::Tags::ConformalFactor<DataType>{});
  *inv_conformal_metric = kerr_schild.get_var(
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataType>{});
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      inv_conformal_metric->get(i, j) *= pow<4>(get(conformal_factor));
    }
  }
}

template <typename DataType>
void KerrVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_conformal_metric,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  const auto& conformal_factor =
      cache->get_var(Xcts::Tags::ConformalFactor<DataType>{});
  const auto& deriv_conformal_factor =
      cache->get_var(::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                   tmpl::size_t<Dim>, Frame::Inertial>{});
  const auto& conformal_metric = cache->get_var(
      Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>{});
  *deriv_conformal_metric = kerr_schild.get_var(
      ::Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
                    tmpl::size_t<Dim>, Frame::Inertial>{});
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      for (size_t k = 0; k <= j; ++k) {
        deriv_conformal_metric->get(i, j, k) /= pow<4>(get(conformal_factor));
        deriv_conformal_metric->get(i, j, k) -= 4. / get(conformal_factor) *
                                                conformal_metric.get(j, k) *
                                                deriv_conformal_factor.get(i);
      }
    }
  }
}

template <typename DataType>
void KerrVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const {
  const auto& extrinsic_curvature = kerr_schild.get_var(
      gr::Tags::ExtrinsicCurvature<Dim, Frame::Inertial, DataType>{});
  const auto& inv_spatial_metric = kerr_schild.get_var(
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataType>{});
  trace(trace_extrinsic_curvature, extrinsic_curvature, inv_spatial_metric);
}

template <typename DataType>
void KerrVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const {
  get(*dt_trace_extrinsic_curvature) = 0.;
}

template <typename DataType>
void KerrVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_factor,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::ConformalFactor<DataType> /*meta*/) const {
  get(*conformal_factor) = 1.;
}

template <typename DataType>
void KerrVariables<DataType>::operator()(
    const gsl::not_null<tnsr::i<DataType, Dim>*> conformal_factor_gradient,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::ConformalFactor<DataType>, tmpl::size_t<Dim>,
                  Frame::Inertial> /*meta*/) const {
  std::fill(conformal_factor_gradient->begin(),
            conformal_factor_gradient->end(), 0.);
}

template <typename DataType>
void KerrVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Lapse<DataType> /*meta*/) const {
  *lapse = kerr_schild.get_var(gr::Tags::Lapse<DataType>{});
}

template <typename DataType>
void KerrVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor,
    const gsl::not_null<Cache*> cache,
    Tags::LapseTimesConformalFactor<DataType> /*meta*/) const {
  *lapse_times_conformal_factor = cache->get_var(gr::Tags::Lapse<DataType>{});
  const auto& conformal_factor =
      cache->get_var(Xcts::Tags::ConformalFactor<DataType>{});
  get(*lapse_times_conformal_factor) *= get(conformal_factor);
}

template <typename DataType>
void KerrVariables<DataType>::operator()(
    const gsl::not_null<tnsr::i<DataType, Dim>*>
        lapse_times_conformal_factor_gradient,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<Dim>,
                  Frame::Inertial> /*meta*/) const {
  const auto& conformal_factor =
      cache->get_var(Xcts::Tags::ConformalFactor<DataType>{});
  const auto& deriv_conformal_factor =
      cache->get_var(::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                   tmpl::size_t<Dim>, Frame::Inertial>{});
  *lapse_times_conformal_factor_gradient =
      kerr_schild.get_var(::Tags::deriv<gr::Tags::Lapse<DataType>,
                                        tmpl::size_t<Dim>, Frame::Inertial>{});
  const auto& lapse = kerr_schild.get_var(gr::Tags::Lapse<DataType>{});
  for (size_t i = 0; i < Dim; ++i) {
    lapse_times_conformal_factor_gradient->get(i) *= get(conformal_factor);
    lapse_times_conformal_factor_gradient->get(i) +=
        get(lapse) * deriv_conformal_factor.get(i);
  }
}

template <typename DataType>
void KerrVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/) const {
  std::fill(shift_background->begin(), shift_background->end(), 0.);
}

template <typename DataType>
void KerrVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, Dim, Frame::Inertial>*>
        longitudinal_shift_background_minus_dt_conformal_metric,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, Dim, Frame::Inertial> /*meta*/) const {
  std::fill(longitudinal_shift_background_minus_dt_conformal_metric->begin(),
            longitudinal_shift_background_minus_dt_conformal_metric->end(), 0.);
}

template <typename DataType>
void KerrVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::ShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/) const {
  *shift_excess =
      kerr_schild.get_var(gr::Tags::Shift<Dim, Frame::Inertial, DataType>{});
}

template <typename DataType>
void KerrVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, Dim>*> shift_strain,
    const gsl::not_null<Cache*> cache,
    Tags::ShiftStrain<DataType, Dim, Frame::Inertial> /*meta*/) const {
  const auto& shift =
      kerr_schild.get_var(gr::Tags::Shift<Dim, Frame::Inertial, DataType>{});
  const auto& deriv_shift = kerr_schild.get_var(
      ::Tags::deriv<gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
                    tmpl::size_t<Dim>, Frame::Inertial>{});
  const auto& conformal_metric = cache->get_var(
      Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>{});
  const auto& deriv_conformal_metric = cache->get_var(
      ::Tags::deriv<Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial>{});
  const auto& conformal_christoffel_first_kind = cache->get_var(
      Tags::ConformalChristoffelFirstKind<DataType, Dim, Frame::Inertial>{});
  Elasticity::strain(shift_strain, deriv_shift, conformal_metric,
                     deriv_conformal_metric, conformal_christoffel_first_kind,
                     shift);
}

template <typename DataType>
void KerrVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> energy_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0> /*meta*/) const {
  std::fill(energy_density->begin(), energy_density->end(), 0.);
}

template <typename DataType>
void KerrVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> stress_trace,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0> /*meta*/) const {
  std::fill(stress_trace->begin(), stress_trace->end(), 0.);
}

template <typename DataType>
void KerrVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> momentum_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<
        gr::Tags::MomentumDensity<Dim, Frame::Inertial, DataType>, 0> /*meta*/)
    const {
  std::fill(momentum_density->begin(), momentum_density->end(), 0.);
}

template class KerrVariables<double>;
template class KerrVariables<DataVector>;

}  // namespace Xcts::Solutions::detail

// Instantiate implementations for common variables
template class Xcts::Solutions::CommonVariables<
    double, typename Xcts::Solutions::detail::KerrVariables<double>::Cache>;
template class Xcts::Solutions::CommonVariables<
    DataVector,
    typename Xcts::Solutions::detail::KerrVariables<DataVector>::Cache>;
template class Xcts::AnalyticData::CommonVariables<
    double, typename Xcts::Solutions::detail::KerrVariables<double>::Cache>;
template class Xcts::AnalyticData::CommonVariables<
    DataVector,
    typename Xcts::Solutions::detail::KerrVariables<DataVector>::Cache>;
