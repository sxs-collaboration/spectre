// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/WavyBBH.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.tpp"
#include "PointwiseFunctions/Elasticity/Strain.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Xcts::AnalyticData {

namespace detail {

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> radius_left,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::Radius_Left<DataType> /*meta*/) const {
  const auto v_x = get<0>(x) - xcoord_left;
  const auto v_y = get<1>(x);
  const auto v_z = get<2>(x);
  get(*radius_left) = sqrt(square(v_x) + square(v_y) + square(v_z));
}

template <typename DataVector>
void WavyBBHVariables<DataVector>::operator()(
    const gsl::not_null<Scalar<DataVector>*> radius_right,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::Radius_Right<DataVector> /*meta*/) const {
  const auto v_x = get<0>(x) - xcoord_right;
  const auto v_y = get<1>(x);
  const auto v_z = get<2>(x);
  get(*radius_right) = sqrt(square(v_x) + square(v_y) + square(v_z));
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> normal_left,
    const gsl::not_null<Cache*> cache,
    detail::Tags::Normal_Left<DataType> /*meta*/) const {
  const auto& radius_left =
      get(cache->get_var(*this, detail::Tags::Radius_Left<DataType>{}));
  get<0>(*normal_left) = (get<0>(x) - xcoord_left) / radius_left;
  get<1>(*normal_left) = get<1>(x) / radius_left;
  get<2>(*normal_left) = get<2>(x) / radius_left;
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> normal_right,
    const gsl::not_null<Cache*> cache,
    detail::Tags::Normal_Right<DataType> /*meta*/) const {
  const auto& radius_right =
      get(cache->get_var(*this, detail::Tags::Radius_Right<DataType>{}));
  get<0>(*normal_right) = (get<0>(x) - xcoord_right) / radius_right;
  get<1>(*normal_right) = get<1>(x) / radius_right;
  get<2>(*normal_right) = get<2>(x) / radius_right;
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> conformal_metric,
    const gsl::not_null<Cache*> cache,
    Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial> /*meta*/) const {
  const auto& radius_left =
      get(cache->get_var(*this, detail::Tags::Radius_Left<DataType>{}));
  const auto& radius_right =
      get(cache->get_var(*this, detail::Tags::Radius_Right<DataType>{}));
  const auto E_left = mass_left + square(ymomentum_left) / (2 * mass_left) -
                      mass_left * mass_right / separation;
  const auto E_right = mass_right + square(ymomentum_right) / (2 * mass_right) -
                       mass_left * mass_right / separation;
  const auto Phi_PN =
      1. + E_left / (2 * radius_left) + E_right / (2 * radius_right);
  get<0, 0>(*conformal_metric) = Phi_PN;
  get<1, 1>(*conformal_metric) = Phi_PN;
  get<2, 2>(*conformal_metric) = Phi_PN;
  get<0, 1>(*conformal_metric) = 0.;
  get<0, 2>(*conformal_metric) = 0.;
  get<1, 2>(*conformal_metric) = 0.;
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3>*> deriv_conformal_metric,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial> /*meta*/) const {
  std::fill(deriv_conformal_metric->begin(), deriv_conformal_metric->end(), 0.);
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const {
  get(*trace_extrinsic_curvature) = 0.;
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const {
  get(*dt_trace_extrinsic_curvature) = 0.;
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial> /*meta*/) const {
  std::fill(shift_background->begin(), shift_background->end(), 0.);
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3, Frame::Inertial>*>
        longitudinal_shift_background_minus_dt_conformal_metric,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial> /*meta*/) const {
  std::fill(longitudinal_shift_background_minus_dt_conformal_metric->begin(),
            longitudinal_shift_background_minus_dt_conformal_metric->end(), 0.);
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    gsl::not_null<tnsr::iJ<DataType, Dim>*> deriv_shift_background,
    gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  std::fill(deriv_shift_background->begin(), deriv_shift_background->end(), 0.);
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_energy_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0> /*meta*/) const {
  get(*conformal_energy_density) = 0;
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_stress_trace,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0> /*meta*/) const {
  get(*conformal_stress_trace) = 0.;
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> conformal_momentum_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, Dim>, 0> /*meta*/)
    const {
  std::fill(conformal_momentum_density->begin(),
            conformal_momentum_density->end(), 0.);
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_factor_minus_one,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ConformalFactorMinusOne<DataType> /*meta*/) const {
  get(*conformal_factor_minus_one) = 0.;
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::LapseTimesConformalFactorMinusOne<DataType> /*meta*/) const {
  get(*lapse_times_conformal_factor_minus_one) = 0.;
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/) const {
  std::fill(shift_excess->begin(), shift_excess->end(), 0.);
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> rest_mass_density,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::RestMassDensity<DataType> /*meta*/) const {
  get(*rest_mass_density) = 0.;
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> specific_enthalpy,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::SpecificEnthalpy<DataType> /*meta*/) const {
  get(*specific_enthalpy) = 0.;
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> pressure,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::Pressure<DataType> /*meta*/) const {
  get(*pressure) = 0.;
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> spatial_velocity,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::SpatialVelocity<DataType, 3> /*meta*/) const {
  std::fill(spatial_velocity->begin(), spatial_velocity->end(), 0.);
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> lorentz_factor,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::LorentzFactor<DataType> /*meta*/) const {
  get(*lorentz_factor) = 0.;
}

template <typename DataType>
void WavyBBHVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> magnetic_field,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::MagneticField<DataType, 3> /*meta*/) const {
  std::fill(magnetic_field->begin(), magnetic_field->end(), 0.);
}

template class WavyBBHVariables<DataVector>;

}  // namespace detail

PUP::able::PUP_ID WavyBBH::my_PUP_ID = 0;  // NOLINT

}  // namespace Xcts::AnalyticData

template class Xcts::AnalyticData::CommonVariables<
    DataVector,
    typename Xcts::AnalyticData::detail::WavyBBHVariables<DataVector>::Cache>;
