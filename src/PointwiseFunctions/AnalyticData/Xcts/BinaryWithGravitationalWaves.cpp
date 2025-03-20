// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/BinaryWithGravitationalWaves.hpp"

#include <boost/math/interpolators/cubic_hermite.hpp>
#include <brigand/brigand.hpp>

#include <array>
#include <cstddef>
#include <iomanip>

#include "DataStructures/BoostMultiArray.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Expressions/TensorIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/SpecialRelativity/LorentzBoostMatrix.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

// Boost MultiArray is used internally in odeint, so odeint must be included
// later
#include <boost/numeric/odeint.hpp>

namespace Xcts::AnalyticData::detail {

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> conformal_metric,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial> /*meta*/) const {
  const auto& conformal_metric_aux = get_t_conformal_metric(present_time);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      conformal_metric->get(i, j) = conformal_metric_aux.get(i, j);
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3>*> deriv_conformal_metric,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  ASSERT(mesh.has_value() and inv_jacobian.has_value(),
         "Need a mesh and a Jacobian for numeric differentiation.");
  if constexpr (std::is_same_v<DataType, DataVector>) {
    const auto& conformal_metric = cache->get_var(
        *this, Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>{});
    partial_derivative(deriv_conformal_metric, conformal_metric, mesh->get(),
                       inv_jacobian->get());
  } else {
    (void)deriv_conformal_metric;
    (void)cache;
    ERROR(
        "Numeric differentiation only works with DataVectors because it needs "
        "a grid.");
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const {
  ASSERT(mesh.has_value() and inv_jacobian.has_value(),
         "Need a mesh and a Jacobian for numeric differentiation.");
  if constexpr (std::is_same_v<DataType, DataVector>) {
    get(*trace_extrinsic_curvature) = get(get_t_trace_extrinsic_curvature(
        present_time, mesh->get(), inv_jacobian->get()));
  } else {
    (void)trace_extrinsic_curvature;
    ERROR(
        "Numeric differentiation only works with DataVectors because it needs "
        "a grid.");
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
    const gsl::not_null<Cache*> cache,
    ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const {
  const auto& trace_extrinsic_curvature =
      cache->get_var(*this, gr::Tags::TraceExtrinsicCurvature<DataType>{});
  DataType time_back(get(trace_extrinsic_curvature).size(), -time_displacement);
  DataType time_back_two(get(trace_extrinsic_curvature).size(),
                         -2. * time_displacement);
  DataType time_back_three(get(trace_extrinsic_curvature).size(),
                           -3. * time_displacement);
  Scalar<DataType> trace_extrinsic_curvature_back;
  Scalar<DataType> trace_extrinsic_curvature_back_two;
  Scalar<DataType> trace_extrinsic_curvature_back_three;
  ASSERT(mesh.has_value() and inv_jacobian.has_value(),
         "Need a mesh and a Jacobian for numeric differentiation.");
  if constexpr (std::is_same_v<DataType, DataVector>) {
    trace_extrinsic_curvature_back = get_t_trace_extrinsic_curvature(
        time_back, mesh->get(), inv_jacobian->get());
    trace_extrinsic_curvature_back_two = get_t_trace_extrinsic_curvature(
        time_back_two, mesh->get(), inv_jacobian->get());
    trace_extrinsic_curvature_back_three = get_t_trace_extrinsic_curvature(
        time_back_three, mesh->get(), inv_jacobian->get());
  } else {
    (void)dt_trace_extrinsic_curvature;
    (void)cache;
    ERROR(
        "Numeric differentiation only works with DataVectors because it needs "
        "a grid.");
  }
  // Third order backwards finite difference
  get(*dt_trace_extrinsic_curvature) =
      (11. * get(trace_extrinsic_curvature) -
       18. * get(trace_extrinsic_curvature_back) +
       9. * get(trace_extrinsic_curvature_back_two) -
       2. * get(trace_extrinsic_curvature_back_three)) /
      (6. * time_displacement);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3, Frame::Inertial>*>
        longitudinal_shift_background_minus_dt_conformal_metric,
    const gsl::not_null<Cache*> cache,
    Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial> /*meta*/) const {
  std::fill(longitudinal_shift_background_minus_dt_conformal_metric->begin(),
            longitudinal_shift_background_minus_dt_conformal_metric->end(), 0.);
  // LongitudinalShiftBackground is zero
  // DtConformalMetric (finite difference 3rd order)
  const auto& conformal_metric = cache->get_var(
      *this, Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      *this,
      ::Xcts::Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial>{});
  DataType time_back(get_size(x.get(0)), -time_displacement);
  DataType time_back_two(get_size(x.get(0)), -2. * time_displacement);
  DataType time_back_three(get_size(x.get(0)), -3. * time_displacement);
  const auto conformal_metric_back = get_t_conformal_metric(time_back);
  const auto conformal_metric_back_two = get_t_conformal_metric(time_back_two);
  const auto conformal_metric_back_three =
      get_t_conformal_metric(time_back_three);

  tnsr::ii<DataType, 3> dt_conformal_metric{get_size(x.get(0))};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      // Third order backwards finite difference
      dt_conformal_metric.get(i, j) =
          (11. * conformal_metric.get(i, j) -
           18. * conformal_metric_back.get(i, j) +
           9. * conformal_metric_back_two.get(i, j) -
           2. * conformal_metric_back_three.get(i, j)) /
          (6. * time_displacement);
    }
  }

  const auto auxiliar_one = tenex::evaluate<ti::I, ti::J>(
      inv_conformal_metric(ti::I, ti::K) * inv_conformal_metric(ti::J, ti::L) *
      dt_conformal_metric(ti::k, ti::l));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      longitudinal_shift_background_minus_dt_conformal_metric->get(i, j) -=
          (auxiliar_one.get(i, j) -
           (1. / 3.) * get(trace(dt_conformal_metric, inv_conformal_metric)) *
               inv_conformal_metric.get(i, j));
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_factor_minus_one,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ConformalFactorMinusOne<DataType> /*meta*/) const {
  get(*conformal_factor_minus_one) = 0.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<Cache*> cache,
    Xcts::Tags::LapseTimesConformalFactorMinusOne<DataType> /*meta*/) const {
  const auto& conformal_factor_minus_one =
      cache->get_var(*this, Xcts::Tags::ConformalFactorMinusOne<DataType>{});
  const auto lapse = get_t_lapse(present_time);
  get(*lapse_times_conformal_factor_minus_one) =
      get(lapse) * (get(conformal_factor_minus_one) + 1.) - 1.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/) const {
  std::fill(shift_excess->begin(), shift_excess->end(), 0.);
  const auto shift_excess_aux = get_t_shift(present_time);
  for (size_t i = 0; i < 3; ++i) {
    shift_excess->get(i) += shift_excess_aux.get(i);
  }
}

// Private functions

template <typename DataType>
Scalar<DataType> BinaryWithGravitationalWavesVariables<DataType>::
    get_t_trace_extrinsic_curvature(
        DataType t, Mesh<3> local_mesh,
        InverseJacobian<DataType, 3, Frame::ElementLogical, Frame::Inertial>
            local_inv_jacobian) const {
  tnsr::ii<DataType, 3> extrinsic_curvature{t.size()};
  std::fill(extrinsic_curvature.begin(), extrinsic_curvature.end(), 0.);

  DataType time_back(get_size(x.get(0)), t[0] - time_displacement);
  DataType time_back_two(get_size(x.get(0)), t[0] - 2. * time_displacement);
  DataType time_back_three(get_size(x.get(0)), t[0] - 3. * time_displacement);
  const auto conformal_metric = get_t_conformal_metric(t);
  const auto inv_conformal_metric =
      determinant_and_inverse(conformal_metric).second;
  const auto conformal_metric_back = get_t_conformal_metric(time_back);
  const auto conformal_metric_back_two = get_t_conformal_metric(time_back_two);
  const auto conformal_metric_back_three =
      get_t_conformal_metric(time_back_three);
  const auto lapse = get_t_lapse(t);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      // Third order backwards finite difference (over -2 alpha)
      extrinsic_curvature.get(i, j) +=
          (11. * conformal_metric.get(i, j) -
           18. * conformal_metric_back.get(i, j) +
           9. * conformal_metric_back_two.get(i, j) -
           2. * conformal_metric_back_three.get(i, j)) /
          (6. * time_displacement * (-2.) * get(lapse));
    }
  }

  const auto shift = get_t_shift(t);
  const auto shift_down = raise_or_lower_index(shift, conformal_metric);
  const auto deriv_shift =
      partial_derivative(shift_down, local_mesh, local_inv_jacobian);
  const auto deriv_conformal_metric =
      partial_derivative(conformal_metric, local_mesh, local_inv_jacobian);
  const auto christoffel_second_kind =
      gr::christoffel_second_kind(deriv_conformal_metric, inv_conformal_metric);
  const auto covariant_deriv_shift_contribution = tenex::evaluate<ti::i, ti::j>(
      deriv_shift(ti::i, ti::j) -
      christoffel_second_kind(ti::K, ti::i, ti::j) * shift_down(ti::k) +
      deriv_shift(ti::j, ti::i) -
      christoffel_second_kind(ti::K, ti::j, ti::i) * shift_down(ti::k));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      extrinsic_curvature.get(i, j) +=
          covariant_deriv_shift_contribution.get(i, j) / (2. * get(lapse));
    }
  }
  return trace(extrinsic_curvature, inv_conformal_metric);
}

template <typename DataType>
tnsr::ii<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_conformal_metric(
    DataType t) const {
  const auto radiative_term = get_t_radiative_term(t);
  const auto superposed_spacetime_metric_t =
      get_t_superposed_spacetime_metric(t);
  auto conformal_metric = gr::spatial_metric(superposed_spacetime_metric_t);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      conformal_metric.get(i, j) += radiative_term.get(i, j);
    }
  }
  return conformal_metric;
}

template <typename DataType>
DataType
BinaryWithGravitationalWavesVariables<DataType>::get_t_attenuation_function(
    DataType t) const {
  const auto distance_left_t = get_t_distance_left(t);
  const auto distance_right_t = get_t_distance_right(t);
  double turn_off = .5;
  if (attenuation_parameter == 0) {
    turn_off = 1.;
  }
  return (turn_off + .5 * tanh(attenuation_parameter *
                               (get(distance_left_t) - attenuation_radius))) *
         (turn_off + .5 * tanh(attenuation_parameter *
                               (get(distance_right_t) - attenuation_radius)));
}

template <typename DataType>
tnsr::ii<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_radiative_term(
    DataType t) const {
  const auto distance_left_t = get_t_distance_left(t);
  const auto distance_right_t = get_t_distance_right(t);
  const auto near_zone_term_t = get_t_near_zone_term(t);
  const auto present_term_t = get_t_present_term(t);
  const auto past_term_t = get_t_past_term(t);
  const auto integral_term_t = get_t_integral_term(t);
  const auto attenuation_t = get_t_attenuation_function(t);
  tnsr::ii<DataType, 3> radiative_term_t{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      radiative_term_t.get(i, j) =
          attenuation_t *
          (near_zone_term_t.get(i, j) + present_term_t.get(i, j) +
           past_term_t.get(i, j) + integral_term_t.get(i, j));
    }
  }
  return radiative_term_t;
}

template <typename DataType>
tnsr::ii<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_near_zone_term(
    DataType t) const {
  // Computation will be added in the future
  tnsr::ii<DataType, 3> near_zone_term_t{t.size()};
  std::fill(near_zone_term_t.begin(), near_zone_term_t.end(), 0.);
  return near_zone_term_t;
}

template <typename DataType>
tnsr::ii<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_present_term(
    DataType t) const {
  // Computation will be added in the future
  tnsr::ii<DataType, 3> present_term_t{t.size()};
  std::fill(present_term_t.begin(), present_term_t.end(), 0.);
  return present_term_t;
}

template <typename DataType>
tnsr::ii<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_past_term(
    DataType t) const {
  // Computation will be added in the future
  tnsr::ii<DataType, 3> past_term_t{t.size()};
  std::fill(past_term_t.begin(), past_term_t.end(), 0.);
  return past_term_t;
}

template <typename DataType>
tnsr::ii<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_integral_term(
    DataType t) const {
  // Computation will be added in the future
  tnsr::ii<DataType, 3> integral_term_t{t.size()};
  std::fill(integral_term_t.begin(), integral_term_t.end(), 0.);
  return integral_term_t;
}

template <typename DataType>
Scalar<DataType> BinaryWithGravitationalWavesVariables<DataType>::get_t_lapse(
    DataType t) const {
  Scalar<DataType> lapse_t{t.size()};
  std::fill(lapse_t.begin(), lapse_t.end(), 0.);

  // PN Lapse in the far zone
  const auto distance_left_t = get_t_distance_left(t);
  const auto distance_right_t = get_t_distance_right(t);
  const auto separation_t = get_t_separation(t);
  const auto momentum_left_t = get_t_momentum_left(t);
  const auto momentum_right_t = get_t_momentum_right(t);
  const auto attenuation_t = get_t_attenuation_function(t);
  get(lapse_t) =
      1 - mass_left / get(distance_left_t) -
      mass_right / get(distance_right_t) +
      (mass_left * mass_right) /
          (get(distance_left_t) * get(distance_right_t)) +
      (mass_left * mass_right) / (get(distance_left_t) * get(separation_t)) +
      (mass_left * mass_right) / (get(distance_right_t) * get(separation_t)) +
      0.5 * (square(mass_left / get(distance_left_t)) +
             square(mass_right / get(distance_right_t)) -
             3. * (get(dot_product(momentum_left_t, momentum_left_t)) /
                       (get(distance_left_t) * mass_left) +
                   get(dot_product(momentum_right_t, momentum_right_t)) /
                       (get(distance_right_t) * mass_right)));
  get(lapse_t) *= attenuation_t;

  // Boosted lapse in the inner zone
  const auto spacetime_metric_left = get_t_boosted_spacetime_metric_left(t);
  const auto conformal_metric_left = gr::spatial_metric(spacetime_metric_left);
  const auto inv_conformal_metric_left =
      determinant_and_inverse(conformal_metric_left).second;
  const auto shift_left =
      gr::shift(spacetime_metric_left, inv_conformal_metric_left);
  const auto lapse_left = gr::lapse(shift_left, spacetime_metric_left);
  const auto spacetime_metric_right = get_t_boosted_spacetime_metric_right(t);
  const auto conformal_metric_right =
      gr::spatial_metric(spacetime_metric_right);
  const auto inv_conformal_metric_right =
      determinant_and_inverse(conformal_metric_right).second;
  const auto shift_right =
      gr::shift(spacetime_metric_right, inv_conformal_metric_right);
  const auto lapse_right = gr::lapse(shift_right, spacetime_metric_right);
  get(lapse_t) += (1. - attenuation_t) * get(lapse_left) * get(lapse_right);
  return lapse_t;
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_shift(DataType t) const {
  tnsr::I<DataType, 3> shift_t{t.size()};
  std::fill(shift_t.begin(), shift_t.end(), 0.);

  // PN shift in the far zone
  const auto distance_left_t = get_t_distance_left(present_time);
  const auto distance_right_t = get_t_distance_right(present_time);
  const auto separation_t = get_t_separation(present_time);
  const auto momentum_left_t = get_t_momentum_left(present_time);
  const auto momentum_right_t = get_t_momentum_right(present_time);
  const auto normal_left_t = get_t_normal_left(present_time);
  const auto normal_right_t = get_t_normal_right(present_time);
  const auto attenuation_t = get_t_attenuation_function(present_time);
  for (size_t i = 0; i < 3; ++i) {
    shift_t.get(i) -= 4. * (momentum_left_t.get(i) / get(distance_left_t) +
                            momentum_right_t.get(i) / get(distance_right_t));
    for (size_t j = 0; j < 3; ++j) {
      shift_t.get(i) += .5 * momentum_left_t.get(j) *
                            (-normal_left_t.get(i) * normal_left_t.get(j) /
                             get(distance_left_t)) +
                        .5 * momentum_right_t.get(j) *
                            (-normal_right_t.get(i) * normal_right_t.get(j) /
                             get(distance_right_t));
    }
    shift_t.get(i) += 0.5 * momentum_left_t.get(i) / get(distance_left_t) +
                      0.5 * momentum_right_t.get(i) / get(distance_right_t);
  }

  for (size_t i = 0; i < 3; ++i) {
    shift_t.get(i) *= attenuation_t;
  }

  // Boosted shifts in the inner zone
  const auto spacetime_metric_left = get_t_boosted_spacetime_metric_left(t);
  const auto conformal_metric_left = gr::spatial_metric(spacetime_metric_left);
  const auto inv_conformal_metric_left =
      determinant_and_inverse(conformal_metric_left).second;
  const auto shift_left =
      gr::shift(spacetime_metric_left, inv_conformal_metric_left);
  const auto lapse_left = gr::lapse(shift_left, spacetime_metric_left);
  const auto spacetime_metric_right = get_t_boosted_spacetime_metric_right(t);
  const auto conformal_metric_right =
      gr::spatial_metric(spacetime_metric_right);
  const auto inv_conformal_metric_right =
      determinant_and_inverse(conformal_metric_right).second;
  const auto shift_right =
      gr::shift(spacetime_metric_right, inv_conformal_metric_right);
  const auto lapse_right = gr::lapse(shift_right, spacetime_metric_right);
  for (size_t i = 0; i < 3; ++i) {
    shift_t.get(i) +=
        (1. - attenuation_t) * (shift_left.get(i) + shift_right.get(i));
  }
  return shift_t;
}

template <typename DataType>
Scalar<DataType>
BinaryWithGravitationalWavesVariables<DataType>::get_t_distance_left(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = x.get(j)[i] - interpolation_position_left.at(j)(time[i]);
    }
  }
  return magnitude(v);
}

template <typename DataType>
Scalar<DataType>
BinaryWithGravitationalWavesVariables<DataType>::get_t_distance_right(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = x.get(j)[i] - interpolation_position_right.at(j)(time[i]);
    }
  }
  return magnitude(v);
}

template <typename DataType>
Scalar<DataType>
BinaryWithGravitationalWavesVariables<DataType>::get_t_separation(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = interpolation_position_right.at(j)(time[i]) -
                    interpolation_position_left.at(j)(time[i]);
    }
  }
  return magnitude(v);
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_momentum_left(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = gsl::at(interpolation_momentum_left, j)(time[i]);
    }
  }
  return v;
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_momentum_right(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = gsl::at(interpolation_momentum_right, j)(time[i]);
    }
  }
  return v;
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_normal_left(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  const Scalar<DataType> distance_left = get_t_distance_left(time);
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = (x.get(j)[i] - interpolation_position_left.at(j)(time[i])) /
                    get(distance_left)[i];
    }
  }
  return v;
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_normal_right(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  const Scalar<DataType> distance_right = get_t_distance_right(time);
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] =
          (x.get(j)[i] - interpolation_position_right.at(j)(time[i])) /
          get(distance_right)[i];
    }
  }

  return v;
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_normal_lr(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  Scalar<DataType> separation_norm = get_t_separation(time);
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = (interpolation_position_right.at(j)(time[i]) -
                     interpolation_position_left.at(j)(time[i])) /
                    get(separation_norm)[i];
    }
  }
  return v;
}

template <typename DataType>
tnsr::aa<DataType, 3> BinaryWithGravitationalWavesVariables<
    DataType>::get_t_boosted_spacetime_metric_left(DataType t) const {
  const auto& lapse_times_conformal_factor_minus_one =
      get<Tags::LapseTimesConformalFactorMinusOne<DataType>>(boost_vars[0]);
  const auto& conformal_factor_minus_one =
      get<Tags::ConformalFactorMinusOne<DataType>>(boost_vars[0]);
  const auto& shift =
      get<Tags::ShiftExcess<DataType, 3, Frame::Inertial>>(boost_vars[0]);

  Scalar<DataType> conformal_factor{t.size()};
  get(conformal_factor) = 1. + get(conformal_factor_minus_one);
  Scalar<DataType> lapse{t.size()};
  get(lapse) = (1. + get(lapse_times_conformal_factor_minus_one)) /
               get(conformal_factor);

  tnsr::ii<DataType, 3> spatial_metric{t.size()};
  std::fill(spatial_metric.begin(), spatial_metric.end(), 0.);
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric.get(i, i) = square(square(get(conformal_factor)));
  }

  tnsr::aa<DataType, 3> spacetime_metric{t.size()};
  gr::spacetime_metric(make_not_null(&spacetime_metric), lapse, shift,
                       spatial_metric);

  const tnsr::I<DataType, 3> momentum_Datavector = get_t_momentum_left(t);
  const std::array<double, 3> boost_velocity = {
      -get<0>(momentum_Datavector)[0] / mass_left,
      -get<1>(momentum_Datavector)[0] / mass_left,
      -get<2>(momentum_Datavector)[0] / mass_left};
  const tnsr::Ab<double, 3, Frame::NoFrame> lorentz_boost_matrix_double =
      sr::lorentz_boost_matrix(boost_velocity);
  tnsr::Ab<DataVector, 3> lorentz_boost_matrix{t.size()};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      lorentz_boost_matrix.get(i, j) = lorentz_boost_matrix_double.get(i, j);
    }
  }

  tnsr::aa<DataType, 3> spacetime_metric_boosted{t.size()};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      spacetime_metric_boosted.get(i, j) = 0.;
      for (size_t k = 0; k < 4; ++k) {
        for (size_t l = 0; l < 4; ++l) {
          spacetime_metric_boosted.get(i, j) += lorentz_boost_matrix.get(k, i) *
                                                lorentz_boost_matrix.get(l, j) *
                                                spacetime_metric.get(k, l);
        }
      }
    }
  }

  return spacetime_metric_boosted;
}

template <typename DataType>
tnsr::aa<DataType, 3> BinaryWithGravitationalWavesVariables<
    DataType>::get_t_boosted_spacetime_metric_right(DataType t) const {
  const auto& lapse_times_conformal_factor_minus_one =
      get<Tags::LapseTimesConformalFactorMinusOne<DataType>>(boost_vars[1]);
  const auto& conformal_factor_minus_one =
      get<Tags::ConformalFactorMinusOne<DataType>>(boost_vars[1]);
  const auto& shift =
      get<Tags::ShiftExcess<DataType, 3, Frame::Inertial>>(boost_vars[1]);

  Scalar<DataType> conformal_factor{t.size()};
  get(conformal_factor) = 1. + get(conformal_factor_minus_one);
  Scalar<DataType> lapse{t.size()};
  get(lapse) = (1. + get(lapse_times_conformal_factor_minus_one)) /
               get(conformal_factor);

  tnsr::ii<DataType, 3> spatial_metric{t.size()};
  std::fill(spatial_metric.begin(), spatial_metric.end(), 0.);
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric.get(i, i) = square(square(get(conformal_factor)));
  }

  tnsr::aa<DataType, 3> spacetime_metric{t.size()};
  gr::spacetime_metric(make_not_null(&spacetime_metric), lapse, shift,
                       spatial_metric);

  const tnsr::I<DataType, 3> momentum_Datavector = get_t_momentum_right(t);
  const std::array<double, 3> boost_velocity = {
      -get<0>(momentum_Datavector)[0] / mass_right,
      -get<1>(momentum_Datavector)[0] / mass_right,
      -get<2>(momentum_Datavector)[0] / mass_right};
  const tnsr::Ab<double, 3, Frame::NoFrame> lorentz_boost_matrix_double =
      sr::lorentz_boost_matrix(boost_velocity);
  tnsr::Ab<DataVector, 3> lorentz_boost_matrix{t.size()};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      lorentz_boost_matrix.get(i, j) = lorentz_boost_matrix_double.get(i, j);
    }
  }

  tnsr::aa<DataType, 3> spacetime_metric_boosted{t.size()};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      spacetime_metric_boosted.get(i, j) = 0.;
      for (size_t k = 0; k < 4; ++k) {
        for (size_t l = 0; l < 4; ++l) {
          spacetime_metric_boosted.get(i, j) += lorentz_boost_matrix.get(k, i) *
                                                lorentz_boost_matrix.get(l, j) *
                                                spacetime_metric.get(k, l);
        }
      }
    }
  }

  return spacetime_metric_boosted;
}

template <typename DataType>
tnsr::aa<DataType, 3> BinaryWithGravitationalWavesVariables<
    DataType>::get_t_superposed_spacetime_metric(DataType t) const {
  const auto spacetime_metric_left = get_t_boosted_spacetime_metric_left(t);
  const auto spacetime_metric_right = get_t_boosted_spacetime_metric_right(t);
  tnsr::aa<DataType, 3> superposed_spacetime_metric{t.size()};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      superposed_spacetime_metric.get(i, j) =
          spacetime_metric_left.get(i, j) + spacetime_metric_right.get(i, j);
    }
    superposed_spacetime_metric.get(i, i) -= 1.;
  }

  superposed_spacetime_metric.get(0, 0) += 2.;

  return superposed_spacetime_metric;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<
    DataType>::interpolate_past_history() {
  // Now interpolate the past history
  using boost::math::interpolators::cardinal_cubic_hermite;

  for (size_t i = 0; i < 3; ++i) {
    // static_cast being used because boost requires non-const arrays
    interpolation_position_left.at(i) = cardinal_cubic_hermite(
        static_cast<std::vector<double>>(past_position_left.at(i)),
        static_cast<std::vector<double>>(past_dt_position_left.at(i)),
        past_time.front(), std::abs(past_time.at(0) - past_time.at(1)));
    interpolation_position_right.at(i) = cardinal_cubic_hermite(
        static_cast<std::vector<double>>(past_position_right.at(i)),
        static_cast<std::vector<double>>(past_dt_position_right.at(i)),
        past_time.front(), std::abs(past_time.at(0) - past_time.at(1)));
    interpolation_momentum_left.at(i) = cardinal_cubic_hermite(
        static_cast<std::vector<double>>(past_momentum_left.at(i)),
        static_cast<std::vector<double>>(past_dt_momentum_left.at(i)),
        past_time.front(), std::abs(past_time.at(0) - past_time.at(1)));
    interpolation_momentum_right.at(i) = cardinal_cubic_hermite(
        static_cast<std::vector<double>>(past_momentum_right.at(i)),
        static_cast<std::vector<double>>(past_dt_momentum_right.at(i)),
        past_time.front(), std::abs(past_time.at(0) - past_time.at(1)));
  }
  // Get the domain of the interpolation to not trigger domain error on 0
  // (zero). The maximum time varies by machine roundoff. The interpolation is
  // done again because above is casted as std::function which does not have
  // access to the domain.
  present_time = make_with_value<DataVector>(
      x, cardinal_cubic_hermite(
             static_cast<std::vector<double>>(past_position_left.at(2)),
             static_cast<std::vector<double>>(past_dt_position_left.at(2)),
             past_time.front(), std::abs(past_time.at(0) - past_time.at(1)))
             .domain()
             .second);
}

template class BinaryWithGravitationalWavesVariables<DataVector>;

void BinaryWithGravitationalWavesHistory::reverse_vector() {
  std::reverse(past_time_.begin(), past_time_.end());
  for (size_t i = 0; i < 3; ++i) {
    reverse(past_position_left_.at(i).begin(), past_position_left_.at(i).end());
    reverse(past_position_right_.at(i).begin(),
            past_position_right_.at(i).end());
    reverse(past_momentum_left_.at(i).begin(), past_momentum_left_.at(i).end());
    reverse(past_momentum_right_.at(i).begin(),
            past_momentum_right_.at(i).end());
    reverse(past_dt_position_left_.at(i).begin(),
            past_dt_position_left_.at(i).end());
    reverse(past_dt_position_right_.at(i).begin(),
            past_dt_position_right_.at(i).end());
    reverse(past_dt_momentum_left_.at(i).begin(),
            past_dt_momentum_left_.at(i).end());
    reverse(past_dt_momentum_right_.at(i).begin(),
            past_dt_momentum_right_.at(i).end());
  }
}

void BinaryWithGravitationalWavesHistory::initialize() {
  const double separation = xcoords[1] - xcoords[0];

  total_mass = masses[0] + masses[1];
  reduced_mass = masses[0] * masses[1] / total_mass;
  reduced_mass_over_total_mass = reduced_mass / total_mass;

  initial_state_position = {{separation / total_mass, 0., 0.}};
  initial_state_momentum = {{momentum_right[0] / reduced_mass,
                             momentum_right[1] / reduced_mass,
                             momentum_right[2] / reduced_mass}};

  // Reserve vector capacity
  for (size_t i = 0; i < 3; ++i) {  // loop over x,y,z components
    past_position_left_.at(i).reserve(number_of_steps + 1);
    past_position_right_.at(i).reserve(number_of_steps + 1);
    past_momentum_left_.at(i).reserve(number_of_steps + 1);
    past_momentum_right_.at(i).reserve(number_of_steps + 1);
    past_dt_position_left_.at(i).reserve(number_of_steps + 1);
    past_dt_position_right_.at(i).reserve(number_of_steps + 1);
    past_dt_momentum_left_.at(i).reserve(number_of_steps + 1);
    past_dt_momentum_right_.at(i).reserve(number_of_steps + 1);
  }
  past_time_.reserve(number_of_steps + 1);
}

void BinaryWithGravitationalWavesHistory::hamiltonian_system(
    const BinaryWithGravitationalWavesHistory::state_type& x,
    BinaryWithGravitationalWavesHistory::state_type& dpdt) const {
  // H = H_Newt + H_1PN + H_2PN + H_3PN

  const double pdotp = x[3] * x[3] + x[4] * x[4] + x[5] * x[5];
  const double qdotq = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
  const double qdotp = x[0] * x[3] + x[1] * x[4] + x[2] * x[5];

  const double dH_dp0_Newt = x[3];
  const double dH_dp0_1 =
      0.5 * x[3] * pdotp * (-1. + 3. * reduced_mass_over_total_mass) -
      (x[0] * (x[4] * x[1] + x[5] * x[2]) * reduced_mass_over_total_mass +
       x[3] * (x[1] * x[1] + x[2] * x[2]) *
           (3. + reduced_mass_over_total_mass) +
       x[3] * x[0] * x[0] * (3. + 2. * reduced_mass_over_total_mass)) /
          std::sqrt(qdotq * qdotq * qdotq);
  const double dH_dp0_2 =
      0.125 *
      (3. * x[3] * pdotp * pdotp *
           (1. - 5. * reduced_mass_over_total_mass +
            5. * reduced_mass_over_total_mass * reduced_mass_over_total_mass) +
       (8. * (3. * x[0] * qdotp * reduced_mass_over_total_mass +
              x[3] * qdotq * (5. + 8. * reduced_mass_over_total_mass))) /
           (qdotq * qdotq) +
       (1. / std::sqrt(qdotq)) *
           (-(12. * x[0] * qdotp * qdotp * qdotp *
              reduced_mass_over_total_mass * reduced_mass_over_total_mass) /
                (qdotq * qdotq) -
            (4. * pdotp * x[0] * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            (4. * x[3] * qdotp * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            4. * x[3] * pdotp *
                (-5. + 20. * reduced_mass_over_total_mass +
                 3. * reduced_mass_over_total_mass *
                     reduced_mass_over_total_mass)));
  const double dH_dp0_3 =
      0.0625 *
      (5. * x[3] * pdotp * pdotp * pdotp *
           (-1. + 7. * (-1. + reduced_mass_over_total_mass) *
                      (-1. + reduced_mass_over_total_mass) *
                      reduced_mass_over_total_mass) +
       1.0 / (3. * qdotq * qdotq * qdotq) * 2 *
           (3. * pdotp * x[0] * qdotp * qdotq * reduced_mass_over_total_mass *
                (17 + 30 * reduced_mass_over_total_mass) +
            3. * x[3] * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17 + 30 * reduced_mass_over_total_mass) +
            8. * x[0] * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5 + 43 * reduced_mass_over_total_mass) +
            6. * x[3] * pdotp * qdotq * qdotq *
                (-27. + reduced_mass_over_total_mass *
                            (136. + 109. * reduced_mass_over_total_mass))) -
       (3. * x[0] * qdotp * reduced_mass_over_total_mass *
            (340. + 3. * M_PI * M_PI + 112. * reduced_mass_over_total_mass) +
        x[3] * qdotq *
            (600. + reduced_mass_over_total_mass *
                        (1340. - 3. * M_PI * M_PI +
                         552. * reduced_mass_over_total_mass))) /
           (6. * sqrt(qdotq * qdotq * qdotq * qdotq * qdotq)) +
       2. / (sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq)) *
           (pdotp * pdotp * x[0] * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass +
            2. * x[3] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            6. * pdotp * x[0] * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3. * x[3] * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1 + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            15. * x[0] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            3. * x[3] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7. +
                 reduced_mass_over_total_mass *
                     (-42. + reduced_mass_over_total_mass *
                                 (53. + 5. * reduced_mass_over_total_mass)))));

  const double dH_dp1_Newt = x[4];
  const double dH_dp1_1 =
      0.5 * x[4] * pdotp * (-1 + 3 * reduced_mass_over_total_mass) -
      (x[1] * (x[3] * x[0] + x[5] * x[2]) * reduced_mass_over_total_mass +
       x[4] * (x[0] * x[0] + x[2] * x[2]) * (3 + reduced_mass_over_total_mass) +
       x[4] * x[1] * x[1] * (3 + 2 * reduced_mass_over_total_mass)) /
          std::sqrt(qdotq * qdotq * qdotq);
  const double dH_dp1_2 =
      0.125 *
      (3. * x[4] * pdotp * pdotp *
           (1. - 5. * reduced_mass_over_total_mass +
            5. * reduced_mass_over_total_mass * reduced_mass_over_total_mass) +
       (8. * (3. * x[1] * qdotp * reduced_mass_over_total_mass +
              x[4] * qdotq * (5. + 8. * reduced_mass_over_total_mass))) /
           (qdotq * qdotq) +
       (1. / std::sqrt(qdotq)) *
           (-(12. * x[1] * qdotp * qdotp * qdotp *
              reduced_mass_over_total_mass * reduced_mass_over_total_mass) /
                (qdotq * qdotq) -
            (4. * pdotp * x[1] * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            (8. * x[4] * qdotp * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            4. * x[4] * pdotp *
                (-5. + 20. * reduced_mass_over_total_mass +
                 3. * reduced_mass_over_total_mass *
                     reduced_mass_over_total_mass)));
  const double dH_dp1_3 =
      0.0625 *
      (5. * x[4] * pdotp * pdotp * pdotp *
           (-1. + 7. * (-1. + reduced_mass_over_total_mass) *
                      (-1. + reduced_mass_over_total_mass) *
                      reduced_mass_over_total_mass) +
       1.0 / (3. * qdotq * qdotq * qdotq) * 2 *
           (3. * pdotp * x[1] * qdotp * qdotq * reduced_mass_over_total_mass *
                (17. + 30. * reduced_mass_over_total_mass) +
            3. * x[4] * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17. + 30. * reduced_mass_over_total_mass) +
            8. * x[1] * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5. + 43. * reduced_mass_over_total_mass) +
            6. * x[4] * pdotp * qdotq * qdotq *
                (-27. + reduced_mass_over_total_mass *
                            (136. + 109. * reduced_mass_over_total_mass))) -
       (3. * x[1] * qdotp * reduced_mass_over_total_mass *
            (340. + 3. * M_PI * M_PI + 112. * reduced_mass_over_total_mass) +
        x[4] * qdotq *
            (600. + reduced_mass_over_total_mass *
                        (1340. - 3. * M_PI * M_PI +
                         552. * reduced_mass_over_total_mass))) /
           (6. * sqrt(qdotq * qdotq * qdotq * qdotq * qdotq)) +
       2. / (sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq)) *
           (pdotp * pdotp * x[1] * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass +
            2. * x[4] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            6. * pdotp * x[1] * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3. * x[4] * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            15. * x[1] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            3. * x[4] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7. +
                 reduced_mass_over_total_mass *
                     (-42. + reduced_mass_over_total_mass *
                                 (53. + 5. * reduced_mass_over_total_mass)))));

  const double dH_dp2_Newt = x[5];
  const double dH_dp2_1 =
      0.5 * x[5] * pdotp * (-1 + 3 * reduced_mass_over_total_mass) -
      (x[2] * (x[4] * x[1] + x[3] * x[0]) * reduced_mass_over_total_mass +
       x[5] * (x[0] * x[0] + x[1] * x[1]) *
           (3. + reduced_mass_over_total_mass) +
       x[5] * x[2] * x[2] * (3. + 2. * reduced_mass_over_total_mass)) /
          std::sqrt(qdotq * qdotq * qdotq);
  const double dH_dp2_2 =
      0.125 *
      (3. * x[5] * pdotp * pdotp *
           (1. - 5. * reduced_mass_over_total_mass +
            5 * reduced_mass_over_total_mass * reduced_mass_over_total_mass) +
       (8. * (3. * x[2] * qdotp * reduced_mass_over_total_mass +
              x[5] * qdotq * (5. + 8. * reduced_mass_over_total_mass))) /
           (qdotq * qdotq) +
       (1. / std::sqrt(qdotq)) *
           (-(12. * x[2] * qdotp * qdotp * qdotp *
              reduced_mass_over_total_mass * reduced_mass_over_total_mass) /
                (qdotq * qdotq) -
            (4. * pdotp * x[2] * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            (8. * x[5] * qdotp * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            4. * x[5] * pdotp *
                (-5. + 20. * reduced_mass_over_total_mass +
                 3. * reduced_mass_over_total_mass *
                     reduced_mass_over_total_mass)));
  const double dH_dp2_3 =
      0.0625 *
      (5. * x[5] * pdotp * pdotp * pdotp *
           (-1. + 7. * (-1. + reduced_mass_over_total_mass) *
                      (-1. + reduced_mass_over_total_mass) *
                      reduced_mass_over_total_mass) +
       1.0 / (3. * qdotq * qdotq * qdotq) * 2. *
           (3. * pdotp * x[2] * qdotp * qdotq * reduced_mass_over_total_mass *
                (17. + 30. * reduced_mass_over_total_mass) +
            3. * x[5] * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17. + 30. * reduced_mass_over_total_mass) +
            8. * x[2] * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5. + 43. * reduced_mass_over_total_mass) +
            6. * x[5] * pdotp * qdotq * qdotq *
                (-27. + reduced_mass_over_total_mass *
                            (136. + 109. * reduced_mass_over_total_mass))) -
       (3. * x[2] * qdotp * reduced_mass_over_total_mass *
            (340. + 3. * M_PI * M_PI + 112. * reduced_mass_over_total_mass) +
        x[5] * qdotq *
            (600. + reduced_mass_over_total_mass *
                        (1340. - 3. * M_PI * M_PI +
                         552. * reduced_mass_over_total_mass))) /
           (6. * sqrt(qdotq * qdotq * qdotq * qdotq * qdotq)) +
       2. / (sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq)) *
           (pdotp * pdotp * x[2] * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass +
            2. * x[5] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            6. * pdotp * x[2] * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3. * x[5] * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            15. * x[2] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            3. * x[5] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7. +
                 reduced_mass_over_total_mass *
                     (-42. + reduced_mass_over_total_mass *
                                 (53. + 5. * reduced_mass_over_total_mass)))));

  const double dH_dq0_Newt = x[0] / std::sqrt(qdotq * qdotq * qdotq);
  const double dH_dq0_1 =
      (-2. * x[0] * std::sqrt(qdotq) +
       3. * x[0] * qdotp * qdotp * reduced_mass_over_total_mass -
       2. * x[3] * qdotp * qdotq * reduced_mass_over_total_mass +
       pdotp * x[0] * qdotq * (3. + reduced_mass_over_total_mass)) /
      (2. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq));
  const double dH_dq0_2 =
      (-48. * x[0] * qdotp * qdotp * std::sqrt(qdotq) *
           reduced_mass_over_total_mass +
       24. * x[3] * qdotp * std::sqrt(qdotq * qdotq * qdotq) *
           reduced_mass_over_total_mass +
       15. * x[0] * qdotp * qdotp * qdotp * qdotp *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass +
       6. * pdotp * x[0] * qdotp * qdotp * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass -
       12. * x[3] * qdotp * qdotp * qdotp * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass -
       4. * x[3] * pdotp * qdotp * qdotq * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass +
       6. * x[0] * qdotq * (1. + 3. * reduced_mass_over_total_mass) -
       8. * pdotp * x[0] * std::sqrt(qdotq * qdotq * qdotq) *
           (5. + 8. * reduced_mass_over_total_mass) +
       pdotp * pdotp * x[0] * qdotq * qdotq *
           (-5. + reduced_mass_over_total_mass *
                      (20. + 3. * reduced_mass_over_total_mass))) /
      (8. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq));
  const double dH_dq0_3 =
      (3. / 2. * qdotp * qdotq *
           (x[0] * (x[1] * x[4] + x[2] * x[5]) -
            x[3] * (x[1] * x[1] + x[2] * x[2])) *
           reduced_mass_over_total_mass *
           (340. + 3. * M_PI * M_PI + 112. * reduced_mass_over_total_mass) +
       2. * x[0] * std::sqrt(qdotq * qdotq * qdotq) *
           (-12. + (-872. + 63. * M_PI * M_PI) * reduced_mass_over_total_mass) -
       6. * qdotp * reduced_mass_over_total_mass *
           reduced_mass_over_total_mass *
           (pdotp * pdotp * x[0] * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) -
            6. * pdotp * x[0] * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) +
            6. * x[3] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (-1. + reduced_mass_over_total_mass) -
            15. * x[0] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass +
            15. * x[3] * qdotp * qdotp * qdotp * qdotp * qdotq *
                reduced_mass_over_total_mass +
            x[3] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (-2. + 3. * reduced_mass_over_total_mass)) -
       2. * qdotp * std::sqrt(qdotq) * reduced_mass_over_total_mass *
           (3. * pdotp * x[0] * qdotp * qdotq *
                (17. + 30. * reduced_mass_over_total_mass) -
            3. * x[3] * pdotp * qdotq * qdotq *
                (17. + 30. * reduced_mass_over_total_mass) +
            8. * x[0] * qdotp * qdotp * qdotp *
                (5. + 43. * reduced_mass_over_total_mass) -
            8. * x[3] * qdotp * qdotp * qdotq *
                (5. + 43. * reduced_mass_over_total_mass)) -
       2. * x[0] * std::sqrt(qdotq) *
           (3. * pdotp * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17. + 30. * reduced_mass_over_total_mass) +
            4. * qdotp * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5. + 43. * reduced_mass_over_total_mass) +
            3. * pdotp * pdotp * qdotq * qdotq *
                (-27. + reduced_mass_over_total_mass *
                            (136. + 109. * reduced_mass_over_total_mass))) +
       0.75 * x[0] * qdotq *
           (3. * qdotp * qdotp * reduced_mass_over_total_mass *
                (340. + 3. * M_PI * M_PI +
                 112. * reduced_mass_over_total_mass) +
            pdotp * qdotq *
                (600. + reduced_mass_over_total_mass *
                            (1340. - 3. * M_PI * M_PI +
                             552. * reduced_mass_over_total_mass))) -
       3. * x[0] *
           (pdotp * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3. * pdotp * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            5. * qdotp * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            pdotp * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7. +
                 reduced_mass_over_total_mass *
                     (-42. + reduced_mass_over_total_mass *
                                 (53. + 5. * reduced_mass_over_total_mass))))) /
      (48. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq *
                       qdotq * qdotq));

  const double dH_dq1_Newt = x[1] / std::sqrt(qdotq * qdotq * qdotq);
  const double dH_dq1_1 =
      (-2. * x[1] * std::sqrt(qdotq) +
       3. * x[1] * qdotp * qdotp * reduced_mass_over_total_mass -
       2. * x[4] * qdotp * qdotq * reduced_mass_over_total_mass +
       pdotp * x[1] * qdotq * (3. + reduced_mass_over_total_mass)) /
      (2. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq));
  const double dH_dq1_2 =
      (-48. * x[1] * qdotp * qdotp * std::sqrt(qdotq) *
           reduced_mass_over_total_mass +
       24. * x[4] * qdotp * std::sqrt(qdotq * qdotq * qdotq) *
           reduced_mass_over_total_mass +
       15. * x[1] * qdotp * qdotp * qdotp * qdotp *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass +
       6. * pdotp * x[1] * qdotp * qdotp * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass -
       12. * x[4] * qdotp * qdotp * qdotp * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass -
       4. * x[4] * pdotp * qdotp * qdotq * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass +
       6. * x[1] * qdotq * (1. + 3. * reduced_mass_over_total_mass) -
       8. * pdotp * x[1] * std::sqrt(qdotq * qdotq * qdotq) *
           (5. + 8. * reduced_mass_over_total_mass) +
       pdotp * pdotp * x[1] * qdotq * qdotq *
           (-5. + reduced_mass_over_total_mass *
                      (20. + 3. * reduced_mass_over_total_mass))) /
      (8. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq));
  const double dH_dq1_3 =
      (3. / 2. * qdotp * qdotq *
           (x[1] * (x[0] * x[3] + x[2] * x[5]) -
            x[4] * (x[0] * x[0] + x[2] * x[2])) *
           reduced_mass_over_total_mass *
           (340. + 3. * M_PI * M_PI + 112. * reduced_mass_over_total_mass) +
       2. * x[1] * std::sqrt(qdotq * qdotq * qdotq) *
           (-12 + (-872. + 63. * M_PI * M_PI) * reduced_mass_over_total_mass) -
       6. * qdotp * reduced_mass_over_total_mass *
           reduced_mass_over_total_mass *
           (pdotp * pdotp * x[1] * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) -
            6. * pdotp * x[1] * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) +
            6. * x[4] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (-1. + reduced_mass_over_total_mass) -
            15. * x[1] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass +
            15. * x[4] * qdotp * qdotp * qdotp * qdotp * qdotq *
                reduced_mass_over_total_mass +
            x[4] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (-2. + 3. * reduced_mass_over_total_mass)) -
       2. * qdotp * std::sqrt(qdotq) * reduced_mass_over_total_mass *
           (3. * pdotp * x[1] * qdotp * qdotq *
                (17. + 30. * reduced_mass_over_total_mass) -
            3. * x[4] * pdotp * qdotq * qdotq *
                (17. + 30. * reduced_mass_over_total_mass) +
            8. * x[1] * qdotp * qdotp * qdotp *
                (5. + 43. * reduced_mass_over_total_mass) -
            8. * x[4] * qdotp * qdotp * qdotq *
                (5. + 43. * reduced_mass_over_total_mass)) -
       2. * x[1] * std::sqrt(qdotq) *
           (3. * pdotp * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17. + 30. * reduced_mass_over_total_mass) +
            4. * qdotp * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5. + 43. * reduced_mass_over_total_mass) +
            3. * pdotp * pdotp * qdotq * qdotq *
                (-27. + reduced_mass_over_total_mass *
                            (136. + 109. * reduced_mass_over_total_mass))) +
       0.75 * x[1] * qdotq *
           (3. * qdotp * qdotp * reduced_mass_over_total_mass *
                (340. + 3. * M_PI * M_PI +
                 112. * reduced_mass_over_total_mass) +
            pdotp * qdotq *
                (600. + reduced_mass_over_total_mass *
                            (1340. - 3. * M_PI * M_PI +
                             552. * reduced_mass_over_total_mass))) -
       3. * x[1] *
           (pdotp * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3. * pdotp * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            5. * qdotp * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            pdotp * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7. +
                 reduced_mass_over_total_mass *
                     (-42. + reduced_mass_over_total_mass *
                                 (53. + 5. * reduced_mass_over_total_mass))))) /
      (48. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq *
                       qdotq * qdotq));

  const double dH_dq2_Newt = x[2] / std::sqrt(qdotq * qdotq * qdotq);
  const double dH_dq2_1 =
      (-2. * x[2] * std::sqrt(qdotq) +
       3. * x[2] * qdotp * qdotp * reduced_mass_over_total_mass -
       2. * x[5] * qdotp * qdotq * reduced_mass_over_total_mass +
       pdotp * x[2] * qdotq * (3. + reduced_mass_over_total_mass)) /
      (2. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq));
  const double dH_dq2_2 =
      (-48. * x[2] * qdotp * qdotp * std::sqrt(qdotq) *
           reduced_mass_over_total_mass +
       24. * x[5] * qdotp * std::sqrt(qdotq * qdotq * qdotq) *
           reduced_mass_over_total_mass +
       15. * x[2] * qdotp * qdotp * qdotp * qdotp *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass +
       6. * pdotp * x[2] * qdotp * qdotp * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass -
       12. * x[5] * qdotp * qdotp * qdotp * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass -
       4. * x[5] * pdotp * qdotp * qdotq * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass +
       6. * x[2] * qdotq * (1. + 3. * reduced_mass_over_total_mass) -
       8. * pdotp * x[2] * std::sqrt(qdotq * qdotq * qdotq) *
           (5. + 8. * reduced_mass_over_total_mass) +
       pdotp * pdotp * x[2] * qdotq * qdotq *
           (-5. + reduced_mass_over_total_mass *
                      (20. + 3. * reduced_mass_over_total_mass))) /
      (8. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq));
  const double dH_dq2_3 =
      (3. / 2. * qdotp * qdotq *
           (x[2] * (x[0] * x[3] + x[1] * x[3]) -
            x[5] * (x[0] * x[0] + x[1] * x[1])) *
           reduced_mass_over_total_mass *
           (340. + 3. * M_PI * M_PI + 112. * reduced_mass_over_total_mass) +
       2. * x[2] * std::sqrt(qdotq * qdotq * qdotq) *
           (-12. + (-872. + 63. * M_PI * M_PI) * reduced_mass_over_total_mass) -
       6. * qdotp * reduced_mass_over_total_mass *
           reduced_mass_over_total_mass *
           (pdotp * pdotp * x[2] * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) -
            6. * pdotp * x[2] * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) +
            6. * x[5] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (-1. + reduced_mass_over_total_mass) -
            15. * x[2] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass +
            15. * x[5] * qdotp * qdotp * qdotp * qdotp * qdotq *
                reduced_mass_over_total_mass +
            x[5] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (-2. + 3. * reduced_mass_over_total_mass)) -
       2. * qdotp * std::sqrt(qdotq) * reduced_mass_over_total_mass *
           (3. * pdotp * x[2] * qdotp * qdotq *
                (17. + 30. * reduced_mass_over_total_mass) -
            3. * x[5] * pdotp * qdotq * qdotq *
                (17. + 30. * reduced_mass_over_total_mass) +
            8. * x[2] * qdotp * qdotp * qdotp *
                (5. + 43. * reduced_mass_over_total_mass) -
            8. * x[5] * qdotp * qdotp * qdotq *
                (5. + 43. * reduced_mass_over_total_mass)) -
       2. * x[2] * std::sqrt(qdotq) *
           (3. * pdotp * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17. + 30. * reduced_mass_over_total_mass) +
            4. * qdotp * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5. + 43. * reduced_mass_over_total_mass) +
            3. * pdotp * pdotp * qdotq * qdotq *
                (-27. + reduced_mass_over_total_mass *
                            (136. + 109. * reduced_mass_over_total_mass))) +
       0.75 * x[2] * qdotq *
           (3. * qdotp * qdotp * reduced_mass_over_total_mass *
                (340. + 3. * M_PI * M_PI +
                 112. * reduced_mass_over_total_mass) +
            pdotp * qdotq *
                (600. + reduced_mass_over_total_mass *
                            (1340. - 3. * M_PI * M_PI +
                             552. * reduced_mass_over_total_mass))) -
       3. * x[2] *
           (pdotp * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2. - 3. * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3. * pdotp * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1. + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            5. * qdotp * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            pdotp * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7. +
                 reduced_mass_over_total_mass *
                     (-42. + reduced_mass_over_total_mass *
                                 (53. + 5. * reduced_mass_over_total_mass))))) /
      (48. * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq *
                       qdotq * qdotq));

  const double L =
      total_mass * reduced_mass *
      sqrt((x[1] * x[5] - x[2] * x[4]) * (x[1] * x[5] - x[2] * x[4]) +
           (x[2] * x[3] - x[0] * x[5]) * (x[2] * x[3] - x[0] * x[5]) +
           (x[0] * x[4] - x[1] * x[3]) * (x[0] * x[4] - x[1] * x[3]));
  const double w = L / (square(total_mass) * qdotq);
  const double vw = std::cbrt(total_mass * w);
  const double gamma_Euler =
      0.57721566490153286060651209008240243104215933593992;

  const double f2 =
      -(1247. / 336.) - (35. / 12.) * reduced_mass_over_total_mass;
  const double f3 = 4 * M_PI;
  const double f4 =
      -(44711. / 9072.) + (9271. / 504.) * reduced_mass_over_total_mass +
      (65. / 18.) * reduced_mass_over_total_mass * reduced_mass_over_total_mass;
  const double f5 =
      -(8191. / 672. + 583. / 24. * reduced_mass_over_total_mass) * M_PI;
  const double f6 = (6643739519. / 69854400.) + (16. / 3.) * M_PI * M_PI -
                    (1712. / 105.) * gamma_Euler +
                    (-134543. / 7776. + (41. / 48.) * M_PI * M_PI) *
                        reduced_mass_over_total_mass -
                    (94403. / 3024.) * reduced_mass_over_total_mass *
                        reduced_mass_over_total_mass -
                    (775. / 324.) * reduced_mass_over_total_mass *
                        reduced_mass_over_total_mass *
                        reduced_mass_over_total_mass;
  const double fl6 = -1712. / 105.;
  const double f7 =
      (-16285. / 504. + 214745. / 1728. * reduced_mass_over_total_mass +
       193385. / 3024. * reduced_mass_over_total_mass *
           reduced_mass_over_total_mass) *
      M_PI;

  const double dE_dt =
      -(32. / 5.) * reduced_mass_over_total_mass *
      reduced_mass_over_total_mass * vw * vw * vw * vw * vw * vw * vw * vw *
      vw * vw *
      (1. + f2 * vw * vw + f3 * vw * vw * vw + f4 * vw * vw * vw * vw +
       f5 * vw * vw * vw * vw * vw + f6 * vw * vw * vw * vw * vw * vw +
       fl6 * vw * vw * vw * vw * vw * vw * std::log(4. * vw) +
       f7 * vw * vw * vw * vw * vw * vw * vw);

  const std::array<double, 3> F{1. / (w * L) * dE_dt * x[3],
                                1. / (w * L) * dE_dt * x[4],
                                1. / (w * L) * dE_dt * x[5]};

  dpdt[0] = (1. / total_mass) *
            (dH_dp0_Newt + dH_dp0_1 + dH_dp0_2 + dH_dp0_3);  // dX0/dt = dH/dP0
  dpdt[1] = (1. / total_mass) *
            (dH_dp1_Newt + dH_dp1_1 + dH_dp1_2 + dH_dp1_3);  // dX1/dt = dH/dP1
  dpdt[2] = (1. / total_mass) *
            (dH_dp2_Newt + dH_dp2_1 + dH_dp2_2 + dH_dp2_3);  // dX2/dt = dH/dP2

  dpdt[3] =
      -(1. / total_mass) * (dH_dq0_Newt + dH_dq0_1 + dH_dq0_2 + dH_dq0_3) +
      F[0];  // dP0/dt = -dH/dX0 + F0
  dpdt[4] =
      -(1. / total_mass) * (dH_dq1_Newt + dH_dq1_1 + dH_dq1_2 + dH_dq1_3) +
      F[1];  // dP1/dt = -dH/dX1 + F1
  dpdt[5] =
      -(1. / total_mass) * (dH_dq2_Newt + dH_dq2_1 + dH_dq2_2 + dH_dq2_3) +
      F[2];  // dP2/dt = -dH/dX2 + F2
}

void BinaryWithGravitationalWavesHistory::observer_vector(
    const BinaryWithGravitationalWavesHistory::state_type& x, const double t) {
  past_time_.push_back(t);

  std::array<double, 3> x_cm = {
      (xcoords[1] * masses[1] + xcoords[0] * masses[0]) / total_mass, offset[0],
      offset[1]};

  for (size_t i = 0; i < 3; ++i) {
    past_position_left_.at(i).push_back(x_cm.at(i) - masses[1] * x.at(i));
    past_position_right_.at(i).push_back(x_cm.at(i) + masses[0] * x.at(i));
  }
  for (size_t i = 3; i < 6; ++i) {
    past_momentum_left_.at(i - 3).push_back(x.at(i) * reduced_mass);
    past_momentum_right_.at(i - 3).push_back(-x.at(i) * reduced_mass);
  }

  state_type dxdt;
  hamiltonian_system(x, dxdt);

  for (size_t i = 0; i < 3; ++i) {
    past_dt_position_left_.at(i).push_back(-dxdt.at(i) * masses[1]);
    past_dt_position_right_.at(i).push_back(dxdt.at(i) * masses[0]);
  }
  for (size_t i = 3; i < 6; ++i) {
    past_dt_momentum_left_.at(i - 3).push_back(dxdt.at(i) * reduced_mass);
    past_dt_momentum_right_.at(i - 3).push_back(-dxdt.at(i) * reduced_mass);
  }
}

void BinaryWithGravitationalWavesHistory::integrate_hamiltonian_system() {
  BinaryWithGravitationalWavesHistory::state_type ini = {
      initial_state_position.at(0),
      initial_state_position.at(1),
      initial_state_position.at(2),
      initial_state_momentum.at(0),
      initial_state_momentum.at(1),
      initial_state_momentum.at(2)};  // initial conditions

  auto hamiltonian_system_lambda = [this](auto&& PH1, auto&& PH2,
                                          const double /*t*/) {
    hamiltonian_system(std::forward<decltype(PH1)>(PH1),
                       std::forward<decltype(PH2)>(PH2));
  };

  auto observer = [this](auto&& PH1, auto&& PH2) {
    observer_vector(std::forward<decltype(PH1)>(PH1),
                    std::forward<decltype(PH2)>(PH2));
  };

  // Integrate the Hamiltonian system
  boost::numeric::odeint::integrate_const(
      boost::numeric::odeint::runge_kutta4<
          BinaryWithGravitationalWavesHistory::state_type>(),
      hamiltonian_system_lambda, ini, initial_time, final_time, -time_step,
      observer);
}

void BinaryWithGravitationalWavesHistory::write_evolution_to_file() const {
  if (write_evolution_option) {
    std::ofstream file;
    file.open("PastHistoryEvolution.csv");
    file << "time, position_left_x, position_left_y, position_left_z, "
            "momentum_left_x, momentum_left_y, momentum_left_z, "
            "position_right_x, position_right_y, position_right_z, "
            "momentum_right_x, momentum_right_y, momentum_right_z, "
            "dt_momentum_left_x, dt_momentum_left_y, dt_momentum_left_z, "
            "dt_momentum_right_x, dt_momentum_right_y, dt_momentum_right_z, "
         << std::endl;
    for (size_t i = 0; i <= number_of_steps; ++i) {
      file << std::fixed << std::setprecision(4) << past_time_.at(i) << ", ";
      for (size_t j = 0; j < 3; ++j) {
        file << std::fixed << std::setprecision(16)
             << past_position_left_.at(j).at(i) << ", ";
      }
      for (size_t j = 0; j < 3; ++j) {
        file << std::fixed << std::setprecision(16)
             << past_momentum_left_.at(j).at(i) << ", ";
      }
      for (size_t j = 0; j < 3; ++j) {
        file << std::fixed << std::setprecision(16)
             << past_position_right_.at(j).at(i) << ", ";
      }
      for (size_t j = 0; j < 3; ++j) {
        file << std::fixed << std::setprecision(16)
             << past_momentum_right_.at(j).at(i) << ", ";
      }
      for (size_t j = 0; j < 3; ++j) {
        file << std::fixed << std::setprecision(16)
             << past_dt_momentum_left_.at(j).at(i) << ", ";
      }
      for (size_t j = 0; j < 3; ++j) {
        file << std::fixed << std::setprecision(16)
             << past_dt_momentum_right_.at(j).at(i) << ", ";
      }
      file << std::endl;
    }
    file.close();
  }
}

}  // namespace Xcts::AnalyticData::detail

template class Xcts::AnalyticData::CommonVariables<
    DataVector, typename Xcts::AnalyticData::detail::
                    BinaryWithGravitationalWavesVariables<DataVector>::Cache>;
