// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/BinaryWithGravitationalWaves.hpp"

#include <brigand/brigand.hpp>

#include <array>
#include <cstddef>

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
  // Third order
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
      // Third order
      dt_conformal_metric.get(i, j) =
          (11. * conformal_metric.get(i, j) -
           18. * conformal_metric_back.get(i, j) +
           9. * conformal_metric_back_two.get(i, j) -
           2. * conformal_metric_back_three.get(i, j)) /
          (6. * time_displacement);
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          longitudinal_shift_background_minus_dt_conformal_metric->get(i, j) -=
              inv_conformal_metric.get(i, k) * inv_conformal_metric.get(j, l) *
              (dt_conformal_metric.get(k, l) -
               (1. / 3.) *
                   get(trace(dt_conformal_metric, inv_conformal_metric)) *
                   conformal_metric.get(k, l));
        }
      }
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
      // Third order
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
tnsr::ii<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_radiative_term(
    DataType t) const {
  const auto distance_left_t = get_t_distance_left(t);
  const auto distance_right_t = get_t_distance_right(t);
  const auto near_zone_term_t = get_t_near_zone_term(t);
  const auto present_term_t = get_t_present_term(t);
  const auto past_term_t = get_t_past_term(t);
  const auto integral_term_t = get_t_integral_term(t);
  double turn_off = .5;
  if (attenuation_parameter == 0) {
    turn_off = 1.;
  }
  tnsr::ii<DataType, 3> radiative_term_t{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      radiative_term_t.get(i, j) =
          (turn_off + .5 * tanh(attenuation_parameter *
                                (get(distance_left_t) - attenuation_radius))) *
          (turn_off + .5 * tanh(attenuation_parameter *
                                (get(distance_right_t) - attenuation_radius))) *
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
  double turn_off = .5;
  if (attenuation_parameter == 0) {
    turn_off = 1.;
  }
  get(lapse_t) *=
      (turn_off + .5 * tanh(attenuation_parameter *
                            (get(distance_left_t) - attenuation_radius))) *
      (turn_off + .5 * tanh(attenuation_parameter *
                            (get(distance_right_t) - attenuation_radius)));

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
  get(lapse_t) +=
      (1. -
       (turn_off + .5 * tanh(attenuation_parameter *
                             (get(distance_left_t) - attenuation_radius))) *
           (turn_off +
            .5 * tanh(attenuation_parameter *
                      (get(distance_right_t) - attenuation_radius)))) *
      get(lapse_left) * get(lapse_right);
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

  double turn_off = .5;
  if (attenuation_parameter == 0) {
    turn_off = 1.;
  }
  for (size_t i = 0; i < 3; ++i) {
    shift_t.get(i) *=
        (turn_off + .5 * tanh(attenuation_parameter *
                              (get(distance_left_t) - attenuation_radius))) *
        (turn_off + .5 * tanh(attenuation_parameter *
                              (get(distance_right_t) - attenuation_radius)));
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
        (1. -
         (turn_off + .5 * tanh(attenuation_parameter *
                               (get(distance_left_t) - attenuation_radius))) *
             (turn_off +
              .5 * tanh(attenuation_parameter *
                        (get(distance_right_t) - attenuation_radius)))) *
        (shift_left.get(i) + shift_right.get(i));
  }
  return shift_t;
}

template <typename DataType>
Scalar<DataType>
BinaryWithGravitationalWavesVariables<DataType>::get_t_distance_left(
    const DataType time) const {
  // later it will depend on t by interpolation of past history
  tnsr::I<DataType, 3> v = x;
  const std::array<double, 3> position_left = {xcoords[0], offset[0],
                                               offset[1]};
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = x.get(j)[i] - gsl::at(position_left, j);
    }
  }
  return magnitude(v);
}

template <typename DataType>
Scalar<DataType>
BinaryWithGravitationalWavesVariables<DataType>::get_t_distance_right(
    const DataType time) const {
  // later it will depend on t by interpolation of past history
  tnsr::I<DataType, 3> v = x;
  const std::array<double, 3> position_right = {xcoords[1], offset[0],
                                                offset[1]};
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = x.get(j)[i] - gsl::at(position_right, j);
    }
  }
  return magnitude(v);
}

template <typename DataType>
Scalar<DataType>
BinaryWithGravitationalWavesVariables<DataType>::get_t_separation(
    const DataType time) const {
  // later it will depend on t by interpolation of past history
  tnsr::I<DataType, 3> v = x;
  const std::array<double, 3> separation = {xcoords[1] - xcoords[0], 0., 0.};
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = gsl::at(separation, j);
    }
  }
  return magnitude(v);
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_momentum_left(
    const DataType time) const {
  // later it will depend on t by interpolation of past history
  tnsr::I<DataType, 3> v = x;
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = gsl::at(momentum_left, j);
    }
  }
  return v;
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_momentum_right(
    const DataType time) const {
  // later it will depend on t by interpolation of past history
  tnsr::I<DataType, 3> v = x;
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = gsl::at(momentum_right, j);
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
  const std::array<double, 3> position_left = {xcoords[0], offset[0],
                                               offset[1]};
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] =
          (x.get(j)[i] - gsl::at(position_left, j)) / get(distance_left)[i];
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
  const std::array<double, 3> position_right = {xcoords[1], offset[0],
                                                offset[1]};
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] =
          (x.get(j)[i] - gsl::at(position_right, j)) / get(distance_right)[i];
    }
  }

  return v;
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_normal_lr(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  std::array<double, 3> separation = {xcoords[1] - xcoords[0], 0., 0.};
  Scalar<DataType> separation_norm = get_t_separation(time);
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = gsl::at(separation, j) / get(separation_norm)[i];
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

template class BinaryWithGravitationalWavesVariables<DataVector>;

}  // namespace Xcts::AnalyticData::detail

template class Xcts::AnalyticData::CommonVariables<
    DataVector, typename Xcts::AnalyticData::detail::
                    BinaryWithGravitationalWavesVariables<DataVector>::Cache>;
