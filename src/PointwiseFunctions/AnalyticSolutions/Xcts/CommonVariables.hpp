// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts::Solutions {

/// Tags for variables that solutions can share
template <typename DataType>
using common_tags = tmpl::push_back<
    AnalyticData::common_tags<DataType>,
    ::Tags::Flux<Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                 Frame::Inertial>,
    ::Tags::Flux<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                 Frame::Inertial>,
    Tags::LongitudinalShiftExcess<DataType, 3, Frame::Inertial>,
    gr::Tags::Shift<3, Frame::Inertial, DataType>,
    Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataType>,
    Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<DataType>,
    Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataType>>;

/// Implementations for variables that solutions can share
template <typename DataType, typename Cache>
struct CommonVariables : AnalyticData::CommonVariables<DataType, Cache> {
  static constexpr size_t Dim = 3;
  using AnalyticData::CommonVariables<DataType, Cache>::operator();
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> conformal_factor_flux,
      gsl::not_null<Cache*> cache,
      ::Tags::Flux<Tags::ConformalFactor<DataType>, tmpl::size_t<Dim>,
                   Frame::Inertial> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> lapse_times_conformal_factor_flux,
      gsl::not_null<Cache*> cache,
      ::Tags::Flux<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<Dim>,
                   Frame::Inertial> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::II<DataType, Dim>*> longitudinal_shift_excess,
      gsl::not_null<Cache*> cache,
      Tags::LongitudinalShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/)
      const;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift, gsl::not_null<Cache*> cache,
      gr::Tags::Shift<Dim, Frame::Inertial, DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*>
          longitudinal_shift_minus_dt_conformal_metric_square,
      gsl::not_null<Cache*> cache,
      Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataType> /*meta*/)
      const;
  void operator()(
      gsl::not_null<Scalar<DataType>*>
          longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      gsl::not_null<Cache*> cache,
      Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
          DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*>
          shift_dot_deriv_extrinsic_curvature_trace,
      gsl::not_null<Cache*> cache,
      Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataType> /*meta*/) const;
};

}  // namespace Xcts::Solutions
