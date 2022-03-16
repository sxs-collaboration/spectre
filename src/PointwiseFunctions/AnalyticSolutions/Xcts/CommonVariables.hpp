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
    // Solved variables
    Tags::ConformalFactor<DataType>, Tags::LapseTimesConformalFactor<DataType>,
    Tags::ShiftExcess<DataType, 3, Frame::Inertial>,
    // ADM variables
    gr::Tags::Lapse<DataType>, gr::Tags::Shift<3, Frame::Inertial, DataType>,
    gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>,
    gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataType>,
    ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>,
                  tmpl::size_t<3>, Frame::Inertial>,
    gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataType>,
    // Derivatives of solved variables
    ::Tags::deriv<Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                  Frame::Inertial>,
    ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                  Frame::Inertial>,
    Tags::ShiftStrain<DataType, 3, Frame::Inertial>,
    // Fluxes
    ::Tags::Flux<Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                 Frame::Inertial>,
    ::Tags::Flux<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                 Frame::Inertial>,
    Tags::LongitudinalShiftExcess<DataType, 3, Frame::Inertial>,
    // Background quantities for subsets of the XCTS equations
    Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataType>,
    Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<DataType>,
    Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataType>>;

/// Implementations for variables that solutions can share
template <typename DataType, typename Cache>
struct CommonVariables : AnalyticData::CommonVariables<DataType, Cache> {
  static constexpr size_t Dim = 3;
  using Base = AnalyticData::CommonVariables<DataType, Cache>;
  using Base::Base;
  using Base::operator();

  virtual void operator()(gsl::not_null<Scalar<DataType>*> conformal_factor,
                          gsl::not_null<Cache*> cache,
                          Tags::ConformalFactor<DataType> /*meta*/) const = 0;
  virtual void operator()(
      gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor,
      gsl::not_null<Cache*> cache,
      Tags::LapseTimesConformalFactor<DataType> /*meta*/) const = 0;
  virtual void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
      gsl::not_null<Cache*> cache,
      Tags::ShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/) const = 0;
  virtual void operator()(gsl::not_null<Scalar<DataType>*> lapse,
                          gsl::not_null<Cache*> cache,
                          gr::Tags::Lapse<DataType> /*meta*/) const = 0;
  virtual void operator()(
      gsl::not_null<tnsr::ii<DataType, Dim>*> spatial_metric,
      gsl::not_null<Cache*> cache,
      gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType> /*meta*/) const;
  virtual void operator()(
      gsl::not_null<tnsr::II<DataType, Dim>*> inv_spatial_metric,
      gsl::not_null<Cache*> cache,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataType> /*meta*/)
      const;
  virtual void operator()(
      gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_spatial_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const;
  virtual void operator()(
      gsl::not_null<tnsr::i<DataType, Dim>*> deriv_conformal_factor,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::ConformalFactor<DataType>, tmpl::size_t<Dim>,
                    Frame::Inertial> /*meta*/) const = 0;
  virtual void operator()(
      gsl::not_null<tnsr::i<DataType, Dim>*> deriv_lapse_times_conformal_factor,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const = 0;
  virtual void operator()(
      gsl::not_null<tnsr::ii<DataType, Dim>*> shift_strain,
      gsl::not_null<Cache*> cache,
      Tags::ShiftStrain<DataType, Dim, Frame::Inertial> /*meta*/) const = 0;
  virtual void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> conformal_factor_flux,
      gsl::not_null<Cache*> cache,
      ::Tags::Flux<Tags::ConformalFactor<DataType>, tmpl::size_t<Dim>,
                   Frame::Inertial> /*meta*/) const;
  virtual void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> lapse_times_conformal_factor_flux,
      gsl::not_null<Cache*> cache,
      ::Tags::Flux<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<Dim>,
                   Frame::Inertial> /*meta*/) const;
  virtual void operator()(
      gsl::not_null<tnsr::II<DataType, Dim>*> longitudinal_shift_excess,
      gsl::not_null<Cache*> cache,
      Tags::LongitudinalShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/)
      const;
  virtual void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift, gsl::not_null<Cache*> cache,
      gr::Tags::Shift<Dim, Frame::Inertial, DataType> /*meta*/) const;
  virtual void operator()(
      gsl::not_null<tnsr::ii<DataType, Dim>*> extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      gr::Tags::ExtrinsicCurvature<Dim, Frame::Inertial, DataType> /*meta*/)
      const = 0;
  virtual void operator()(
      gsl::not_null<Scalar<DataType>*>
          longitudinal_shift_minus_dt_conformal_metric_square,
      gsl::not_null<Cache*> cache,
      Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataType> /*meta*/)
      const;
  virtual void operator()(
      gsl::not_null<Scalar<DataType>*>
          longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      gsl::not_null<Cache*> cache,
      Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
          DataType> /*meta*/) const;
  virtual void operator()(
      gsl::not_null<Scalar<DataType>*>
          shift_dot_deriv_extrinsic_curvature_trace,
      gsl::not_null<Cache*> cache,
      Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataType> /*meta*/) const;
};

}  // namespace Xcts::Solutions
