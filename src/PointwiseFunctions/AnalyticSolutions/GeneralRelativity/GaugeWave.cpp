// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"

#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "ErrorHandling/Assert.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace {
// compute gauge wave H
template <typename DataType>
DataType gauge_wave_h(const tnsr::I<DataType, 3>& x, const double t,
                      const double amplitude,
                      const double wavelength) noexcept {
  return {-1.0 * amplitude * sin((get<0>(x) - t) * (2.0 * M_PI / wavelength)) +
          1.0};
}

// compute gauge wave derivH: \partial_x H
template <typename DataType>
DataType gauge_wave_deriv_h(const tnsr::I<DataType, 3>& x, const double t,
                            const double amplitude,
                            const double wavelength) noexcept {
  return {-1.0 * amplitude * (2.0 * M_PI / wavelength) *
          cos((get<0>(x) - t) * (2.0 * M_PI / wavelength))};
}
}  // namespace

namespace gr {
namespace Solutions {
GaugeWave::GaugeWave(const double amplitude, const double wavelength) noexcept
    : amplitude_(amplitude), wavelength_(wavelength) {
  ASSERT(amplitude > 0.0,
         "Amplitude must be non-negative. Given amplitude: " << amplitude_);
  ASSERT(wavelength > 0.0,
         "Wavelength must be non-negative. Given wavelength: " << wavelength_);
}

void GaugeWave::pup(PUP::er& p) noexcept {
  p | amplitude_;
  p | wavelength_;
}

template <typename DataType>
GaugeWave::IntermediateVars<DataType>::IntermediateVars(
    const double amplitude, const double wavelength,
    const tnsr::I<DataType, GaugeWave::volume_dim, Frame::Inertial>& x,
    const double t) noexcept {
  h = gauge_wave_h(x, t, amplitude, wavelength);
  dx_h = gauge_wave_deriv_h(x, t, amplitude, wavelength);
  sqrt_h = sqrt(h);
  dx_h_over_2_sqrt_h = 0.5 * dx_h / sqrt(h);
}

template <typename DataType>
auto GaugeWave::variables(
    const tnsr::I<DataType, GaugeWave::volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const GaugeWave::IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const noexcept
    -> tuples::TaggedTuple<gr::Tags::Lapse<DataType>> {
  return {Scalar<DataType>{vars.sqrt_h}};
}

template <typename DataType>
auto GaugeWave::variables(
    const tnsr::I<DataType, GaugeWave::volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const GaugeWave::IntermediateVars<DataType>& vars,
    tmpl::list<::Tags::dt<gr::Tags::Lapse<DataType>>> /*meta*/) const noexcept
    -> tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DataType>>> {
  return {Scalar<DataType>{-1.0 * vars.dx_h_over_2_sqrt_h}};
}

template <typename DataType>
auto GaugeWave::variables(
    const tnsr::I<DataType, GaugeWave::volume_dim, Frame::Inertial>& x,
    const double /*t*/, const GaugeWave::IntermediateVars<DataType>& vars,
    tmpl::list<DerivLapse<DataType>> /*meta*/) const noexcept
    -> tuples::TaggedTuple<DerivLapse<DataType>> {
  // most parts of d_lapse are zero, so make_with_value() here
  auto d_lapse = make_with_value<
      tnsr::i<DataType, GaugeWave::volume_dim, Frame::Inertial>>(x, 0.0);
  get<0>(d_lapse) = vars.dx_h_over_2_sqrt_h;
  return d_lapse;
}

template <typename DataType>
auto GaugeWave::variables(
    const tnsr::I<DataType, GaugeWave::volume_dim, Frame::Inertial>& x,
    const double /*t*/, const GaugeWave::IntermediateVars<DataType>& /*vars*/,
    tmpl::list<gr::Tags::Shift<GaugeWave::volume_dim, Frame::Inertial,
                               DataType>> /*meta*/) const noexcept
    -> tuples::TaggedTuple<
        gr::Tags::Shift<GaugeWave::volume_dim, Frame::Inertial, DataType>> {
  return {make_with_value<
      tnsr::I<DataType, GaugeWave::volume_dim, Frame::Inertial>>(x, 0.0)};
}

template <typename DataType>
auto GaugeWave::variables(
    const tnsr::I<DataType, GaugeWave::volume_dim, Frame::Inertial>& x,
    const double /*t*/, const GaugeWave::IntermediateVars<DataType>& /*vars*/,
    tmpl::list<::Tags::dt<gr::Tags::Shift<
        GaugeWave::volume_dim, Frame::Inertial, DataType>>> /*meta*/) const
    noexcept -> tuples::TaggedTuple<::Tags::dt<
        gr::Tags::Shift<GaugeWave::volume_dim, Frame::Inertial, DataType>>> {
  return {make_with_value<
      tnsr::I<DataType, GaugeWave::volume_dim, Frame::Inertial>>(x, 0.0)};
}

template <typename DataType>
auto GaugeWave::variables(
    const tnsr::I<DataType, GaugeWave::volume_dim, Frame::Inertial>& x,
    const double /*t*/, const GaugeWave::IntermediateVars<DataType>& /*vars*/,
    tmpl::list<DerivShift<DataType>> /*meta*/) const noexcept
    -> tuples::TaggedTuple<DerivShift<DataType>> {
  return {make_with_value<
      tnsr::iJ<DataType, GaugeWave::volume_dim, Frame::Inertial>>(x, 0.0)};
}

template <typename DataType>
auto GaugeWave::variables(
    const tnsr::I<DataType, GaugeWave::volume_dim, Frame::Inertial>& x,
    const double /*t*/, const GaugeWave::IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::SpatialMetric<GaugeWave::volume_dim, Frame::Inertial,
                                       DataType>> /*meta*/) const noexcept
    -> tuples::TaggedTuple<gr::Tags::SpatialMetric<GaugeWave::volume_dim,
                                                   Frame::Inertial, DataType>> {
  auto spatial_metric = make_with_value<
      tnsr::ii<DataType, GaugeWave::volume_dim, Frame::Inertial>>(x, 0.0);
  get<0, 0>(spatial_metric) = vars.h;
  for (size_t i = 1; i < GaugeWave::volume_dim; ++i) {
    spatial_metric.get(i, i) = 1.0;
  }
  return spatial_metric;
}

template <typename DataType>
auto GaugeWave::variables(
    const tnsr::I<DataType, GaugeWave::volume_dim, Frame::Inertial>& x,
    const double /*t*/, const GaugeWave::IntermediateVars<DataType>& vars,
    tmpl::list<::Tags::dt<gr::Tags::SpatialMetric<
        GaugeWave::volume_dim, Frame::Inertial, DataType>>> /*meta*/) const
    noexcept -> tuples::TaggedTuple<::Tags::dt<gr::Tags::SpatialMetric<
        GaugeWave::volume_dim, Frame::Inertial, DataType>>> {
  auto dt_spatial_metric = make_with_value<
      tnsr::ii<DataType, GaugeWave::volume_dim, Frame::Inertial>>(x, 0.0);
  get<0, 0>(dt_spatial_metric) = -1.0 * vars.dx_h;
  return dt_spatial_metric;
}

template <typename DataType>
auto GaugeWave::variables(
    const tnsr::I<DataType, GaugeWave::volume_dim, Frame::Inertial>& x,
    const double /*t*/, const GaugeWave::IntermediateVars<DataType>& vars,
    tmpl::list<GaugeWave::DerivSpatialMetric<DataType>> /*meta*/) const noexcept
    -> tuples::TaggedTuple<GaugeWave::DerivSpatialMetric<DataType>> {
  auto d_spatial_metric = make_with_value<
      tnsr::ijj<DataType, GaugeWave::volume_dim, Frame::Inertial>>(x, 0.0);
  get<0, 0, 0>(d_spatial_metric) = vars.dx_h;
  return d_spatial_metric;
}

template <typename DataType>
auto GaugeWave::variables(
    const tnsr::I<DataType, GaugeWave::volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const GaugeWave::IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>> /*meta*/) const
    noexcept -> tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataType>> {
  return {Scalar<DataType>{vars.sqrt_h}};
}

template <typename DataType>
auto GaugeWave::variables(
    const tnsr::I<DataType, GaugeWave::volume_dim, Frame::Inertial>& x,
    const double /*t*/, const GaugeWave::IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::ExtrinsicCurvature<
        GaugeWave::volume_dim, Frame::Inertial, DataType>> /*meta*/) const
    noexcept -> tuples::TaggedTuple<gr::Tags::ExtrinsicCurvature<
        GaugeWave::volume_dim, Frame::Inertial, DataType>> {
  auto extrinsic_curvature = make_with_value<
      tnsr::ii<DataType, GaugeWave::volume_dim, Frame::Inertial>>(x, 0.0);
  get<0, 0>(extrinsic_curvature) = vars.dx_h_over_2_sqrt_h;
  return extrinsic_curvature;
}

template <typename DataType>
auto GaugeWave::variables(
    const tnsr::I<DataType, GaugeWave::volume_dim, Frame::Inertial>& x,
    const double /*t*/, const GaugeWave::IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::InverseSpatialMetric<
        GaugeWave::volume_dim, Frame::Inertial, DataType>> /*meta*/) const
    noexcept -> tuples::TaggedTuple<gr::Tags::InverseSpatialMetric<
        GaugeWave::volume_dim, Frame::Inertial, DataType>> {
  auto inverse_spatial_metric = make_with_value<
      tnsr::II<DataType, GaugeWave::volume_dim, Frame::Inertial>>(x, 0.0);
  get<0, 0>(inverse_spatial_metric) = 1.0 / vars.h;
  for (size_t i = 1; i < GaugeWave::volume_dim; ++i) {
    inverse_spatial_metric.get(i, i) = 1.0;
  }
  return inverse_spatial_metric;
}

bool operator==(const GaugeWave& lhs, const GaugeWave& rhs) noexcept {
  return lhs.amplitude() == rhs.amplitude() and
         lhs.wavelength() == rhs.wavelength();
}

bool operator!=(const GaugeWave& lhs, const GaugeWave& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace Solutions
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template gr::Solutions::GaugeWave::IntermediateVars<DTYPE(data)>::           \
      IntermediateVars(                                                        \
          const double amplitude, const double wavelength,                     \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& x,           \
          const double t) noexcept;                                            \
  template tuples::TaggedTuple<gr::Tags::Lapse<DTYPE(data)>>                   \
  gr::Solutions::GaugeWave::variables(                                         \
      const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& /*x*/,           \
      const double /*t*/,                                                      \
      const gr::Solutions::GaugeWave::IntermediateVars<DTYPE(data)>& vars,     \
      tmpl::list<gr::Tags::Lapse<DTYPE(data)>> /*meta*/) const noexcept;       \
  template tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DTYPE(data)>>>       \
  gr::Solutions::GaugeWave::variables(                                         \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const gr::Solutions::GaugeWave::IntermediateVars<DTYPE(data)>& vars,     \
      tmpl::list<::Tags::dt<gr::Tags::Lapse<DTYPE(data)>>> /*meta*/)           \
      const noexcept;                                                          \
  template tuples::TaggedTuple<::Tags::deriv<                                  \
      gr::Tags::Lapse<DTYPE(data)>, tmpl::size_t<DIM(data)>, Frame::Inertial>> \
  gr::Solutions::GaugeWave::variables(                                         \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const gr::Solutions::GaugeWave::IntermediateVars<DTYPE(data)>& vars,     \
      tmpl::list<                                                              \
          ::Tags::deriv<gr::Tags::Lapse<DTYPE(data)>, tmpl::size_t<DIM(data)>, \
                        Frame::Inertial>> /*meta*/) const noexcept;            \
  template tuples::TaggedTuple<                                                \
      gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>>                \
  gr::Solutions::GaugeWave::variables(                                         \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const gr::Solutions::GaugeWave::IntermediateVars<DTYPE(data)>& vars,     \
      tmpl::list<                                                              \
          gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>> /*meta*/)  \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::dt<gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>>>    \
  gr::Solutions::GaugeWave::variables(                                         \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const gr::Solutions::GaugeWave::IntermediateVars<DTYPE(data)>& vars,     \
      tmpl::list<::Tags::dt<                                                   \
          gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>>> /*meta*/) \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>,  \
                    tmpl::size_t<DIM(data)>, Frame::Inertial>>                 \
  gr::Solutions::GaugeWave::variables(                                         \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const gr::Solutions::GaugeWave::IntermediateVars<DTYPE(data)>& vars,     \
      tmpl::list<::Tags::deriv<                                                \
          gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>,            \
          tmpl::size_t<DIM(data)>, Frame::Inertial>> /*meta*/) const noexcept; \
  template tuples::TaggedTuple<                                                \
      gr::Tags::SpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>>        \
  gr::Solutions::GaugeWave::variables(                                         \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const gr::Solutions::GaugeWave::IntermediateVars<DTYPE(data)>& vars,     \
      tmpl::list<gr::Tags::SpatialMetric<DIM(data), Frame::Inertial,           \
                                         DTYPE(data)>> /*meta*/)               \
      const noexcept;                                                          \
  template tuples::TaggedTuple<::Tags::dt<                                     \
      gr::Tags::SpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>>>       \
  gr::Solutions::GaugeWave::variables(                                         \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const gr::Solutions::GaugeWave::IntermediateVars<DTYPE(data)>& vars,     \
      tmpl::list<::Tags::dt<gr::Tags::SpatialMetric<                           \
          DIM(data), Frame::Inertial, DTYPE(data)>>> /*meta*/) const noexcept; \
  template tuples::TaggedTuple<::Tags::deriv<                                  \
      gr::Tags::SpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>,        \
      tmpl::size_t<DIM(data)>, Frame::Inertial>>                               \
  gr::Solutions::GaugeWave::variables(                                         \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const gr::Solutions::GaugeWave::IntermediateVars<DTYPE(data)>& vars,     \
      tmpl::list<::Tags::deriv<                                                \
          gr::Tags::SpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>,    \
          tmpl::size_t<DIM(data)>, Frame::Inertial>> /*meta*/) const noexcept; \
  template tuples::TaggedTuple<                                                \
      gr::Tags::InverseSpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>> \
  gr::Solutions::GaugeWave::variables(                                         \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const gr::Solutions::GaugeWave::IntermediateVars<DTYPE(data)>& vars,     \
      tmpl::list<gr::Tags::InverseSpatialMetric<DIM(data), Frame::Inertial,    \
                                                DTYPE(data)>> /*meta*/)        \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      gr::Tags::ExtrinsicCurvature<DIM(data), Frame::Inertial, DTYPE(data)>>   \
  gr::Solutions::GaugeWave::variables(                                         \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const gr::Solutions::GaugeWave::IntermediateVars<DTYPE(data)>& vars,     \
      tmpl::list<gr::Tags::ExtrinsicCurvature<DIM(data), Frame::Inertial,      \
                                              DTYPE(data)>> /*meta*/)          \
      const noexcept;                                                          \
  template tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>>    \
  gr::Solutions::GaugeWave::variables(                                         \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const gr::Solutions::GaugeWave::IntermediateVars<DTYPE(data)>& vars,     \
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>> /*meta*/)        \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (3), (double, DataVector))

#undef DIM
#undef DTYPE
#undef INSTANTIATE
/// \endcond
