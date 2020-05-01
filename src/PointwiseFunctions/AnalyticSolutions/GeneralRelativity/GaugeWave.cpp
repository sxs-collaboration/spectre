// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"

#include <cmath>
#include <cstddef>
#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace {
// compute gauge wave H
template <typename DataType, size_t Dim>
DataType gauge_wave_h(const tnsr::I<DataType, Dim>& x, const double t,
                      const double amplitude,
                      const double wavelength) noexcept {
  return {-1.0 * amplitude * sin((get<0>(x) - t) * (2.0 * M_PI / wavelength)) +
          1.0};
}

// compute gauge wave derivH: \partial_x H
template <typename DataType, size_t Dim>
DataType gauge_wave_deriv_h(const tnsr::I<DataType, Dim>& x, const double t,
                            const double amplitude,
                            const double wavelength) noexcept {
  return {-1.0 * amplitude * (2.0 * M_PI / wavelength) *
          cos((get<0>(x) - t) * (2.0 * M_PI / wavelength))};
}
}  // namespace

namespace gr {
namespace Solutions {
template <size_t Dim>
GaugeWave<Dim>::GaugeWave(const double amplitude, const double wavelength,
                          const OptionContext& context)
    : amplitude_(amplitude), wavelength_(wavelength) {
  if (abs(amplitude) >= 1.0) {
    PARSE_ERROR(context,
                "Amplitude must be less than one. Given amplitude: "
                << amplitude_);
  }
  if (wavelength <= 0.0) {
    PARSE_ERROR(context,
                "Wavelength must be non-negative. Given wavelength: "
                << wavelength_);
  }
}

template <size_t Dim>
void GaugeWave<Dim>::pup(PUP::er& p) noexcept {
  p | amplitude_;
  p | wavelength_;
}

template <size_t Dim>
template <typename DataType>
GaugeWave<Dim>::IntermediateVars<DataType>::IntermediateVars(
    const double amplitude, const double wavelength,
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x,
    const double t) noexcept {
  h = gauge_wave_h(x, t, amplitude, wavelength);
  dx_h = gauge_wave_deriv_h(x, t, amplitude, wavelength);
  sqrt_h = sqrt(h);
  dx_h_over_2_sqrt_h = 0.5 * dx_h / sqrt(h);
}

template <size_t Dim>
template <typename DataType>
auto GaugeWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const noexcept
    -> tuples::TaggedTuple<gr::Tags::Lapse<DataType>> {
  return {Scalar<DataType>{vars.sqrt_h}};
}

template <size_t Dim>
template <typename DataType>
auto GaugeWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<::Tags::dt<gr::Tags::Lapse<DataType>>> /*meta*/) const noexcept
    -> tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DataType>>> {
  return {Scalar<DataType>{-1.0 * vars.dx_h_over_2_sqrt_h}};
}

template <size_t Dim>
template <typename DataType>
auto GaugeWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<DerivLapse<DataType>> /*meta*/) const noexcept
    -> tuples::TaggedTuple<DerivLapse<DataType>> {
  // most parts of d_lapse are zero, so make_with_value() here
  auto d_lapse =
      make_with_value<tnsr::i<DataType, volume_dim, Frame::Inertial>>(x, 0.0);
  get<0>(d_lapse) = vars.dx_h_over_2_sqrt_h;
  return d_lapse;
}

template <size_t Dim>
template <typename DataType>
auto GaugeWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& /*vars*/,
    tmpl::list<gr::Tags::Shift<volume_dim, Frame::Inertial, DataType>> /*meta*/)
    const noexcept -> tuples::TaggedTuple<
        gr::Tags::Shift<volume_dim, Frame::Inertial, DataType>> {
  return {
      make_with_value<tnsr::I<DataType, volume_dim, Frame::Inertial>>(x, 0.0)};
}

template <size_t Dim>
template <typename DataType>
auto GaugeWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& /*vars*/,
    tmpl::list<::Tags::dt<
        gr::Tags::Shift<volume_dim, Frame::Inertial, DataType>>> /*meta*/) const
    noexcept -> tuples::TaggedTuple<
        ::Tags::dt<gr::Tags::Shift<volume_dim, Frame::Inertial, DataType>>> {
  return {
      make_with_value<tnsr::I<DataType, volume_dim, Frame::Inertial>>(x, 0.0)};
}

template <size_t Dim>
template <typename DataType>
auto GaugeWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& /*vars*/,
    tmpl::list<DerivShift<DataType>> /*meta*/) const noexcept
    -> tuples::TaggedTuple<DerivShift<DataType>> {
  return {
      make_with_value<tnsr::iJ<DataType, volume_dim, Frame::Inertial>>(x, 0.0)};
}

template <size_t Dim>
template <typename DataType>
auto GaugeWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::SpatialMetric<volume_dim, Frame::Inertial,
                                       DataType>> /*meta*/) const noexcept
    -> tuples::TaggedTuple<
        gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataType>> {
  auto spatial_metric =
      make_with_value<tnsr::ii<DataType, volume_dim, Frame::Inertial>>(x, 0.0);
  get<0, 0>(spatial_metric) = vars.h;
  for (size_t i = 1; i < volume_dim; ++i) {
    spatial_metric.get(i, i) = 1.0;
  }
  return spatial_metric;
}

template <size_t Dim>
template <typename DataType>
auto GaugeWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<::Tags::dt<gr::Tags::SpatialMetric<volume_dim, Frame::Inertial,
                                                  DataType>>> /*meta*/) const
    noexcept -> tuples::TaggedTuple<::Tags::dt<
        gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataType>>> {
  auto dt_spatial_metric =
      make_with_value<tnsr::ii<DataType, volume_dim, Frame::Inertial>>(x, 0.0);
  get<0, 0>(dt_spatial_metric) = -1.0 * vars.dx_h;
  return dt_spatial_metric;
}

template <size_t Dim>
template <typename DataType>
auto GaugeWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<DerivSpatialMetric<DataType>> /*meta*/) const noexcept
    -> tuples::TaggedTuple<DerivSpatialMetric<DataType>> {
  auto d_spatial_metric =
      make_with_value<tnsr::ijj<DataType, volume_dim, Frame::Inertial>>(x, 0.0);
  get<0, 0, 0>(d_spatial_metric) = vars.dx_h;
  return d_spatial_metric;
}

template <size_t Dim>
template <typename DataType>
auto GaugeWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>> /*meta*/) const
    noexcept -> tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataType>> {
  return {Scalar<DataType>{vars.sqrt_h}};
}

template <size_t Dim>
template <typename DataType>
auto GaugeWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::ExtrinsicCurvature<volume_dim, Frame::Inertial,
                                            DataType>> /*meta*/) const noexcept
    -> tuples::TaggedTuple<
        gr::Tags::ExtrinsicCurvature<volume_dim, Frame::Inertial, DataType>> {
  auto extrinsic_curvature =
      make_with_value<tnsr::ii<DataType, volume_dim, Frame::Inertial>>(x, 0.0);
  get<0, 0>(extrinsic_curvature) = vars.dx_h_over_2_sqrt_h;
  return extrinsic_curvature;
}

template <size_t Dim>
template <typename DataType>
auto GaugeWave<Dim>::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::InverseSpatialMetric<volume_dim, Frame::Inertial,
                                              DataType>> /*meta*/) const
    noexcept -> tuples::TaggedTuple<
        gr::Tags::InverseSpatialMetric<volume_dim, Frame::Inertial, DataType>> {
  auto inverse_spatial_metric =
      make_with_value<tnsr::II<DataType, volume_dim, Frame::Inertial>>(x, 0.0);
  get<0, 0>(inverse_spatial_metric) = 1.0 / vars.h;
  for (size_t i = 1; i < volume_dim; ++i) {
    inverse_spatial_metric.get(i, i) = 1.0;
  }
  return inverse_spatial_metric;
}

template <size_t Dim>
bool operator==(const GaugeWave<Dim>& lhs, const GaugeWave<Dim>& rhs) noexcept {
  return lhs.amplitude() == rhs.amplitude() and
         lhs.wavelength() == rhs.wavelength();
}

template <size_t Dim>
bool operator!=(const GaugeWave<Dim>& lhs, const GaugeWave<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template GaugeWave<DIM(data)>::IntermediateVars<DTYPE(data)>::               \
      IntermediateVars(                                                        \
          const double amplitude, const double wavelength,                     \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& x,           \
          const double t) noexcept;                                            \
  template tuples::TaggedTuple<gr::Tags::Lapse<DTYPE(data)>>                   \
  GaugeWave<DIM(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& /*x*/,           \
      const double /*t*/,                                                      \
      const GaugeWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<gr::Tags::Lapse<DTYPE(data)>> /*meta*/) const noexcept;       \
  template tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DTYPE(data)>>>       \
  GaugeWave<DIM(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const GaugeWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<::Tags::dt<gr::Tags::Lapse<DTYPE(data)>>> /*meta*/)           \
      const noexcept;                                                          \
  template tuples::TaggedTuple<::Tags::deriv<                                  \
      gr::Tags::Lapse<DTYPE(data)>, tmpl::size_t<DIM(data)>, Frame::Inertial>> \
  GaugeWave<DIM(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const GaugeWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<                                                              \
          ::Tags::deriv<gr::Tags::Lapse<DTYPE(data)>, tmpl::size_t<DIM(data)>, \
                        Frame::Inertial>> /*meta*/) const noexcept;            \
  template tuples::TaggedTuple<                                                \
      gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>>                \
  GaugeWave<DIM(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const GaugeWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<                                                              \
          gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>> /*meta*/)  \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::dt<gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>>>    \
  GaugeWave<DIM(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const GaugeWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<::Tags::dt<                                                   \
          gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>>> /*meta*/) \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>,  \
                    tmpl::size_t<DIM(data)>, Frame::Inertial>>                 \
  GaugeWave<DIM(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const GaugeWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<::Tags::deriv<                                                \
          gr::Tags::Shift<DIM(data), Frame::Inertial, DTYPE(data)>,            \
          tmpl::size_t<DIM(data)>, Frame::Inertial>> /*meta*/) const noexcept; \
  template tuples::TaggedTuple<                                                \
      gr::Tags::SpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>>        \
  GaugeWave<DIM(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const GaugeWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<gr::Tags::SpatialMetric<DIM(data), Frame::Inertial,           \
                                         DTYPE(data)>> /*meta*/)               \
      const noexcept;                                                          \
  template tuples::TaggedTuple<::Tags::dt<                                     \
      gr::Tags::SpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>>>       \
  GaugeWave<DIM(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const GaugeWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<::Tags::dt<gr::Tags::SpatialMetric<                           \
          DIM(data), Frame::Inertial, DTYPE(data)>>> /*meta*/) const noexcept; \
  template tuples::TaggedTuple<::Tags::deriv<                                  \
      gr::Tags::SpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>,        \
      tmpl::size_t<DIM(data)>, Frame::Inertial>>                               \
  GaugeWave<DIM(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const GaugeWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<::Tags::deriv<                                                \
          gr::Tags::SpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>,    \
          tmpl::size_t<DIM(data)>, Frame::Inertial>> /*meta*/) const noexcept; \
  template tuples::TaggedTuple<                                                \
      gr::Tags::InverseSpatialMetric<DIM(data), Frame::Inertial, DTYPE(data)>> \
  GaugeWave<DIM(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const GaugeWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<gr::Tags::InverseSpatialMetric<DIM(data), Frame::Inertial,    \
                                                DTYPE(data)>> /*meta*/)        \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      gr::Tags::ExtrinsicCurvature<DIM(data), Frame::Inertial, DTYPE(data)>>   \
  GaugeWave<DIM(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const GaugeWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<gr::Tags::ExtrinsicCurvature<DIM(data), Frame::Inertial,      \
                                              DTYPE(data)>> /*meta*/)          \
      const noexcept;                                                          \
  template tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>>    \
  GaugeWave<DIM(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const GaugeWave<DIM(data)>::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>> /*meta*/)        \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector))

#undef INSTANTIATE
#define INSTANTIATE(_, data)                                          \
  template class GaugeWave<DIM(data)>;                                \
  template bool operator==(const GaugeWave<DIM(data)>& lhs,           \
                           const GaugeWave<DIM(data)>& rhs) noexcept; \
  template bool operator!=(const GaugeWave<DIM(data)>& lhs,           \
                           const GaugeWave<DIM(data)>& rhs) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DTYPE
#undef DIM
}  // namespace Solutions
}  // namespace gr
/// \endcond
