// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedWaveHelpers.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace GeneralizedHarmonic::gauges::DampedHarmonicGauge_detail {
template <size_t SpatialDim, typename Frame, typename DataType>
void spatial_weight_function(const gsl::not_null<Scalar<DataType>*> weight,
                             const tnsr::I<DataType, SpatialDim, Frame>& coords,
                             const double sigma_r) noexcept {
  if (UNLIKELY(get_size(get(*weight)) != get_size(get<0>(coords)))) {
    *weight = Scalar<DataType>(get_size(get<0>(coords)));
  }
  const auto r_squared = dot_product(coords, coords);
  get(*weight) = exp(-get(r_squared) / pow<2>(sigma_r));
}

template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_deriv_of_spatial_weight_function(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> d4_weight,
    const tnsr::I<DataType, SpatialDim, Frame>& coords, const double sigma_r,
    const Scalar<DataType>& weight_function) noexcept {
  if (UNLIKELY(get_size(get<0>(*d4_weight)) != get_size(get<0>(coords)))) {
    *d4_weight = tnsr::a<DataType, SpatialDim, Frame>(get_size(get<0>(coords)));
  }
  // use 0th component to avoid allocations
  get<0>(*d4_weight) = get(weight_function) * (-2. / pow<2>(sigma_r));
  for (size_t i = 0; i < SpatialDim; ++i) {
    d4_weight->get(1 + i) = get<0>(*d4_weight) * coords.get(i);
  }
  // time derivative of weight function is zero
  get<0>(*d4_weight) = 0.;
}

template <typename DataType>
void log_factor_metric_lapse(const gsl::not_null<Scalar<DataType>*> logfac,
                             const Scalar<DataType>& lapse,
                             const Scalar<DataType>& sqrt_det_spatial_metric,
                             const double exponent) noexcept {
  if (UNLIKELY(get_size(get(*logfac)) != get_size(get(lapse)))) {
    *logfac = Scalar<DataType>(get_size(get(lapse)));
  }
  // branching below is to avoid using pow for performance reasons
  if (exponent == 0.) {
    get(*logfac) = -log(get(lapse));
  } else if (exponent == 0.5) {
    get(*logfac) = log(get(sqrt_det_spatial_metric) / get(lapse));
  } else {
    get(*logfac) =
        2. * exponent * log(get(sqrt_det_spatial_metric)) - log(get(lapse));
  }
}

template <typename DataType>
Scalar<DataType> log_factor_metric_lapse(
    const Scalar<DataType>& lapse,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const double exponent) noexcept {
  Scalar<DataType> logfac{};
  log_factor_metric_lapse(make_not_null(&logfac), lapse,
                          sqrt_det_spatial_metric, exponent);
  return logfac;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_deriv_of_log_factor_metric_lapse(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> d4_logfac,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const double exponent) noexcept {
  if (UNLIKELY(get_size(get<0>(*d4_logfac)) != get_size(get(lapse)))) {
    *d4_logfac = tnsr::a<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }
  // Use a TempBuffer to reduce total number of allocations. This is especially
  // important in a multithreaded environment.
  TempBuffer<tmpl::list<::Tags::Tempa<0, SpatialDim, Frame, DataType>,
                        ::Tags::Tempi<1, SpatialDim, Frame, DataType>,
                        ::Tags::Tempa<2, SpatialDim, Frame, DataType>,
                        ::Tags::TempScalar<3, DataType>,
                        ::Tags::TempScalar<4, DataType>>>
      buffer(get_size(get(lapse)));
  auto& d_g = get<::Tags::Tempa<0, SpatialDim, Frame, DataType>>(buffer);
  auto& d3_lapse = get<::Tags::Tempi<1, SpatialDim, Frame, DataType>>(buffer);
  auto& d_lapse = get<::Tags::Tempa<2, SpatialDim, Frame, DataType>>(buffer);
  auto& dt_lapse = get<::Tags::TempScalar<3, DataType>>(buffer);
  auto& one_over_lapse = get<::Tags::TempScalar<4, DataType>>(buffer);

  // Get \f$ \partial_a g\f$
  spacetime_deriv_of_det_spatial_metric<SpatialDim, Frame, DataType>(
      make_not_null(&d_g), sqrt_det_spatial_metric, inverse_spatial_metric,
      dt_spatial_metric, phi);
  // Get \f$ \partial_a N\f$
  time_deriv_of_lapse<SpatialDim, Frame, DataType>(
      make_not_null(&dt_lapse), lapse, shift, spacetime_unit_normal, phi, pi);
  spatial_deriv_of_lapse<SpatialDim, Frame, DataType>(
      make_not_null(&d3_lapse), lapse, spacetime_unit_normal, phi);
  get<0>(d_lapse) = get(dt_lapse);
  for (size_t i = 0; i < SpatialDim; ++i) {
    d_lapse.get(1 + i) = d3_lapse.get(i);
  }
  // Compute
  get(one_over_lapse) = 1. / get(lapse);
  if (exponent == 0.) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      d4_logfac->get(a) = -get(one_over_lapse) * d_lapse.get(a);
    }
  } else {
    const auto p_over_g = exponent / square(get(sqrt_det_spatial_metric));
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      d4_logfac->get(a) =
          p_over_g * d_g.get(a) - get(one_over_lapse) * d_lapse.get(a);
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> spacetime_deriv_of_log_factor_metric_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const double exponent) noexcept {
  tnsr::a<DataType, SpatialDim, Frame> d4_logfac{};
  spacetime_deriv_of_log_factor_metric_lapse(
      make_not_null(&d4_logfac), lapse, shift, spacetime_unit_normal,
      inverse_spatial_metric, sqrt_det_spatial_metric, dt_spatial_metric, pi,
      phi, exponent);
  return d4_logfac;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_deriv_of_power_log_factor_metric_lapse(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> d4_powlogfac,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi, const double g_exponent,
    const int exponent) noexcept {
  if (UNLIKELY(get_size(get<0>(*d4_powlogfac)) != get_size(get(lapse)))) {
    *d4_powlogfac = tnsr::a<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }
  // Use a TempBuffer to reduce total number of allocations. This is especially
  // important in a multithreaded environment.
  TempBuffer<tmpl::list<::Tags::Tempa<0, SpatialDim, Frame, DataType>,
                        ::Tags::TempScalar<1, DataType>,
                        ::Tags::TempScalar<2, DataType>>>
      buffer(get_size(get(lapse)));
  auto& dlogfac = get<::Tags::Tempa<0, SpatialDim, Frame, DataType>>(buffer);
  auto& logfac = get<::Tags::TempScalar<1, DataType>>(buffer);
  auto& prefac = get<::Tags::TempScalar<2, DataType>>(buffer);

  // Compute derivative
  spacetime_deriv_of_log_factor_metric_lapse<SpatialDim, Frame, DataType>(
      make_not_null(&dlogfac), lapse, shift, spacetime_unit_normal,
      inverse_spatial_metric, sqrt_det_spatial_metric, dt_spatial_metric, pi,
      phi, g_exponent);
  // Apply pre-factor
  if (UNLIKELY(exponent == 0)) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      d4_powlogfac->get(a) = 0.;
    }
  } else if (UNLIKELY(exponent == 1)) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      d4_powlogfac->get(a) = dlogfac.get(a);
    }
  } else {
    log_factor_metric_lapse<DataType>(make_not_null(&logfac), lapse,
                                      sqrt_det_spatial_metric, g_exponent);
    get(prefac) =
        static_cast<double>(exponent) * pow(get(logfac), exponent - 1);
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      d4_powlogfac->get(a) = get(prefac) * dlogfac.get(a);
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame>
spacetime_deriv_of_power_log_factor_metric_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi, const double g_exponent,
    const int exponent) noexcept {
  tnsr::a<DataType, SpatialDim, Frame> d4_powlogfac{};
  spacetime_deriv_of_power_log_factor_metric_lapse(
      make_not_null(&d4_powlogfac), lapse, shift, spacetime_unit_normal,
      inverse_spatial_metric, sqrt_det_spatial_metric, dt_spatial_metric, pi,
      phi, g_exponent, exponent);
  return d4_powlogfac;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define DTYPE_SCAL(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template void spatial_weight_function(                                      \
      const gsl::not_null<Scalar<DTYPE(data)>*> weight,                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& coords,             \
      const double sigma_r) noexcept;                                         \
  template void spacetime_deriv_of_spatial_weight_function(                   \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>      \
          d4_weight,                                                          \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& coords,             \
      const double sigma_r,                                                   \
      const Scalar<DTYPE(data)>& weight_function) noexcept;                   \
  template void spacetime_deriv_of_log_factor_metric_lapse(                   \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>      \
          d4_logfac,                                                          \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_unit_normal,                                              \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                     \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const double exponent) noexcept;                                        \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                       \
  spacetime_deriv_of_log_factor_metric_lapse(                                 \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_unit_normal,                                              \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                     \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const double exponent) noexcept;                                        \
  template void spacetime_deriv_of_power_log_factor_metric_lapse(             \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>      \
          d4_powlogfac,                                                       \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_unit_normal,                                              \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                     \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const double g_exponent, const int exponent) noexcept;                  \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                       \
  spacetime_deriv_of_power_log_factor_metric_lapse(                           \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_unit_normal,                                              \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric,                                             \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                     \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,              \
      const double g_exponent, const int exponent) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Inertial))

#undef INSTANTIATE

#define INSTANTIATE(_, data)                                   \
  template void log_factor_metric_lapse(                       \
      const gsl::not_null<Scalar<DTYPE_SCAL(data)>*> logfac,   \
      const Scalar<DTYPE_SCAL(data)>& lapse,                   \
      const Scalar<DTYPE_SCAL(data)>& sqrt_det_spatial_metric, \
      const double exponent) noexcept;                         \
  template Scalar<DTYPE_SCAL(data)> log_factor_metric_lapse(   \
      const Scalar<DTYPE_SCAL(data)>& lapse,                   \
      const Scalar<DTYPE_SCAL(data)>& sqrt_det_spatial_metric, \
      const double exponent) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef INSTANTIATE

#undef DTYPE_SCAL
#undef FRAME
#undef DTYPE
#undef DIM
}  // namespace GeneralizedHarmonic::gauges::DampedHarmonicGauge_detail
/// \endcond
