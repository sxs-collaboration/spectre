// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedWaveHelpers.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivOfDetSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfLapse.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::gauges::DampedHarmonicGauge_detail {
template <typename DataType, size_t SpatialDim, typename Frame>
void spatial_weight_function(const gsl::not_null<Scalar<DataType>*> weight,
                             const tnsr::I<DataType, SpatialDim, Frame>& coords,
                             const double sigma_r) {
  const auto r_squared = dot_product(coords, coords);
  get(*weight) = exp(-get(r_squared) / pow<2>(sigma_r));
}

template <typename DataType, size_t SpatialDim, typename Frame>
void spacetime_deriv_of_spatial_weight_function(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> d4_weight,
    const tnsr::I<DataType, SpatialDim, Frame>& coords, const double sigma_r,
    const Scalar<DataType>& weight_function) {
  set_number_of_grid_points(d4_weight, coords);
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
                             const double exponent) {
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
    const Scalar<DataType>& sqrt_det_spatial_metric, const double exponent) {
  Scalar<DataType> logfac{get_size(get(lapse))};
  log_factor_metric_lapse(make_not_null(&logfac), lapse,
                          sqrt_det_spatial_metric, exponent);
  return logfac;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define DTYPE_SCAL(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                             \
  template void spatial_weight_function(                                 \
      const gsl::not_null<Scalar<DTYPE(data)>*> weight,                  \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& coords,        \
      const double sigma_r);                                             \
  template void spacetime_deriv_of_spatial_weight_function(              \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*> \
          d4_weight,                                                     \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& coords,        \
      const double sigma_r, const Scalar<DTYPE(data)>& weight_function);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Inertial))

#undef INSTANTIATE

#define INSTANTIATE(_, data)                                   \
  template void log_factor_metric_lapse(                       \
      const gsl::not_null<Scalar<DTYPE_SCAL(data)>*> logfac,   \
      const Scalar<DTYPE_SCAL(data)>& lapse,                   \
      const Scalar<DTYPE_SCAL(data)>& sqrt_det_spatial_metric, \
      const double exponent);                                  \
  template Scalar<DTYPE_SCAL(data)> log_factor_metric_lapse(   \
      const Scalar<DTYPE_SCAL(data)>& lapse,                   \
      const Scalar<DTYPE_SCAL(data)>& sqrt_det_spatial_metric, \
      const double exponent);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef INSTANTIATE

#undef DTYPE_SCAL
#undef FRAME
#undef DTYPE
#undef DIM
}  // namespace gh::gauges::DampedHarmonicGauge_detail
