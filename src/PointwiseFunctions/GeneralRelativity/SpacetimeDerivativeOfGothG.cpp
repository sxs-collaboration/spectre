// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/SpacetimeDerivativeOfGothG.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {
template <typename DataType, size_t SpatialDim, typename Frame>
void spacetime_deriv_of_goth_g(
    gsl::not_null<tnsr::aBB<DataType, SpatialDim, Frame>*> da_goth_g,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::abb<DataType, SpatialDim, Frame>& da_spacetime_metric,
    const Scalar<DataType>& lapse,
    const tnsr::a<DataType, SpatialDim, Frame>& da_lapse,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& da_det_spatial_metric) {
  set_number_of_grid_points(da_goth_g, da_spacetime_metric);

  tenex::evaluate<ti::a, ti::B, ti::C>(da_goth_g,
    (da_lapse(ti::a) * sqrt_det_spatial_metric()
        + 0.5 * lapse() * da_det_spatial_metric(ti::a)
        / sqrt_det_spatial_metric()) * inverse_spacetime_metric(ti::B, ti::C)
     - lapse() * sqrt_det_spatial_metric()
                                    * inverse_spacetime_metric(ti::B, ti::D)
                                    * inverse_spacetime_metric(ti::C, ti::E)
                                    * da_spacetime_metric(ti::a, ti::d, ti::e));
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::aBB<DataType, SpatialDim, Frame> spacetime_deriv_of_goth_g(
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::abb<DataType, SpatialDim, Frame>& da_spacetime_metric,
    const Scalar<DataType>& lapse,
    const tnsr::a<DataType, SpatialDim, Frame>& da_lapse,
    const Scalar<DataType>& sqrt_det_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& da_det_spatial_metric) {
  tnsr::aBB<DataType, SpatialDim, Frame> da_goth_g{};
  gr::spacetime_deriv_of_goth_g(make_not_null(&da_goth_g),
          inverse_spacetime_metric, da_spacetime_metric, lapse, da_lapse,
                   sqrt_det_spatial_metric, da_det_spatial_metric);
  return da_goth_g;
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                              \
  template void gr::spacetime_deriv_of_goth_g(                            \
      const gsl::not_null<tnsr::aBB<DTYPE(data), DIM(data), FRAME(data)>*>\
          da_goth_g,                                                      \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                \
          inverse_spacetime_metric,                                       \
      const tnsr::abb<DTYPE(data), DIM(data), FRAME(data)>&               \
          da_spacetime_metric,                                            \
      const Scalar<DTYPE(data)>& lapse,                                   \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& da_lapse,       \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                 \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                 \
          da_det_spatial_metric);                                         \
  template tnsr::aBB<DTYPE(data), DIM(data), FRAME(data)>                 \
  gr::spacetime_deriv_of_goth_g(                                          \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                \
          inverse_spacetime_metric,                                       \
      const tnsr::abb<DTYPE(data), DIM(data), FRAME(data)>&               \
          da_spacetime_metric,                                            \
      const Scalar<DTYPE(data)>& lapse,                                   \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& da_lapse,       \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric,                 \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                 \
          da_det_spatial_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
