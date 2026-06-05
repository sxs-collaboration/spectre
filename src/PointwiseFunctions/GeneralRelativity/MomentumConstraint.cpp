// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/MomentumConstraint.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace gr {
template <typename DataType, size_t SpatialDim, typename Frame>
void momentum_constraint_in_vacuum(
    const gsl::not_null<tnsr::i<DataType, SpatialDim, Frame>*>
        momentum_constraint,
    const tnsr::ijj<DataType, SpatialDim, Frame>& d_extrinsic_curvature,
    const tnsr::i<DataType, SpatialDim, Frame>& d_trace_extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric) {
  set_number_of_grid_points(momentum_constraint, d_trace_extrinsic_curvature);
  tenex::evaluate<ti::i>(momentum_constraint,
                         inverse_spatial_metric(ti::J, ti::K) *
                                 d_extrinsic_curvature(ti::j, ti::k, ti::i) -
                             d_trace_extrinsic_curvature(ti::i));
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::i<DataType, SpatialDim, Frame> momentum_constraint_in_vacuum(
    const tnsr::ijj<DataType, SpatialDim, Frame>& d_extrinsic_curvature,
    const tnsr::i<DataType, SpatialDim, Frame>& d_trace_extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric) {
  tnsr::i<DataType, SpatialDim, Frame> result{};
  momentum_constraint_in_vacuum(make_not_null(&result), d_extrinsic_curvature,
                                d_trace_extrinsic_curvature,
                                inverse_spatial_metric);
  return result;
}
}  // namespace gr

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                       \
  template void gr::momentum_constraint_in_vacuum(                 \
      gsl::not_null<tnsr::i<DTYPE(data), DIM(data), FRAME(data)>*> \
          momentum_constraint,                                     \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&        \
          d_extrinsic_curvature,                                   \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>&          \
          d_trace_extrinsic_curvature,                             \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&         \
          inverse_spatial_metric);                                 \
  template tnsr::i<DTYPE(data), DIM(data), FRAME(data)>            \
  gr::momentum_constraint_in_vacuum(                               \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&        \
          d_extrinsic_curvature,                                   \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>&          \
          d_trace_extrinsic_curvature,                             \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&         \
          inverse_spatial_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector), (1, 2, 3),
                        (Frame::Grid, Frame::Inertial))

#undef DTYPE
#undef DIM
#undef FRAME
#undef INSTANTIATE
