// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {

template <typename DataType, size_t SpatialDim, typename Frame>
void extrinsic_curvature(
    const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*> ex_curvature,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>& deriv_spatial_metric) {
  const DataType half_over_lapse = 0.5 / get(lapse);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {  // Symmetry
      ex_curvature->get(i, j) = -dt_spatial_metric.get(i, j);
      for (size_t k = 0; k < SpatialDim; ++k) {
        ex_curvature->get(i, j) +=
            shift.get(k) * deriv_spatial_metric.get(k, i, j) +
            spatial_metric.get(k, i) * deriv_shift.get(j, k) +
            spatial_metric.get(k, j) * deriv_shift.get(i, k);
      }
      ex_curvature->get(i, j) *= half_over_lapse;
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::ii<DataType, SpatialDim, Frame> extrinsic_curvature(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>& deriv_spatial_metric) {
  tnsr::ii<DataType, SpatialDim, Frame> ex_curvature{};
  extrinsic_curvature(make_not_null(&ex_curvature), lapse, shift, deriv_shift,
                      spatial_metric, dt_spatial_metric, deriv_spatial_metric);
  return ex_curvature;
}

template <typename DataType, size_t SpatialDim, typename Frame>
void covariant_derivative_of_extrinsic_curvature(
    const gsl::not_null<tnsr::ijj<DataType, SpatialDim, Frame>*> grad_ex_curv,
    const tnsr::ijj<DataType, SpatialDim, Frame>& d_ex_curv,
    const tnsr::ii<DataType, SpatialDim, Frame>& ex_curv,
    const tnsr::Ijj<DataType, SpatialDim, Frame>&
        spatial_christoffel_second_kind) {
  set_number_of_grid_points(grad_ex_curv, ex_curv);
  tenex::evaluate<ti::i, ti::j, ti::k>(
      grad_ex_curv, d_ex_curv(ti::i, ti::j, ti::k) -
                        spatial_christoffel_second_kind(ti::L, ti::i, ti::j) *
                            ex_curv(ti::l, ti::k) -
                        spatial_christoffel_second_kind(ti::L, ti::i, ti::k) *
                            ex_curv(ti::j, ti::l));
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::ijj<DataType, SpatialDim, Frame>
covariant_derivative_of_extrinsic_curvature(
    const tnsr::ijj<DataType, SpatialDim, Frame>& d_ex_curv,
    const tnsr::ii<DataType, SpatialDim, Frame>& ex_curv,
    const tnsr::Ijj<DataType, SpatialDim, Frame>&
        spatial_christoffel_second_kind) {
  tnsr::ijj<DataType, SpatialDim, Frame> grad_ex_curv{};
  covariant_derivative_of_extrinsic_curvature(make_not_null(&grad_ex_curv),
                                              d_ex_curv, ex_curv,
                                              spatial_christoffel_second_kind);
  return grad_ex_curv;
}

}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                  \
  template void gr::extrinsic_curvature(                                      \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*>     \
          ex_curvature,                                                       \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,       \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                   \
          deriv_spatial_metric);                                              \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                      \
  gr::extrinsic_curvature(                                                    \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,       \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                   \
          deriv_spatial_metric);                                              \
  template void gr::covariant_derivative_of_extrinsic_curvature(              \
      const gsl::not_null<tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>*>    \
          grad_ex_curv,                                                       \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& d_ex_curv,        \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& ex_curv,           \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&                   \
          spatial_christoffel_second_kind);                                   \
  template tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>                     \
  gr::covariant_derivative_of_extrinsic_curvature(                            \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>& d_ex_curv,        \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& ex_curv,           \
      const tnsr::Ijj<DTYPE(data), DIM(data), FRAME(data)>&                   \
          spatial_christoffel_second_kind);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
