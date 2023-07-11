// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/WeylMagnetic.hpp"

#include <cstddef>

#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace gr {
template <typename Frame, typename DataType>
tnsr::ii<DataType, 3, Frame> weyl_magnetic(
    const tnsr::ijj<DataType, 3, Frame>& grad_extrinsic_curvature,
    const tnsr::ii<DataType, 3, Frame>& spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric) {
  auto weyl_magnetic_part = make_with_value<tnsr::ii<DataType, 3, Frame>>(
      get<0, 0>(spatial_metric), 0.0);
  weyl_magnetic<Frame, DataType>(make_not_null(&weyl_magnetic_part),
                                 grad_extrinsic_curvature, spatial_metric,
                                 sqrt_det_spatial_metric);

  return weyl_magnetic_part;
}

template <typename Frame, typename DataType>
void weyl_magnetic(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> weyl_magnetic_part,
    const tnsr::ijj<DataType, 3, Frame>& grad_extrinsic_curvature,
    const tnsr::ii<DataType, 3, Frame>& spatial_metric,
    const Scalar<DataType>& sqrt_det_spatial_metric) {
  auto grad_extrinsic_curvature_cross_spatial_metric =
      make_with_value<tnsr::ij<DataType, 3, Frame>>(get<0, 0>(spatial_metric),
                                                    0.0);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (LeviCivitaIterator<3> it; it; ++it) {
        grad_extrinsic_curvature_cross_spatial_metric.get(i, j) +=
            it.sign() * grad_extrinsic_curvature.get(it[2], it[1], i) *
            spatial_metric.get(j, it[0]);
      }
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      weyl_magnetic_part->get(i, j) =
          (grad_extrinsic_curvature_cross_spatial_metric.get(i, j) +
           grad_extrinsic_curvature_cross_spatial_metric.get(j, i)) *
          (0.5 / get(sqrt_det_spatial_metric));
    }
  }
}

template <typename Frame, typename DataType>
void weyl_magnetic_scalar(
    const gsl::not_null<Scalar<DataType>*> weyl_magnetic_scalar_result,
    const tnsr::ii<DataType, 3, Frame>& weyl_magnetic,
    const tnsr::II<DataType, 3, Frame>& inverse_spatial_metric) {
  auto weyl_magnetic_up_down = make_with_value<tnsr::Ij<DataType, 3, Frame>>(
      get<0, 0>(inverse_spatial_metric), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        weyl_magnetic_up_down.get(j, k) +=
            weyl_magnetic.get(i, k) * inverse_spatial_metric.get(i, j);
      }
    }
  }
  for (size_t j = 0; j < 3; ++j) {
    for (size_t k = 0; k < 3; ++k) {
      if (UNLIKELY(j == 0 and k == 0)) {
        get(*weyl_magnetic_scalar_result) =
            weyl_magnetic_up_down.get(j, k) * weyl_magnetic_up_down.get(k, j);
      } else {
        get(*weyl_magnetic_scalar_result) +=
            weyl_magnetic_up_down.get(j, k) * weyl_magnetic_up_down.get(k, j);
      }
    }
  }
}

template <typename Frame, typename DataType>
Scalar<DataType> weyl_magnetic_scalar(
    const tnsr::ii<DataType, 3, Frame>& weyl_magnetic,
    const tnsr::II<DataType, 3, Frame>& inverse_spatial_metric) {
  Scalar<DataType> weyl_magnetic_scalar_result{
      get<0, 0>(inverse_spatial_metric)};
  weyl_magnetic_scalar<Frame, DataType>(
      make_not_null(&weyl_magnetic_scalar_result), weyl_magnetic,
      inverse_spatial_metric);
  return weyl_magnetic_scalar_result;
}
}  // namespace gr

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                  \
  template tnsr::ii<DTYPE(data), 3, FRAME(data)> gr::weyl_magnetic(           \
      const tnsr::ijj<DTYPE(data), 3, FRAME(data)>& grad_extrinsic_curvature, \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& spatial_metric,            \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric);                    \
  template void gr::weyl_magnetic(                                            \
      const gsl::not_null<tnsr::ii<DTYPE(data), 3, FRAME(data)>*>             \
          weyl_magnetic_part,                                                 \
      const tnsr::ijj<DTYPE(data), 3, FRAME(data)>& grad_extrinsic_curvature, \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& spatial_metric,            \
      const Scalar<DTYPE(data)>& sqrt_det_spatial_metric);                    \
  template Scalar<DTYPE(data)> gr::weyl_magnetic_scalar(                      \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& weyl_magnetic,             \
      const tnsr::II<DTYPE(data), 3, FRAME(data)>& inverse_spatial_metric);   \
  template void gr::weyl_magnetic_scalar(                                     \
      const gsl::not_null<Scalar<DTYPE(data)>*> weyl_magnetic_scalar_result,  \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& weyl_magnetic,             \
      const tnsr::II<DTYPE(data), 3, FRAME(data)>& inverse_spatial_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DTYPE
#undef FRAME
#undef INSTANTIATE
