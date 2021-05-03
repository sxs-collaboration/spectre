// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/WeylMagnetic.hpp"

#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

///\cond
namespace gr {
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> weyl_magnetic(
    const tnsr::ijj<DataType, SpatialDim, Frame>& grad_extrinsic_curvature,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric) noexcept {
  tnsr::ii<DataType, SpatialDim, Frame> weyl_magnetic_part{};
  weyl_magnetic<SpatialDim, Frame, DataType>(make_not_null(&weyl_magnetic_part),
                                             grad_extrinsic_curvature,
                                             spatial_metric);

  return weyl_magnetic_part;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void weyl_magnetic(
    const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>
        weyl_magnetic_part,
    const tnsr::ijj<DataType, SpatialDim, Frame>& grad_extrinsic_curvature,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric) noexcept {
  auto grad_extrinsic_curvature_cross_spatial_metric =
      make_with_value<tnsr::ij<DataType, SpatialDim, Frame>>(
          get<0, 0>(spatial_metric), 0.0);

  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (LeviCivitaIterator<3> it; it; ++it) {
        grad_extrinsic_curvature_cross_spatial_metric.get(i, j) +=
            it.sign() * grad_extrinsic_curvature.get(it[2], it[1], i) *
            spatial_metric.get(j, it[0]);
      }
    }
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      weyl_magnetic_part->get(i, j) =
          (grad_extrinsic_curvature_cross_spatial_metric.get(i, j) +
           grad_extrinsic_curvature_cross_spatial_metric.get(j, i)) *
          (0.5 / sqrt(get(determinant_and_inverse(spatial_metric).first)));
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void weyl_magnetic_scalar(
    const gsl::not_null<Scalar<DataType>*> weyl_magnetic_scalar_result,
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_magnetic,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept {
  *weyl_magnetic_scalar_result =
      make_with_value<Scalar<DataType>>(get<0, 0>(inverse_spatial_metric), 0.0);

  auto weyl_magnetic_up_down =
      make_with_value<tnsr::Ij<DataType, SpatialDim, Frame>>(
          get<0, 0>(inverse_spatial_metric), 0.0);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        weyl_magnetic_up_down.get(j, k) +=
            weyl_magnetic.get(i, k) * inverse_spatial_metric.get(i, j);
      }
    }
  }
  for (size_t j = 0; j < SpatialDim; ++j) {
    for (size_t k = 0; k < SpatialDim; ++k) {
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

template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> weyl_magnetic_scalar(
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_magnetic,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept {
  Scalar<DataType> weyl_magnetic_scalar_result{};
  weyl_magnetic_scalar<SpatialDim, Frame, DataType>(
      make_not_null(&weyl_magnetic_scalar_result), weyl_magnetic,
      inverse_spatial_metric);
  return weyl_magnetic_scalar_result;
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                 \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)> gr::weyl_magnetic(  \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                  \
          grad_extrinsic_curvature,                                          \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                   \
          spatial_metric) noexcept;                                          \
  template void gr::weyl_magnetic(                                           \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*>    \
          weyl_magnetic_part,                                                \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                  \
          grad_extrinsic_curvature,                                          \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                   \
          spatial_metric) noexcept;                                          \
  template Scalar<DTYPE(data)> gr::weyl_magnetic_scalar(                     \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& weyl_magnetic,    \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spatial_metric) noexcept;                                  \
  template void gr::weyl_magnetic_scalar(                                    \
      const gsl::not_null<Scalar<DTYPE(data)>*> weyl_magnetic_scalar_result, \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& weyl_magnetic,    \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
// endcond
