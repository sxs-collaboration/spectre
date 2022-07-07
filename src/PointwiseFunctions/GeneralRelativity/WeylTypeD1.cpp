// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/WeylTypeD1.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

//

namespace gr {

template <size_t SpatialDim, typename Frame, typename DataType>
void weyl_type_D1_tensor(
    const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>
        weyl_type_D1_tensor,
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_electric,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric) {
  // compute a
  Scalar<DataType> a =
      weyl_electric_scalar(weyl_electric, inverse_spatial_metric);
  get(a) *= 16.0;

  tnsr::Ij<DataType, SpatialDim, Frame> weyl_electric_up_down =
      make_with_value<tnsr::Ij<DataType, SpatialDim, Frame>>(
          get<0, 0>(inverse_spatial_metric), 0.0);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        weyl_electric_up_down.get(j, k) +=
            weyl_electric.get(i, k) * inverse_spatial_metric.get(i, j);
      }
    }
  }

  auto upper_weyl_electric =
      make_with_value<tnsr::II<DataType, SpatialDim, Frame>>(
          get<0, 0>(inverse_spatial_metric), 0.0);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t l = i; l < SpatialDim; ++l) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        upper_weyl_electric.get(i, l) +=
            weyl_electric_up_down.get(i, j) * inverse_spatial_metric.get(j, l);
      }
    }
  }

  auto b =
      make_with_value<Scalar<DataType>>(get<0, 0>(inverse_spatial_metric), 0.0);

  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        get(b) += weyl_electric_up_down.get(k, i) *
                  upper_weyl_electric.get(i, j) * weyl_electric.get(j, k);
      }
    }
  }
  get(b) *= -64.0;

  // compute type D1:
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      weyl_type_D1_tensor->get(i, j) =
          (get(a) / 12.0) * spatial_metric.get(i, j) -
          (get(b) / get(a) * weyl_electric.get(i, j));
      for (size_t k = 0; k < SpatialDim; ++k) {
        weyl_type_D1_tensor->get(i, j) -=
            4.0 * (weyl_electric_up_down.get(k, i) * weyl_electric.get(j, k));
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> weyl_type_D1_tensor(
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_electric,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric) {
  tnsr::ii<DataType, SpatialDim, Frame> weyl_type_D_tensor_result{};
  weyl_type_D1_tensor<SpatialDim, Frame, DataType>(
      make_not_null(&weyl_type_D_tensor_result), weyl_electric, spatial_metric,
      inverse_spatial_metric);
  return weyl_type_D_tensor_result;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void weyl_type_D1_scalar(
    const gsl::not_null<Scalar<DataType>*> weyl_type_D1_tensor_scalar_result,
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_type_D1_tensor,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric) {
  *weyl_type_D1_tensor_scalar_result =
      make_with_value<Scalar<DataType>>(get<0, 0>(inverse_spatial_metric), 0.0);

  auto weyl_type_D1_tensor_up_down =
      make_with_value<tnsr::Ij<DataType, SpatialDim, Frame>>(
          get<0, 0>(inverse_spatial_metric), 0.0);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        weyl_type_D1_tensor_up_down.get(j, k) +=
            weyl_type_D1_tensor.get(i, k) * inverse_spatial_metric.get(i, j);
      }
    }
  }
  for (size_t j = 0; j < SpatialDim; ++j) {
    for (size_t k = 0; k < SpatialDim; ++k) {
      if (UNLIKELY(j == 0 and k == 0)) {
        get(*weyl_type_D1_tensor_scalar_result) =
            weyl_type_D1_tensor_up_down.get(j, k) *
            weyl_type_D1_tensor_up_down.get(k, j);
      } else {
        get(*weyl_type_D1_tensor_scalar_result) +=
            weyl_type_D1_tensor_up_down.get(j, k) *
            weyl_type_D1_tensor_up_down.get(k, j);
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> weyl_type_D1_scalar(
    const tnsr::ii<DataType, SpatialDim, Frame>& weyl_type_D1_tensor,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric) {
  Scalar<DataType> weyl_type_D1_tensor_scalar_result{};
  weyl_type_D1_scalar<SpatialDim, Frame, DataType>(
      make_not_null(&weyl_type_D1_tensor_scalar_result), weyl_type_D1_tensor,
      inverse_spatial_metric);
  return weyl_type_D1_tensor_scalar_result;
}

}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                               \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                   \
  gr::weyl_type_D1_tensor(                                                 \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& weyl_electric,  \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric, \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spatial_metric);                                         \
  template void gr::weyl_type_D1_tensor(                                   \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*>  \
          weyl_type_D1_tensor,                                             \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& weyl_electric,  \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric, \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spatial_metric);                                         \
  template void gr::weyl_type_D1_scalar(                                   \
      const gsl::not_null<Scalar<DTYPE(data)>*>                            \
          weyl_type_D1_tensor_scalar_result,                               \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                 \
          weyl_type_D1_tensor,                                             \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spatial_metric);                                         \
  template Scalar<DTYPE(data)> gr::weyl_type_D1_scalar(                    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                 \
          weyl_type_D1_tensor,                                             \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                 \
          inverse_spatial_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))
#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
