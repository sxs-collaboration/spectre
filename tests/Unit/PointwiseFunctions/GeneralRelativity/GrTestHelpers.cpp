// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/PointwiseFunctions/GeneralRelativity/GrTestHelpers.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <typename DataType>
Scalar<DataType> make_lapse(const DataType& used_for_size) {
  return Scalar<DataType>{make_with_value<DataType>(used_for_size, 3.)};
}

template <size_t SpatialDim, typename DataType>
tnsr::I<DataType, SpatialDim> make_shift(const DataType& used_for_size) {
  tnsr::I<DataType, SpatialDim> shift{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    shift.get(i) = make_with_value<DataType>(used_for_size, i + 1);
  }
  return shift;
}

template <size_t SpatialDim, typename DataType>
tnsr::ii<DataType, SpatialDim> make_spatial_metric(
    const DataType& used_for_size) {
  tnsr::ii<DataType, SpatialDim> metric{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      metric.get(i, j) =
          make_with_value<DataType>(used_for_size, (i + 1.) * (j + 1.));
    }
  }
  return metric;
}

template <size_t SpatialDim, typename DataType>
tnsr::II<DataType, SpatialDim> make_inverse_spatial_metric(
    const DataType& used_for_size) {
  tnsr::II<DataType, SpatialDim> metric{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      metric.get(i, j) =
          make_with_value<DataType>(used_for_size, (i + 1.) * (j + 1.));
    }
  }
  return metric;
}

template <typename DataType>
Scalar<DataType> make_dt_lapse(const DataType& used_for_size) {
  return Scalar<DataType>{make_with_value<DataType>(used_for_size, 5.)};
}

template <size_t SpatialDim, typename DataType>
tnsr::I<DataType, SpatialDim> make_dt_shift(const DataType& used_for_size) {
  tnsr::I<DataType, SpatialDim> dt_shift{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    dt_shift.get(i) =
        make_with_value<DataType>(used_for_size, SpatialDim * (i + 1.));
  }
  return dt_shift;
}

template <size_t SpatialDim, typename DataType>
tnsr::ii<DataType, SpatialDim> make_dt_spatial_metric(
    const DataType& used_for_size) {
  tnsr::ii<DataType, SpatialDim> metric{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      metric.get(i, j) = make_with_value<DataType>(used_for_size, i + j);
    }
  }
  return metric;
}

template <size_t SpatialDim, typename DataType>
tnsr::i<DataType, SpatialDim> make_deriv_lapse(const DataType& used_for_size) {
  tnsr::i<DataType, SpatialDim> deriv_lapse{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    deriv_lapse.get(i) =
        make_with_value<DataType>(used_for_size, 2.5 * (i + 1.));
  }
  return deriv_lapse;
}

template <size_t SpatialDim, typename DataType>
tnsr::iJ<DataType, SpatialDim> make_deriv_shift(const DataType& used_for_size) {
  tnsr::iJ<DataType, SpatialDim> deriv_shift{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      deriv_shift.get(i, j) =
          make_with_value<DataType>(used_for_size, 3. * (j + 1.) - i + 4.);
    }
  }
  return deriv_shift;
}

template <size_t SpatialDim, typename DataType>
tnsr::ijj<DataType, SpatialDim> make_deriv_spatial_metric(
    const DataType& used_for_size) {
  tnsr::ijj<DataType, SpatialDim> deriv_spatial_metric{};
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      for (size_t k = 0; k < SpatialDim; ++k) {
        deriv_spatial_metric.get(k, i, j) = make_with_value<DataType>(
            used_for_size, 3. * (i + 1.) * (j + 1.) + k);
      }
    }
  }
  return deriv_spatial_metric;
}

template <size_t SpatialDim, typename DataType>
tnsr::abb<DataType, SpatialDim> make_spacetime_deriv_spacetime_metric(
    const DataType& used_for_size) {
  tnsr::abb<DataType, SpatialDim> d_spacetime_metric{};
  for (size_t i = 0; i < SpatialDim + 1; i++) {
    for (size_t j = i; j < SpatialDim + 1; j++) {
      for (size_t k = 0; k < SpatialDim + 1; k++) {
        d_spacetime_metric.get(k, i, j) = make_with_value<DataType>(
            used_for_size, (i + 1.) * (j + 1.) * (k + 3.));
      }
    }
  }
  return d_spacetime_metric;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                 \
  template tnsr::I<DTYPE(data), DIM(data)> make_shift(const DTYPE(data) &    \
                                                      used_for_size);        \
  template tnsr::ii<DTYPE(data), DIM(data)> make_spatial_metric(             \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::II<DTYPE(data), DIM(data)> make_inverse_spatial_metric(     \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::I<DTYPE(data), DIM(data)> make_dt_shift(const DTYPE(data) & \
                                                         used_for_size);     \
  template tnsr::ii<DTYPE(data), DIM(data)> make_dt_spatial_metric(          \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::i<DTYPE(data), DIM(data)> make_deriv_lapse(                 \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::iJ<DTYPE(data), DIM(data)> make_deriv_shift(                \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::ijj<DTYPE(data), DIM(data)> make_deriv_spatial_metric(      \
      const DTYPE(data) & used_for_size);                                    \
  template tnsr::abb<DTYPE(data), DIM(data)>                                 \
  make_spacetime_deriv_spacetime_metric(const DTYPE(data) & used_for_size);

template Scalar<double> make_lapse(const double& used_for_size);
template Scalar<DataVector> make_lapse(const DataVector& used_for_size);
template Scalar<double> make_dt_lapse(const double& used_for_size);
template Scalar<DataVector> make_dt_lapse(const DataVector& used_for_size);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector))

#undef DIM
#undef DTYPE
#undef INSTANTIATE
