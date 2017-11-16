// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/MakeWithValue.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <size_t Dim, typename Frame, typename DataType>
tnsr::aa<DataType, Dim, Frame> compute_spacetime_metric(
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric) noexcept {
  tnsr::aa<DataType, Dim, Frame> spacetime_metric{
      make_with_value<DataType>(lapse.get(), 0.)};

  get<0, 0>(spacetime_metric) = -lapse.get() * lapse.get();

  for (size_t m = 0; m < Dim; ++m) {
    get<0, 0>(spacetime_metric) +=
        spatial_metric.get(m, m) * shift.get(m) * shift.get(m);
    for (size_t n = 0; n < m; ++n) {
      get<0, 0>(spacetime_metric) +=
          2 * spatial_metric.get(m, n) * shift.get(m) * shift.get(n);
    }
  }

  for (size_t i = 0; i < Dim; ++i) {
    for (size_t m = 0; m < Dim; ++m) {
      spacetime_metric.get(0, i + 1) += spatial_metric.get(m, i) * shift.get(m);
    }
    for (size_t j = i; j < Dim; ++j) {
      spacetime_metric.get(i + 1, j + 1) = spatial_metric.get(i, j);
    }
  }
  return spacetime_metric;
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::AA<DataType, Dim, Frame> compute_inverse_spacetime_metric(
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric) noexcept {
  tnsr::AA<DataType, Dim, Frame> inverse_spacetime_metric{};

  get<0, 0>(inverse_spacetime_metric) =
      -1.0 / (lapse.get() * lapse.get());

  const auto& minus_one_over_lapse_sqrd = get<0, 0>(inverse_spacetime_metric);

  for (size_t i = 0; i < Dim; ++i) {
    inverse_spacetime_metric.get(0, i + 1) = -
        shift.get(i) * minus_one_over_lapse_sqrd;
  }

  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      inverse_spacetime_metric.get(i + 1, j + 1) =
          inverse_spatial_metric.get(i, j) +
          shift.get(i) * shift.get(j) * minus_one_over_lapse_sqrd;
    }
  }
  return inverse_spacetime_metric;
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::abb<DataType, Dim, Frame> compute_derivatives_of_spacetime_metric(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, Dim, Frame>& deriv_lapse,
    const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::I<DataType, Dim, Frame>& dt_shift,
    const tnsr::iJ<DataType, Dim, Frame>& deriv_shift,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric,
    const tnsr::ii<DataType, Dim, Frame>& dt_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& deriv_spatial_metric) noexcept {
  tnsr::abb<DataType, Dim, Frame> spacetime_deriv_spacetime_metric{
      make_with_value<DataType>(lapse.get(), 0.)};

  get<0, 0, 0>(spacetime_deriv_spacetime_metric) =
      -2.0 * lapse.get() * dt_lapse.get();

  for (size_t m = 0; m < Dim; ++m) {
    for (size_t n = 0; n < Dim; ++n) {
      get<0, 0, 0>(spacetime_deriv_spacetime_metric) +=
          dt_spatial_metric.get(m, n) * shift.get(m) * shift.get(n) +
          2.0 * spatial_metric.get(m, n) * shift.get(m) * dt_shift.get(n);
    }
  }

  for (size_t i = 0; i < Dim; ++i) {
    for (size_t m = 0; m < Dim; ++m) {
      spacetime_deriv_spacetime_metric.get(0, 0, i + 1) +=
          dt_spatial_metric.get(m, i) * shift.get(m) +
          spatial_metric.get(m, i) * dt_shift.get(m);
    }
    for (size_t j = i; j < Dim; ++j) {
      spacetime_deriv_spacetime_metric.get(0, i + 1, j + 1) =
          dt_spatial_metric.get(i, j);
    }
  }

  for (size_t k = 0; k < Dim; ++k) {
    spacetime_deriv_spacetime_metric.get(k + 1, 0, 0) =
        -2.0 * lapse.get() * deriv_lapse.get(k);
    for (size_t m = 0; m < Dim; ++m) {
      for (size_t n = 0; n < Dim; ++n) {
        spacetime_deriv_spacetime_metric.get(k + 1, 0, 0) +=
            deriv_spatial_metric.get(k, m, n) * shift.get(m) * shift.get(n) +
            2.0 * spatial_metric.get(m, n) * shift.get(m) *
                deriv_shift.get(k, n);
      }
    }

    for (size_t i = 0; i < Dim; ++i) {
      for (size_t m = 0; m < Dim; ++m) {
        spacetime_deriv_spacetime_metric.get(k + 1, 0, i + 1) +=
            deriv_spatial_metric.get(k, m, i) * shift.get(m) +
            spatial_metric.get(m, i) * deriv_shift.get(k, m);
      }
      for (size_t j = i; j < Dim; ++j) {
        spacetime_deriv_spacetime_metric.get(k + 1, i + 1, j + 1) =
            deriv_spatial_metric.get(k, i, j);
      }
    }
  }

  return spacetime_deriv_spacetime_metric;
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                  \
  template tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>                      \
  compute_spacetime_metric(                                                   \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spatial_metric) noexcept;                                           \
  template tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>                      \
  compute_inverse_spacetime_metric(                                           \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                    \
          inverse_spatial_metric) noexcept;                                   \
  template tnsr::abb<DTYPE(data), DIM(data), FRAME(data)>                     \
  compute_derivatives_of_spacetime_metric(                                    \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse,  \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& deriv_lapse,        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,           \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,       \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                   \
          deriv_spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
