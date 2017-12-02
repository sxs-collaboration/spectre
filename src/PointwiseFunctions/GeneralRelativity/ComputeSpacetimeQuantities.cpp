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

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iaa<DataType, SpatialDim, Frame> compute_phi(
    const Scalar<DataType>& lapse,
    const tnsr::i<DataType, SpatialDim, Frame>& deriv_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>&
        deriv_spatial_metric) noexcept {
  tnsr::iaa<DataType, SpatialDim, Frame> phi(
      make_with_value<DataType>(*deriv_lapse.begin(), 0.));
  for (size_t k = 0; k < SpatialDim; ++k) {
    phi.get(k, 0, 0) = -2.0 * lapse.get() * deriv_lapse.get(k);
    for (size_t m = 0; m < SpatialDim; ++m) {
      for (size_t n = 0; n < SpatialDim; ++n) {
        phi.get(k, 0, 0) +=
            deriv_spatial_metric.get(k, m, n) * shift.get(m) * shift.get(n) +
            2.0 * spatial_metric.get(m, n) * shift.get(m) *
                deriv_shift.get(k, n);
      }
    }

    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t m = 0; m < SpatialDim; ++m) {
        phi.get(k, 0, i + 1) +=
            deriv_spatial_metric.get(k, m, i) * shift.get(m) +
            spatial_metric.get(m, i) * deriv_shift.get(k, m);
      }
      for (size_t j = i; j < SpatialDim; ++j) {
        phi.get(k, i + 1, j + 1) = deriv_spatial_metric.get(k, i, j);
      }
    }
  }
  return phi;
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame> compute_pi(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  tnsr::aa<DataType, SpatialDim, Frame> pi{
      make_with_value<DataType>(*lapse.begin(), 0.)};

  pi.get(0, 0) = -2.0 * lapse.get() * dt_lapse.get();

  for (size_t m = 0; m < SpatialDim; ++m) {
    for (size_t n = 0; n < SpatialDim; ++n) {
      pi.get(0, 0) +=
          dt_spatial_metric.get(m, n) * shift.get(m) * shift.get(n) +
          2.0 * spatial_metric.get(m, n) * shift.get(m) * dt_shift.get(n);
    }
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t m = 0; m < SpatialDim; ++m) {
      pi.get(0, i + 1) += dt_spatial_metric.get(m, i) * shift.get(m) +
                          spatial_metric.get(m, i) * dt_shift.get(m);
    }
    for (size_t j = i; j < SpatialDim; ++j) {
      pi.get(i + 1, j + 1) = dt_spatial_metric.get(i, j);
    }
  }
  for (size_t mu = 0; mu < SpatialDim + 1; ++mu) {
    for (size_t nu = mu; nu < SpatialDim + 1; ++nu) {
      for (size_t i = 0; i < SpatialDim; ++i) {
        pi.get(mu, nu) -= shift.get(i) * phi.get(i, mu, nu);
      }
      pi.get(mu, nu) /= -lapse.get();
    }
  }
  return pi;
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> compute_spacetime_normal_one_form(
    const Scalar<DataType>& lapse) noexcept {
  auto normal_one_form =
      make_with_value<tnsr::a<DataType, SpatialDim, Frame>>(get<>(lapse), 0.);
  get<0>(normal_one_form) = -get<>(lapse);
  return normal_one_form;
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
          deriv_spatial_metric) noexcept;                                     \
  template tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)> compute_phi(        \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& deriv_lapse,        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,       \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                   \
          deriv_spatial_metric) noexcept;                                     \
  template tnsr::aa<DTYPE(data), DIM(data), FRAME(data)> compute_pi(          \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse,  \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,           \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,    \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric, \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;    \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                       \
  compute_spacetime_normal_one_form(                                          \
      const Scalar<DTYPE(data)>& lapse) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
