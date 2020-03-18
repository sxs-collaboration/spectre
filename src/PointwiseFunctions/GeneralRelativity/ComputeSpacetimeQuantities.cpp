// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"

#include <cmath>  // IWYU pragma: keep

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_include <complex>

/// \cond
namespace gr {
template <size_t Dim, typename Frame, typename DataType>
void spacetime_metric(
    const gsl::not_null<tnsr::aa<DataType, Dim, Frame>*> spacetime_metric,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric) noexcept {
  if (UNLIKELY(get_size(get<0, 0>(*spacetime_metric)) !=
               get_size(get(lapse)))) {
    *spacetime_metric = tnsr::aa<DataType, Dim, Frame>(get_size(get(lapse)));
  }

  get<0, 0>(*spacetime_metric) = -square(get(lapse));

  for (size_t m = 0; m < Dim; ++m) {
    get<0, 0>(*spacetime_metric) +=
        spatial_metric.get(m, m) * square(shift.get(m));
    for (size_t n = 0; n < m; ++n) {
      get<0, 0>(*spacetime_metric) +=
          2. * spatial_metric.get(m, n) * shift.get(m) * shift.get(n);
    }
  }

  for (size_t i = 0; i < Dim; ++i) {
    spacetime_metric->get(0, i + 1) = 0.;
    for (size_t m = 0; m < Dim; ++m) {
      spacetime_metric->get(0, i + 1) +=
          spatial_metric.get(m, i) * shift.get(m);
    }
    for (size_t j = i; j < Dim; ++j) {
      spacetime_metric->get(i + 1, j + 1) = spatial_metric.get(i, j);
    }
  }
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::aa<DataType, Dim, Frame> spacetime_metric(
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric) noexcept {
  tnsr::aa<DataType, Dim, Frame> spacetime_metric{};
  gr::spacetime_metric<Dim, Frame, DataType>(make_not_null(&spacetime_metric),
                                             lapse, shift, spatial_metric);
  return spacetime_metric;
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> spatial_metric(
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept {
  tnsr::ii<DataType, SpatialDim, Frame> local_spatial_metric{
      get_size(get<0, 0>(spacetime_metric))};
  spatial_metric(make_not_null(&local_spatial_metric), spacetime_metric);
  return local_spatial_metric;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void spatial_metric(
    const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*> spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept {
  destructive_resize_components(spatial_metric,
                                get_size(get<0, 0>(spacetime_metric)));
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      spatial_metric->get(i, j) = spacetime_metric.get(i + 1, j + 1);
    }
  }
}

template <size_t Dim, typename Frame, typename DataType>
tnsr::AA<DataType, Dim, Frame> inverse_spacetime_metric(
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric) noexcept {
  tnsr::AA<DataType, Dim, Frame> inv_spacetime_metric{};
  inverse_spacetime_metric(make_not_null(&inv_spacetime_metric), lapse, shift,
                           inverse_spatial_metric);
  return inv_spacetime_metric;
}

template <size_t Dim, typename Frame, typename DataType>
void inverse_spacetime_metric(
    const gsl::not_null<tnsr::AA<DataType, Dim, Frame>*>
        inverse_spacetime_metric,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric) noexcept {
  get<0, 0>(*inverse_spacetime_metric) = -1.0 / (get(lapse) * get(lapse));

  const auto& minus_one_over_lapse_sqrd = get<0, 0>(*inverse_spacetime_metric);

  for (size_t i = 0; i < Dim; ++i) {
    inverse_spacetime_metric->get(0, i + 1) =
        -shift.get(i) * minus_one_over_lapse_sqrd;
  }

  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      inverse_spacetime_metric->get(i + 1, j + 1) =
          inverse_spatial_metric.get(i, j) +
          shift.get(i) * shift.get(j) * minus_one_over_lapse_sqrd;
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::I<DataType, SpatialDim, Frame> shift(
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
    const tnsr::II<DataType, SpatialDim, Frame>&
        inverse_spatial_metric) noexcept {
  tnsr::I<DataType, SpatialDim, Frame> local_shift{get_size(
      get<0, 0>(spacetime_metric))};
  shift(make_not_null(&local_shift), spacetime_metric, inverse_spatial_metric);
  return local_shift;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void shift(const gsl::not_null<tnsr::I<DataType, SpatialDim, Frame>*> shift,
           const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric,
           const tnsr::II<DataType, SpatialDim, Frame>&
               inverse_spatial_metric) noexcept {
  destructive_resize_components(shift, get_size(get<0, 0>(spacetime_metric)));
  for (size_t i = 0; i < SpatialDim; ++i) {
    shift->get(i) =
        inverse_spatial_metric.get(i, 0) * get<1, 0>(spacetime_metric);
    for (size_t j = 1; j < SpatialDim; ++j) {
      shift->get(i) +=
          inverse_spatial_metric.get(i, j) * spacetime_metric.get(j + 1, 0);
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> lapse(
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept {
  Scalar<DataType> local_lapse{get_size(get<0, 0>(spacetime_metric))};
  lapse(make_not_null(&local_lapse), shift, spacetime_metric);
  return local_lapse;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void lapse(
    const gsl::not_null<Scalar<DataType>*> lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::aa<DataType, SpatialDim, Frame>& spacetime_metric) noexcept {
  destructive_resize_components(lapse, get_size(get<0, 0>(spacetime_metric)));
  get(*lapse) = -get<0, 0>(spacetime_metric);
  for (size_t i = 0; i < SpatialDim; ++i) {
    get(*lapse) += shift.get(i) * spacetime_metric.get(i + 1, 0);
  }
  get(*lapse) = sqrt(get(*lapse));
}

template <size_t SpatialDim, typename Frame, typename DataType>
void time_derivative_of_spacetime_metric(
    const gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame>*>
        dt_spacetime_metric,
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric) noexcept {
  destructive_resize_components(dt_spacetime_metric, get_size(get(dt_lapse)));
  get<0, 0>(*dt_spacetime_metric) = -2.0 * get(lapse) * get(dt_lapse);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = 0; j < SpatialDim; ++j) {
      get<0, 0>(*dt_spacetime_metric) +=
          dt_spatial_metric.get(i, j) * shift.get(i) * shift.get(j) +
          2.0 * spatial_metric.get(i, j) * shift.get(i) * dt_shift.get(j);
    }
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    dt_spacetime_metric->get(0, i + 1) =
        dt_spatial_metric.get(i, 0) * get<0>(shift) +
        spatial_metric.get(i, 0) * get<0>(dt_shift);
    for (size_t j = 1; j < SpatialDim; ++j) {
      dt_spacetime_metric->get(0, i + 1) +=
          dt_spatial_metric.get(i, j) * shift.get(j) +
          spatial_metric.get(i, j) * dt_shift.get(j);
    }
  }

  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      dt_spacetime_metric->get(i + 1, j + 1) = dt_spatial_metric.get(i, j);
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::aa<DataType, SpatialDim, Frame> time_derivative_of_spacetime_metric(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric) noexcept {
  tnsr::aa<DataType, SpatialDim, Frame> dt_spacetime_metric{
      get_size(get(lapse))};
  time_derivative_of_spacetime_metric(make_not_null(&dt_spacetime_metric),
                                      lapse, dt_lapse, shift, dt_shift,
                                      spatial_metric, dt_spatial_metric);
  return dt_spacetime_metric;
}


template <size_t Dim, typename Frame, typename DataType>
tnsr::abb<DataType, Dim, Frame> derivatives_of_spacetime_metric(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::i<DataType, Dim, Frame>& deriv_lapse,
    const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::I<DataType, Dim, Frame>& dt_shift,
    const tnsr::iJ<DataType, Dim, Frame>& deriv_shift,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric,
    const tnsr::ii<DataType, Dim, Frame>& dt_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& deriv_spatial_metric) noexcept {
  auto spacetime_deriv_spacetime_metric =
      make_with_value<tnsr::abb<DataType, Dim, Frame>>(lapse, 0.);

  get<0, 0, 0>(spacetime_deriv_spacetime_metric) =
      -2.0 * get(lapse) * get(dt_lapse);

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
        -2.0 * get(lapse) * deriv_lapse.get(k);
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
tnsr::a<DataType, SpatialDim, Frame> spacetime_normal_one_form(
    const Scalar<DataType>& lapse) noexcept {
  auto normal_one_form =
      make_with_value<tnsr::a<DataType, SpatialDim, Frame>>(lapse, 0.);
  get<0>(normal_one_form) = -get(lapse);
  return normal_one_form;
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::A<DataType, SpatialDim, Frame> spacetime_normal_vector(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift) noexcept {
  tnsr::A<DataType, SpatialDim, Frame> local_spacetime_normal_vector{
      get_size(get(lapse))};
  spacetime_normal_vector(make_not_null(&local_spacetime_normal_vector), lapse,
                          shift);
  return local_spacetime_normal_vector;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void spacetime_normal_vector(
    const gsl::not_null<tnsr::A<DataType, SpatialDim, Frame>*>
        spacetime_normal_vector,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift) noexcept {
  destructive_resize_components(spacetime_normal_vector, get_size(get(lapse)));
  get<0>(*spacetime_normal_vector) = 1. / get(lapse);
  for (size_t i = 0; i < SpatialDim; i++) {
    spacetime_normal_vector->get(i + 1) =
        -shift.get(i) * get<0>(*spacetime_normal_vector);
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> extrinsic_curvature(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iJ<DataType, SpatialDim, Frame>& deriv_shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::ii<DataType, SpatialDim, Frame>& dt_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>&
        deriv_spatial_metric) noexcept {
  const DataType half_over_lapse = 0.5 / get(lapse);

  auto ex_curvature =
      make_with_value<tnsr::ii<DataType, SpatialDim, Frame>>(lapse, 0.0);
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {  // Symmetry
      for (size_t k = 0; k < SpatialDim; ++k) {
        ex_curvature.get(i, j) +=
            shift.get(k) * deriv_spatial_metric.get(k, i, j) +
            spatial_metric.get(k, i) * deriv_shift.get(j, k) +
            spatial_metric.get(k, j) * deriv_shift.get(i, k);
      }
      ex_curvature.get(i, j) -= dt_spatial_metric.get(i, j);
      ex_curvature.get(i, j) *= half_over_lapse;
    }
  }

  return ex_curvature;
}
}  // namespace gr

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                   \
  template void gr::spacetime_metric(                                          \
      const gsl::not_null<tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>*>      \
          spacetime_metric,                                                    \
      const Scalar<DTYPE(data)>& lapse,                                        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,               \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spatial_metric) noexcept;                                            \
  template tnsr::aa<DTYPE(data), DIM(data), FRAME(data)> gr::spacetime_metric( \
      const Scalar<DTYPE(data)>& lapse,                                        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,               \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spatial_metric) noexcept;                                            \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)> gr::spatial_metric(   \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_metric) noexcept;                                          \
  template void gr::inverse_spacetime_metric(                                  \
      const gsl::not_null<tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>*>      \
          inv_spacetime_metric,                                                \
      const Scalar<DTYPE(data)>& lapse,                                        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,               \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                     \
          inverse_spatial_metric) noexcept;                                    \
  template tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>                       \
  gr::inverse_spacetime_metric(                                                \
      const Scalar<DTYPE(data)>& lapse,                                        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,               \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                     \
          inverse_spatial_metric) noexcept;                                    \
  template tnsr::I<DTYPE(data), DIM(data), FRAME(data)> gr::shift(             \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& spacetime_metric,   \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                     \
          inverse_spatial_metric) noexcept;                                    \
  template Scalar<DTYPE(data)> gr::lapse(                                      \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,               \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>&                     \
          spacetime_metric) noexcept;                                          \
  template tnsr::abb<DTYPE(data), DIM(data), FRAME(data)>                      \
  gr::derivatives_of_spacetime_metric(                                         \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse,   \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& deriv_lapse,         \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,               \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,            \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,        \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,     \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric,  \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                    \
          deriv_spatial_metric) noexcept;                                      \
  template tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>                       \
  gr::time_derivative_of_spacetime_metric(                                     \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse,   \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,               \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,            \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,     \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                     \
          dt_spatial_metric) noexcept;                                         \
  template void gr::time_derivative_of_spacetime_metric(                       \
      const gsl::not_null<tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>*>      \
          dt_spacetime_metric,                                                 \
      const Scalar<DTYPE(data)>& lapse, const Scalar<DTYPE(data)>& dt_lapse,   \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,               \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& dt_shift,            \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,     \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                     \
          dt_spatial_metric) noexcept;                                         \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                        \
  gr::spacetime_normal_one_form(const Scalar<DTYPE(data)>& lapse) noexcept;    \
  template tnsr::A<DTYPE(data), DIM(data), FRAME(data)>                        \
  gr::spacetime_normal_vector(                                                 \
      const Scalar<DTYPE(data)>& lapse,                                        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift) noexcept;     \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                       \
  gr::extrinsic_curvature(                                                     \
      const Scalar<DTYPE(data)>& lapse,                                        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,               \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,        \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric,     \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& dt_spatial_metric,  \
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                    \
          deriv_spatial_metric) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
